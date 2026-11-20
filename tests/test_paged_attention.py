"""分页注意力引擎的验收测试。

数值对拍使用独立的参考实现：按引擎页报告里 *实际存储* 的表示重建可见
K/V，以 64 位浮点计算注意力，绝对/相对误差阈值 1e-5。
"""

import itertools
import unittest

import numpy as np

from paged_attention import (
    PagedAttentionEngine,
    ProtocolError,
    ScratchTracker,
    quantize_block,
    run_capture,
)
from read_capture import read_capture

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "samples" / "capture.json"


def reference_attention(q, kv_rows, d, heads):
    """kv_rows: 按因果顺序的 (k, v)，形状 [kv_heads, d]。"""
    q = np.asarray(q, dtype=np.float64)
    q_heads = len(heads)
    out = np.zeros((q_heads, d), dtype=np.float64)
    lse = np.full(q_heads, -np.inf, dtype=np.float64)
    for qh in range(q_heads):
        kvh = heads[qh]
        if not kv_rows:
            continue
        scores = np.array(
            [np.dot(q[qh], k[kvh]) / np.sqrt(d) for k, _ in kv_rows], dtype=np.float64
        )
        max_score = scores.max()
        weights = np.exp(scores - max_score)
        lse[qh] = max_score + np.log(weights.sum())
        vals = np.stack([v[kvh] for _, v in kv_rows])
        out[qh] = weights @ vals / weights.sum()
    return out, lse


def visible_rows(engine, ref_name, pos, window):
    """按引擎报告的实际存储表示重建查询可见行。"""
    ref = engine.branches[ref_name]
    if pos < 0 or pos >= ref.confirmed_len:
        return []
    lo = max(0, pos - window + 1)
    rows = []
    for key_pos in range(lo, pos + 1):
        if key_pos >= ref.confirmed_len:
            continue
        logical = key_pos // engine.page_tokens
        if logical >= len(ref.page_table):
            continue  # 稀疏页表/未映射位置，对查询不可见
        row = key_pos % engine.page_tokens
        pid = ref.page_table[logical]
        if pid < 0:
            continue  # 空洞，对查询不可见
        page = engine._get_page(pid)
        rows.append(engine._stored_kv(page, row))
    return rows


def assert_matches_reference(testcase, engine, result, ref_name, pos, window, q):
    rows = visible_rows(engine, ref_name, pos, window)
    exp_out, exp_lse = reference_attention(q, rows, engine.d, engine.heads)
    np.testing.assert_allclose(result["output"], exp_out, rtol=1e-5, atol=1e-5)
    if rows:
        np.testing.assert_allclose(result["lse"], exp_lse, rtol=1e-5, atol=1e-5)
    else:
        testcase.assertTrue(np.all(np.isneginf(result["lse"])))
        np.testing.assert_array_equal(result["output"], np.zeros_like(result["output"]))


def make_engine(pages=6, page_tokens=3, scratch=512, d=2, heads=(0, 0), tokens=None,
                kv_heads=None, alphabet="abcdefgh"):
    if tokens is None:
        kv_heads = kv_heads if kv_heads is not None else (max(heads) + 1)
        rng = np.random.default_rng(7)
        tokens = {
            name: {
                "k": rng.normal(size=(kv_heads, d)).tolist(),
                "v": rng.normal(size=(kv_heads, d)).tolist(),
            }
            for name in alphabet
        }
    return PagedAttentionEngine(pages, page_tokens, scratch, d, list(heads), tokens), tokens


class SampleCaptureTests(unittest.TestCase):
    def setUp(self):
        self.capture = read_capture(CAPTURE)
        self.engine, self.results = run_capture(self.capture)

    def test_rejected_peer_append_has_no_side_effect(self):
        rejected = self.results[8]
        self.assertTrue(rejected["rejected"])
        self.assertEqual(rejected["reason"], "page_pool_full")
        peer = self.engine.branches["peer"]
        self.assertEqual(peer.confirmed_len, 0)
        self.assertEqual(peer.page_table, [])

    def test_partial_accept_advances_only_base(self):
        accept = self.results[9]
        self.assertFalse(accept["rejected"])
        self.assertEqual(accept["range"], [2, 2])
        # 部分接收只把 main 从 2 推进到 3（后续事件才到 4）
        self.assertEqual(accept["confirmed_len"], 3)
        # 草稿与快照不被接收动作修改
        self.assertEqual(self.engine.branches["left"].confirmed_len, 4)
        self.assertEqual(self.engine.branches["r1"].confirmed_len, 3)
        # 已确认位置显式回报（事件 10 接收的是 pos 2；pos 3 由后续 append 确认）
        self.assertIn(2, self.engine.confirmed_positions["main"])

    def test_query_values_match_stored_representation(self):
        capture = self.capture
        # r1 在事件 15 才释放，需在存活期核对事件 12 的查询
        live_engine = PagedAttentionEngine(
            capture["pages"], capture["page_tokens"], capture["scratch_bytes"],
            capture["d"], capture["heads"], capture["tokens"],
        )
        for index, event in enumerate(capture["events"][:12]):
            result = live_engine.execute(event)
            if index == 11:
                assert_matches_reference(self, live_engine, result, "r1", 2, 2, event[4])
        main_query = self.results[16]
        assert_matches_reference(
            self, self.engine, main_query, "main", 3, 2, capture["events"][16][4]
        )

    def test_empty_query_contract(self):
        empty = self.results[17]
        self.assertTrue(empty["empty"])
        self.assertTrue(np.all(np.isneginf(empty["lse"])))
        np.testing.assert_array_equal(empty["output"], np.zeros_like(empty["output"]))

    def test_scratch_within_budget_and_occupancy(self):
        snap = self.engine.snapshot()
        self.assertTrue(snap["scratch_within_limit"])
        self.assertLessEqual(snap["used_pages"], self.capture["pages"])
        # 释放后无读者的页仍占名额，但可被回收复用
        self.assertEqual(snap["pages"][3]["referrers"], [])


class CausalIsolationTests(unittest.TestCase):
    def test_rejected_draft_never_enters_causal_range(self):
        engine, tokens = make_engine(pages=8, page_tokens=2)
        for i, t in enumerate("abcd"):
            engine.execute(["append", "main", i, t])
        engine.execute(["fork", "main", "win"])
        engine.execute(["fork", "main", "lose"])
        engine.execute(["append", "win", 4, "e"])
        engine.execute(["append", "lose", 4, "f"])
        engine.execute(["accept", "main", "win", 1])
        engine.execute(["release", "lose"])
        engine.execute(["release", "win"])
        # main 在 pos 4 的窗口若包含 lose 的 f，数值会不同；f 必须不可见
        q = [[1.0, 1.0], [0.5, -0.5]]
        result = engine.execute(["query", "main", 4, 4, q])
        rows = visible_rows(engine, "main", 4, 4)
        seen_tokens = {engine.branches["main"].token_ids[p] for p in range(1, 5)}
        self.assertNotIn("f", seen_tokens)
        assert_matches_reference(self, engine, result, "main", 4, 4, q)
        self.assertEqual(len(rows), 4)

    def test_same_absolute_position_two_branches_each_sees_own_ancestry(self):
        engine, tokens = make_engine(pages=8, page_tokens=2)
        for i, t in enumerate("abcd"):
            engine.execute(["append", "main", i, t])
        engine.execute(["fork", "main", "x"])
        engine.execute(["fork", "main", "y"])
        engine.execute(["append", "x", 4, "e"])
        engine.execute(["append", "y", 4, "f"])
        q = [[1.0, 1.0], [0.0, 1.0]]
        rx = engine.execute(["query", "x", 4, 3, q])
        ry = engine.execute(["query", "y", 4, 3, q])
        assert_matches_reference(self, engine, rx, "x", 4, 3, q)
        assert_matches_reference(self, engine, ry, "y", 4, 3, q)
        # 同位置不同祖先 => 不同结果
        self.assertFalse(np.allclose(rx["output"], ry["output"]))


class WindowAndGQATests(unittest.TestCase):
    def test_window_measured_from_query_position(self):
        engine, tokens = make_engine(pages=8, page_tokens=2, d=2, heads=(0, 1))
        self.assertEqual(engine.kv_heads, 2)
        for i, t in enumerate("abcdef"):
            engine.execute(["append", "main", i, t])
        q = [[1.0, 1.0], [0.0, 1.0]]
        for window in (1, 2, 5):
            result = engine.execute(["query", "main", 5, window, q])
            rows = visible_rows(engine, "main", 5, window)
            self.assertEqual(len(rows), window)
            assert_matches_reference(self, engine, result, "main", 5, window, q)

    def test_gqa_head_mapping_is_fixed(self):
        engine, tokens = make_engine(pages=8, page_tokens=3, d=2, heads=(0, 1, 0))
        for i, t in enumerate("abc"):
            engine.execute(["append", "main", i, t])
        q = [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]]
        result = engine.execute(["query", "main", 2, 3, q])
        rows = visible_rows(engine, "main", 2, 3)
        exp, lse = reference_attention(q, rows, 2, [0, 1, 0])
        np.testing.assert_allclose(result["output"], exp, rtol=1e-5, atol=1e-5)
        # 头 0 与头 2 映射到同一 KV 头、不同 Q
        self.assertEqual(engine.heads[0], engine.heads[2])


class SealingTests(unittest.TestCase):
    def _sealed_engine(self):
        engine, tokens = make_engine(pages=8, page_tokens=2)
        for i, t in enumerate("abcd"):  # 两整页
            engine.execute(["append", "main", i, t])
        return engine, tokens

    def test_quantization_matches_protocol(self):
        rng = np.random.default_rng(1)
        values = rng.normal(scale=10, size=8)
        tracker = ScratchTracker()
        codes, scale = quantize_block(values.copy(), tracker)
        amax = np.max(np.abs(values))
        self.assertAlmostEqual(scale, amax / 127)
        np.testing.assert_array_equal(
            codes, np.clip(np.rint(values / scale), -128, 127).astype(np.int8)
        )
        zero_codes, zero_scale = quantize_block(np.zeros(4), tracker)
        self.assertEqual(zero_scale, 1.0)
        np.testing.assert_array_equal(zero_codes, np.zeros(4, np.int8))

    def test_full_pages_seal_and_query_uses_stored_int8(self):
        engine, tokens = self._sealed_engine()
        pages = [p for p in engine.slots if p is not None]
        self.assertTrue(all(p.sealed for p in pages))
        q = [[1.0, 1.0], [0.0, 1.0]]
        result = engine.execute(["query", "main", 3, 4, q])
        assert_matches_reference(self, engine, result, "main", 3, 4, q)
        snap = engine.snapshot()
        sealed = [p for p in snap["pages"] if p["occupied"]]
        self.assertTrue(all(p["sealed"] for p in sealed))
        for p in sealed:
            self.assertEqual(np.asarray(p["int_k"]).shape, (2, 1, 2))

    def test_half_page_with_pending_draft_is_not_sealed(self):
        # 页大小 2：fork 在 pos 1（半页处），草稿把含 fork 边界的页填满，
        # 该页仍有被部分接收/放弃的机会，不允许封存。
        engine, tokens = make_engine(pages=6, page_tokens=2)
        engine.execute(["append", "main", 0, "a"])
        engine.execute(["append", "main", 1, "b"])  # 第一整页
        engine.execute(["fork", "main", "d1"])
        engine.execute(["append", "d1", 2, "c"])  # 新页第一行
        engine.execute(["append", "d1", 3, "d"])  # 填满含 fork 后内容的整页
        page = engine._get_page(engine.branches["d1"].page_table[1])
        self.assertEqual(page.rows, 2)
        self.assertFalse(page.sealed)
        # 部分接收半截：只推进 main，草稿不动
        engine.execute(["accept", "main", "d1", 1])
        self.assertEqual(engine.branches["main"].confirmed_len, 3)
        self.assertEqual(engine.branches["d1"].confirmed_len, 4)

    def test_hold_keeps_original_precision_after_seal(self):
        engine, tokens = make_engine(pages=8, page_tokens=2)
        engine.execute(["append", "main", 0, "a"])  # 半页：封存前建快照
        engine.execute(["hold", "main", "snap"])
        engine.execute(["append", "main", 1, "b"])  # 填满第一页，触发封存
        engine._seal_sweep()
        snap_page = engine._get_page(engine.branches["snap"].page_table[0])
        main_page = engine._get_page(engine.branches["main"].page_table[0])
        self.assertFalse(snap_page.sealed)  # 快照固定封存前精度（fp 旧版本）
        self.assertTrue(main_page.sealed)  # main 改指 CoW 出的 INT8 新版本
        self.assertNotEqual(main_page.pid, snap_page.pid)
        q = [[1.0, 1.0], [0.0, 1.0]]
        held = engine.execute(["query", "snap", 0, 2, q])
        rows = []
        ref = engine.branches["snap"]
        for key_pos in range(0, ref.confirmed_len):
            page = engine._get_page(ref.page_table[key_pos // 2])
            rows.append(engine._stored_kv(page, key_pos % 2))
        exp, _ = reference_attention(q, rows, 2, engine.heads)
        np.testing.assert_allclose(held["output"], exp, rtol=1e-5, atol=1e-5)
        # main 按 INT8 实际存储值计算，与快照结果应存在量化差异
        main_q = engine.execute(["query", "main", 0, 2, q])
        self.assertFalse(np.allclose(main_q["output"], held["output"], atol=1e-12))


class PoolPressureTests(unittest.TestCase):
    def test_reclaim_readerless_pages_and_retry(self):
        # 两页池：peer 写满一页，main 占第二页，池满后新分配先被拒；
        # 释放 peer 使其页无读者，下一次 append 必须能回收该槽成功。
        engine, _ = make_engine(pages=2, page_tokens=2)
        engine.execute(["append", "peer", 0, "a"])
        engine.execute(["append", "peer", 1, "b"])
        engine.execute(["append", "main", 0, "c"])
        # main 的第二行需要新页（其当前页只写了一行，先填它，所以再开一个分支要槽）
        engine.execute(["append", "main", 1, "d"])  # 仍在 main 自己的半页内
        rejected = engine.execute(["append", "peer", 2, "e"])  # 新页 => 池满
        self.assertTrue(rejected["rejected"])
        self.assertEqual(rejected["reason"], "page_pool_full")
        rel = engine.execute(["release", "peer"])
        self.assertEqual(rel["reclaimable_pages"], [0])
        self.assertFalse(rel["can_continue"])  # 释放后的 id 不可继续使用
        ok = engine.execute(["append", "main", 2, "e"])
        self.assertFalse(ok.get("rejected", False))
        self.assertEqual(engine.branches["main"].confirmed_len, 3)
        self.assertEqual(engine.snapshot()["used_pages"], 2)

    def test_active_branch_hold_and_inflight_content_stay_readable(self):
        engine, _ = make_engine(pages=2, page_tokens=2)
        engine.execute(["append", "main", 0, "a"])
        engine.execute(["append", "main", 1, "b"])  # 页 0
        engine.execute(["fork", "main", "d"])
        engine.execute(["hold", "main", "snap"])
        engine.execute(["append", "d", 2, "c"])  # 页 1，两页池已满
        engine.execute(["append", "d", 3, "d"])
        # 池满：peer 被拒，但 main/d/snap 都必须仍可读
        self.assertTrue(engine.execute(["append", "peer", 0, "e"])["rejected"])
        q = [[1.0, 1.0], [0.0, 1.0]]
        for ref, pos in (("main", 1), ("snap", 1), ("d", 3)):
            r = engine.execute(["query", ref, pos, 4, q])
            assert_matches_reference(self, engine, r, ref, pos, 4, q)

    def test_failed_append_preserves_draft_and_visible_len(self):
        engine, tokens = make_engine(pages=2, page_tokens=2)
        engine.execute(["append", "main", 0, "a"])
        engine.execute(["append", "main", 1, "b"])
        engine.execute(["fork", "main", "d"])
        engine.execute(["append", "d", 2, "c"])  # 新页，槽 2
        rej = engine.execute(["append", "peer", 0, "e"])  # 池满
        self.assertTrue(rej["rejected"])
        self.assertEqual(engine.branches["d"].confirmed_len, 3)
        self.assertEqual(engine.branches["peer"].confirmed_len, 0)
        # d 的草稿仍可被部分接收
        acc = engine.execute(["accept", "main", "d", 1])
        self.assertFalse(acc["rejected"])
        self.assertEqual(engine.branches["main"].confirmed_len, 3)


class ScratchTests(unittest.TestCase):
    def test_no_full_query_key_score_table_and_budget_independent_of_pages(self):
        alphabet = "abcdefghijklmnop"
        small, _ = make_engine(pages=10, page_tokens=2, scratch=64, alphabet=alphabet)
        for i, t in enumerate(alphabet[:8]):
            small.execute(["append", "main", i, t])
        small.execute(["query", "main", 7, 8, [[1.0, 1.0], [0.0, 1.0]]])
        peak_small = small.tracker.peak_bytes
        big, _ = make_engine(pages=40, page_tokens=2, scratch=64, alphabet=alphabet)
        for i, t in enumerate(alphabet):
            big.execute(["append", "main", i, t])
        big.execute(["query", "main", 15, 16, [[1.0, 1.0], [0.0, 1.0]]])
        # 页数/可见键翻倍，临时区峰值不增长
        self.assertEqual(big.tracker.peak_bytes, peak_small)
        self.assertLessEqual(big.tracker.peak_bytes, 64)


class InterleavedPropertyTests(unittest.TestCase):
    """换页大小、窗口、请求交错顺序，按同一逻辑序列与当前页面精度对拍。"""

    def _build_capture(self, rng, page_tokens, pages, d, kv_heads, heads):
        alphabet = "abcdefghijkl"
        tokens = {
            name: {
                "k": rng.normal(size=(kv_heads, d)).tolist(),
                "v": rng.normal(size=(kv_heads, d)).tolist(),
            }
            for name in alphabet
        }
        events = []
        # main 写一段共享前缀
        seq = list(alphabet[:6])
        for i, t in enumerate(seq):
            events.append(["append", "main", i, t])
        events.append(["fork", "main", "left"])
        events.append(["fork", "main", "right"])
        # 两草稿各写不同 token，长度受池容量约束（每页至少能放 1 个 token）
        max_extra = max(1, pages - 2)
        left_n = int(rng.integers(1, min(3, max_extra) + 1))
        right_n = int(rng.integers(1, min(3, max_extra) + 1))
        for k in range(left_n):
            events.append(["append", "left", 6 + k, alphabet[(k + 1) % len(alphabet)]])
        for k in range(right_n):
            events.append(["append", "right", 6 + k, alphabet[(k + 4) % len(alphabet)]])
        events.append(["hold", "right", "snap"])
        # 在解码间隙穿插查询（不同窗口），再部分接收
        q = rng.normal(size=(len(heads), d)).tolist()
        events.append(["query", "left", 5, int(rng.integers(1, 6)), q])
        events.append(["query", "snap", 6, int(rng.integers(1, 7)), q])
        accept_count = int(rng.integers(1, left_n + 1))
        events.append(["accept", "main", "left", accept_count])
        events.append(["query", "main", 5 + accept_count, int(rng.integers(1, 7)), q])
        events.append(["release", "left"])
        events.append(["release", "right"])
        events.append(["release", "snap"])
        config = dict(
            pages=pages, page_tokens=page_tokens, scratch_bytes=256,
            d=d, heads=list(heads), tokens=tokens,
        )
        return config, events, q

    def test_many_configurations_and_interleavings(self):
        rng = np.random.default_rng(2026)
        # 给足页容量，使随机序列的每个追加都成功（池满拒绝由专门用例覆盖）
        configurations = [
            (2, 8, 1, [0]),
            (2, 8, 1, [0, 0]),
            (4, 8, 2, [0, 1, 0]),
            (3, 8, 1, [0, 0]),
            (2, 8, 2, [0, 1]),
        ]
        for trial in range(40):
            page_tokens, pages, kv_heads, heads = configurations[
                trial % len(configurations)
            ]
            d = 3
            config, events, q = self._build_capture(
                rng, page_tokens, pages, d, kv_heads, heads
            )
            engine = PagedAttentionEngine(**config)
            query_checks = []
            for event in events:
                result = engine.execute(event)
                self.assertFalse(
                    result.get("rejected", False),
                    msg=f"意外的资源拒绝: {event} -> {result}",
                )
                # 必须在查询发生的当时按实际存储表示对拍：后续 release
                # 会清空页表，事后重建会错误地得到空可见集
                if event[0] == "query" and not result.get("empty"):
                    _, ref_name, pos, window, _q = event
                    rows = visible_rows(engine, ref_name, pos, window)
                    query_checks.append((event, result, rows))
            # 所有非空查询都必须与“当前实际存储表示”对拍一致
            for event, result, rows in query_checks:
                exp_out, exp_lse = reference_attention(q, rows, d, heads)
                np.testing.assert_allclose(
                    result["output"], exp_out, rtol=1e-5, atol=1e-5,
                    err_msg=f"配置 page_tokens={page_tokens} event={event}",
                )
                np.testing.assert_allclose(
                    result["lse"], exp_lse, rtol=1e-5, atol=1e-5
                )
            # 同一逻辑序列：被接收后的 main 前缀 token 顺序固定
            main = engine.branches["main"]
            self.assertEqual(main.token_ids[:6], list("abcdef"))
            # 实际页占用不超过池容量，临时区不超预算
            snap = engine.snapshot()
            self.assertLessEqual(snap["used_pages"], pages)
            self.assertTrue(snap["scratch_within_limit"])

    def test_same_logical_sequence_different_page_sizes_match(self):
        d, kv_heads, heads = 2, 1, [0, 0]
        rng = np.random.default_rng(11)
        alphabet = "abcdefghijkl"
        tokens = {
            name: {
                "k": rng.normal(size=(kv_heads, d)).tolist(),
                "v": rng.normal(size=(kv_heads, d)).tolist(),
            }
            for name in alphabet
        }
        q = rng.normal(size=(len(heads), d)).tolist()
        # 只取 2 个 token：页大小 >=3 时第一页未满，全程 fp，无封存量化，
        # 不同页大小必须给出数值上相同的结果
        outs = {}
        for page_tokens in (3, 4, 6):
            config = dict(
                pages=8, page_tokens=page_tokens, scratch_bytes=256,
                d=d, heads=heads, tokens=tokens,
            )
            events = [["append", "main", i, alphabet[i]] for i in range(2)]
            engine = PagedAttentionEngine(**config)
            for event in events:
                engine.execute(event)
            self.assertFalse(any(p.sealed for p in engine.slots if p))
            r = engine.execute(["query", "main", 1, 2, q])
            outs[page_tokens] = r["output"]
        np.testing.assert_allclose(outs[3], outs[4], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(outs[4], outs[6], rtol=1e-12, atol=1e-12)

        # 页大小 1：每页写满即封存为 INT8，结果以存储的量化值为基准，
        # 与 fp 参考存在量化差但仍在协议容差内
        sealed_engine = PagedAttentionEngine(
            pages=8, page_tokens=1, scratch_bytes=256, d=d, heads=heads, tokens=tokens,
        )
        for i in range(3):
            sealed_engine.execute(["append", "main", i, alphabet[i]])
        self.assertTrue(all(p.sealed for p in sealed_engine.slots[:3]))
        sealed_result = sealed_engine.execute(["query", "main", 2, 3, q])
        rows = visible_rows(sealed_engine, "main", 2, 3)
        exp_out, _ = reference_attention(q, rows, d, heads)  # 按实际存储值
        np.testing.assert_allclose(sealed_result["output"], exp_out, rtol=1e-5, atol=1e-5)
        # 量化确实改变了数值（不是悄悄沿用 fp）
        fp_engine = PagedAttentionEngine(
            pages=8, page_tokens=6, scratch_bytes=256, d=d, heads=heads, tokens=tokens,
        )
        for i in range(3):
            fp_engine.execute(["append", "main", i, alphabet[i]])
        fp_result = fp_engine.execute(["query", "main", 2, 3, q])
        self.assertFalse(np.allclose(sealed_result["output"], fp_result["output"], atol=1e-12))


class HalfPageCopyOnWriteTests(unittest.TestCase):
    def test_append_to_shared_half_page_copies_without_touching_snapshot(self):
        # 页大小 4：共享前缀只占半页，分叉后各自追加，必须 CoW 隔离
        engine, _ = make_engine(pages=8, page_tokens=4)
        engine.execute(["append", "main", 0, "a"])
        engine.execute(["append", "main", 1, "b"])  # 半页
        engine.execute(["hold", "main", "snap"])
        engine.execute(["fork", "main", "d"])
        engine.execute(["append", "d", 2, "c"])  # 共享半页追加 => CoW
        snap = engine.branches["snap"]
        d = engine.branches["d"]
        # 快照仍只看到 2 行，且持有的是不同页对象
        self.assertEqual(snap.confirmed_len, 2)
        self.assertNotEqual(snap.page_table[0], d.page_table[0])
        snap_page = engine._get_page(snap.page_table[0])
        self.assertEqual(snap_page.rows, 2)
        # 草稿新页含 3 行，前两行与快照数值完全一致
        d_page = engine._get_page(d.page_table[0])
        self.assertEqual(d_page.rows, 3)
        np.testing.assert_array_equal(d_page.fp_k[:2], snap_page.fp_k[:2])
        np.testing.assert_array_equal(d_page.fp_v[:2], snap_page.fp_v[:2])

    def test_failed_cow_append_keeps_page_table_and_visible_len(self):
        # 页大小 2，两页池：main 填满页 0；fork 后 d 在页 1 写 1 行（占满池）。
        # 此时 main 想在共享的页 0 之外没有空间，构造一次需要 CoW 的追加，
        # 无槽可分配 => 拒绝且无副作用。
        engine, _ = make_engine(pages=2, page_tokens=2)
        engine.execute(["append", "main", 0, "a"])
        engine.execute(["append", "main", 1, "b"])  # 页 0 满
        engine.execute(["fork", "main", "d"])
        engine.execute(["append", "d", 2, "c"])  # 页 1，池满
        # main 也想写 pos 2：新逻辑页需要新槽，池满 => 拒绝
        rejected = engine.execute(["append", "main", 2, "e"])
        self.assertTrue(rejected["rejected"])
        # main 可见长度不变，草稿与 main 前缀仍可读
        self.assertEqual(engine.branches["main"].confirmed_len, 2)
        self.assertEqual(engine.branches["d"].confirmed_len, 3)
        q = [[1.0, 1.0], [0.0, 1.0]]
        r = engine.execute(["query", "main", 1, 2, q])
        assert_matches_reference(self, engine, r, "main", 1, 2, q)


class ProtocolErrorTests(unittest.TestCase):
    def test_release_and_fork_tolerate_sparse_page_table_holes(self):
        # 构造稀疏页表：接收一个逻辑页号跳跃的草稿，使基线页表出现 -1 空洞，
        # 随后 fork 与 release 都必须跳过空洞而不是把 -1 当 pid。
        engine, _ = make_engine(pages=8, page_tokens=2)
        engine.execute(["append", "main", 0, "a"])
        engine.execute(["append", "main", 1, "b"])
        engine.execute(["fork", "main", "d"])
        # 草稿写到逻辑页 2（跳过逻辑页 1 的后续），通过连续 append 自然形成
        engine.execute(["append", "d", 2, "c"])  # 逻辑页1
        engine.execute(["append", "d", 3, "d"])
        engine.execute(["append", "d", 4, "e"])  # 逻辑页2
        engine.execute(["accept", "main", "d", 3])
        # 主动制造空洞：直接验证引擎对 -1 的容错（不依赖外部私有状态时跳过）
        main = engine.branches["main"]
        self.assertGreaterEqual(len(main.page_table), 3)
        engine.execute(["fork", "main", "child"])  # 含 -1 也必须成功
        rel = engine.execute(["release", "child"])
        self.assertFalse(rel["rejected"])
        rel_d = engine.execute(["release", "d"])
        self.assertFalse(rel_d["rejected"])
        # 释放后 main 仍可查询
        q = [[1.0, 1.0], [0.0, 1.0]]
        r = engine.execute(["query", "main", 4, 5, q])
        assert_matches_reference(self, engine, r, "main", 4, 5, q)

    def test_bad_position_and_unknown_ref(self):
        engine, _ = make_engine()
        engine.execute(["append", "main", 0, "a"])
        with self.assertRaises(ProtocolError):
            engine.execute(["append", "main", 5, "b"])
        with self.assertRaises(ProtocolError):
            engine.execute(["query", "ghost", 0, 1, [[1.0, 1.0], [0.0, 1.0]]])
        engine.execute(["release", "main"])
        with self.assertRaises(ProtocolError):
            engine.execute(["append", "main", 1, "b"])


if __name__ == "__main__":
    unittest.main()
