"""交接协议分页注意力引擎的验收测试。

覆盖：样例采集回放、float64/INT8 对拍、GQA、绝对位置因果与窗口、
分支隔离、hold 固定内容与精度、页池耗尽的原子拒绝与释放恢复、
accept 不泄漏被否决 token、scratch 预算与分块、长提示有界临时区，
以及跨页大小/窗口/交错顺序的随机对拍。
"""

import json
import math
import unittest
from pathlib import Path

import numpy as np

from paged_attention import (
    HandoffEngine,
    reference_attention,
    run_capture,
    quantize_symmetric_int8,
    symmetric_int8_scale,
)

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "samples" / "capture.json"


def load_sample():
    return json.loads(CAPTURE.read_text(encoding="utf-8"))


def tok_seq(cap, ids):
    return [cap["tokens"][t] for t in ids]


def oracle_for_seq(seq, q, heads, d, pos, w):
    keys = [np.asarray(t["k"], dtype=np.float64) for t in seq]
    vals = [np.asarray(t["v"], dtype=np.float64) for t in seq]
    return reference_attention(keys, vals, q, heads, d,
                               window=w, pos=pos)


class SampleReplayTests(unittest.TestCase):
    def setUp(self):
        self.cap = load_sample()

    def test_receipts_resource_and_positions(self):
        r = run_capture(self.cap)
        status = [(e.index, e.op, e.status, e.get("error"))
                  for e in r.events]
        # 第 9 个事件（0-based 8）peer 追加被页池容量拒绝。
        peer_append = r.events[8]
        self.assertFalse(peer_append.ok)
        self.assertEqual(peer_append.error, "pages_exhausted")
        self.assertEqual(peer_append.visible_length, 0)
        # 重复的 main@3 在已经有 pos3 后属于位置冲突，而不是资源问题。
        dup = r.events[15]
        self.assertFalse(dup.ok)
        self.assertEqual(dup.error, "position_exists")
        self.assertTrue(dup.already_present)
        self.assertEqual(dup.next_position, 4)
        # accept(main,left,1) 确认到 pos2（长度 3）。
        acc = r.events[9]
        self.assertTrue(acc.ok)
        self.assertEqual(acc.confirmed_positions, [2])
        self.assertEqual(acc.confirmed_through, 2)
        self.assertEqual(r.confirmed["main"], 3)
        # 释放 r1 后报告明确还有空闲页、可以继续。
        rel_r1 = r.events[14]
        self.assertTrue(rel_r1.ok)
        self.assertGreaterEqual(rel_r1.free_pages, 1)
        self.assertTrue(rel_r1.can_continue)

    def test_occupancy_and_scratch_protocol(self):
        r = run_capture(self.cap)
        self.assertLessEqual(r.occupancy.occupied_pages,
                             r.config["pages"])
        self.assertEqual(r.occupancy.capacity, 4)
        self.assertEqual(r.occupancy.peak_occupied_pages, 4)
        # 每个存活页都在报告里，且 slot 唯一。
        self.assertEqual(len(r.pages), r.occupancy.occupied_pages)
        slots = [p["slot"] for p in r.pages]
        self.assertEqual(len(slots), len(set(slots)))
        # 样例几何 H=2,KV=1,d=2,P=3：整页 tile 恰为 192 字节。
        self.assertEqual(r.scratch.canonical_page_tile_bytes, 192)
        self.assertEqual(r.scratch.peak_bytes, 192)
        self.assertTrue(r.scratch.within_budget)

    def test_sample_float_query_values(self):
        # 关闭自动封存 -> 全部按原始 float64 对拍。
        r = run_capture(self.cap, auto_seal=False)
        q1, q2, q3 = r.queries
        qA = np.array([[1, 2], [-1, 0]], dtype=np.float64)
        o1, l1 = oracle_for_seq(
            tok_seq(self.cap, ["a", "b", "d"]),
            qA, [0, 0], 2, 2, 2)
        np.testing.assert_allclose(q1.values, o1, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(q1.lse, l1, atol=1e-10, rtol=1e-10)
        o2, l2 = oracle_for_seq(
            tok_seq(self.cap, ["a", "b", "c", "e"]),
            qA, [0, 0], 2, 3, 2)
        np.testing.assert_allclose(q2.values, o2, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(q2.lse, l2, atol=1e-10, rtol=1e-10)
        # 空键：零向量 + -inf LSE。
        self.assertEqual(q3.key_count, [0, 0])
        np.testing.assert_array_equal(q3.values, np.zeros((2, 2)))
        self.assertTrue(np.all(np.isneginf(np.asarray(q3.lse))))

    def test_sample_int8_published_variant(self):
        r = run_capture(self.cap)
        q2 = r.queries[1]
        # 主口径为旧表示 float；同时报告中确实存在已封存的 INT8 页。
        self.assertEqual(q2.precision, "float64")
        self.assertTrue(any(p["sealed"] for p in r.pages))
        self.assertTrue(np.all(np.isfinite(np.asarray(q2.lse_published))))
        # float 与 INT8 发布口径不同（误差远大于 1e-5）。
        self.assertFalse(np.allclose(q2.lse, q2.lse_published, atol=1e-5))
        # 以发布口径重放时，主结果即 symmetric_int8。
        rp = run_capture(self.cap, branch_precision="published",
                         compute_alternates=False)
        self.assertIn("symmetric_int8", rp.queries[1].representations)


class GeometryAndMaskTests(unittest.TestCase):
    def _engine(self, heads, kvh, d=3, P=4, pages=8, seed=0):
        rng = np.random.default_rng(seed)
        eng = HandoffEngine(pages=pages, page_tokens=P, d=d, heads=heads,
                            kv_heads=kvh, scratch_bytes=None,
                            auto_seal=False, compute_alternates=False)
        for i in range(12):
            k = rng.standard_normal((kvh, d)).round(3)
            v = rng.standard_normal((kvh, d)).round(3)
            eng.tokens[f"t{i}"] = {"k": k.tolist(), "v": v.tolist()}
            rr = eng.append("main", i, f"t{i}")
            assert rr.ok, rr.get("error")
        return eng

    def test_gqa_grouping(self):
        # 4 个 Q 头映射到 2 个 KV 头，组内结果必须相同（Q 相同）。
        heads = [0, 0, 1, 1]
        eng = self._engine(heads, 2, d=3, seed=1)
        q = np.tile(np.array([[0.7, -1.1, 0.2]]), (4, 1))
        r = eng.query("main", 9, 5, q)
        self.assertTrue(r.ok)
        np.testing.assert_allclose(r.values[0], r.values[1], atol=1e-12)
        np.testing.assert_allclose(r.values[2], r.values[3], atol=1e-12)
        self.assertFalse(np.allclose(r.values[0], r.values[2]))

    def test_window_and_causal_absolute_positions(self):
        eng = self._engine([0, 0], 1, d=2, P=3, seed=2)
        q = np.array([[1.0, 0.0], [0.0, 1.0]])
        # pos=8, window=3 -> [6,7,8]
        r = eng.query("main", 8, 3, q)
        self.assertEqual(sorted(r.positions[0]), [6, 7, 8])
        # 窗口从查询位置算；更大的 pos 不会看到未来。
        r0 = eng.query("main", 5, 2, q)
        self.assertEqual(sorted(r0.positions[0]), [4, 5])
        # pos=-1 显式为空。
        re = eng.query("main", -1, 1, q)
        self.assertEqual(re.key_count, [0, 0])
        self.assertTrue(np.all(np.isneginf(np.asarray(re.lse))))

    def test_per_head_windows(self):
        eng = self._engine([0, 0], 1, d=2, P=3, seed=3)
        q = np.array([[1.0, 0.0], [0.0, 1.0]])
        r = eng.query("main", 8, [2, 4], q)
        self.assertEqual(sorted(r.positions[0]), [7, 8])
        self.assertEqual(sorted(r.positions[1]), [5, 6, 7, 8])


class IsolationAndLifecycleTests(unittest.TestCase):
    def test_two_drafts_same_position_isolated(self):
        cap = load_sample()
        eng = HandoffEngine.from_capture(cap, auto_seal=False,
                                         compute_alternates=False)
        for i, ev in enumerate(cap["events"][:8]):  # 到 hold right 为止
            eng.event_index = i
            eng.run_event(ev)
        q = np.array([[1.0, 2.0], [-1.0, 0.0]])
        left = eng.query("left", 2, 2, q)
        right = eng.query("r1", 2, 2, q)
        # pos2：left 看到 c，right/r1 看到 d，数值必须不同。
        self.assertFalse(np.allclose(left.values, right.values))
        c = cap["tokens"]["c"]
        d = cap["tokens"]["d"]
        self.assertNotEqual(c["k"], d["k"])

    def test_hold_freezes_content(self):
        eng = HandoffEngine(pages=6, page_tokens=2, d=2, heads=[0, 0],
                            scratch_bytes=None, kv_heads=1,
                            auto_seal=False, compute_alternates=False)
        for i, (a, b) in enumerate([(1, 0), (0, 2)]):
            eng.tokens[f"m{i}"] = {"k": [[a, 0]], "v": [[b, 1]]}
            self.assertTrue(eng.append("main", i, f"m{i}").ok)
        self.assertTrue(eng.hold("main", "snap").ok)
        q = np.array([[1.0, 0.0], [0.0, 1.0]])
        before = eng.query("snap", 1, 2, q)
        # 后续追加 / 封存 / 释放 main 都不能改动快照。
        eng.tokens["m2"] = {"k": [[5.0, 5.0]], "v": [[9.0, 9.0]]}
        self.assertTrue(eng.append("main", 2, "m2").ok)
        eng.seal_eligible_pages()
        self.assertTrue(eng.release("main").ok)
        after = eng.query("snap", 1, 2, q)
        np.testing.assert_array_equal(before.values, after.values)
        np.testing.assert_array_equal(before.lse, after.lse)

    def test_half_page_partial_accept_protects_snapshot(self):
        P = 4  # 已确认前缀长度 2，正好落在半页
        eng = HandoffEngine(pages=8, page_tokens=P, d=2, heads=[0, 0],
                            scratch_bytes=None, kv_heads=1,
                            auto_seal=False, compute_alternates=False)
        rng = np.random.default_rng(11)
        for i in range(2):
            eng.tokens[f"p{i}"] = {
                "k": rng.standard_normal((1, 2)).tolist(),
                "v": rng.standard_normal((1, 2)).tolist()}
            eng.append("main", i, f"p{i}")
        eng.fork("main", "dr")
        draft = {}
        for pos in (2, 3, 4):
            spec = {"k": rng.standard_normal((1, 2)).tolist(),
                    "v": rng.standard_normal((1, 2)).tolist()}
            eng.tokens[f"d{pos}"] = spec
            draft[pos] = spec
            eng.append("dr", pos, f"d{pos}")
        # 在部分接收前固定草稿快照。
        self.assertTrue(eng.hold("dr", "view").ok)
        q = rng.standard_normal((2, 2))
        before = eng.query("view", 4, 10, q)
        # 只接收 1 个分叉 token：accepted_len=3，仍在半页内部。
        self.assertTrue(eng.accept("main", "dr", 1).ok)
        # 主流程在 pos3 写入自己的（不同于被否决草稿的）token：
        # 必须 COW，不得改动 view 仍在读的页。
        mine = {"k": rng.standard_normal((1, 2)).tolist(),
                "v": rng.standard_normal((1, 2)).tolist()}
        eng.tokens["mine3"] = mine
        ap = eng.append("main", 3, "mine3")
        self.assertTrue(ap.ok, ap.get("error"))
        after = eng.query("view", 4, 10, q)
        np.testing.assert_array_equal(before.values, after.values)
        np.testing.assert_array_equal(before.lse, after.lse)
        # 快照仍能看到草稿自己的 pos3（被主流程否决的 token）。
        self.assertIn(3, after.positions[0])
        seq_d = [eng.tokens["p0"], eng.tokens["p1"], draft[2],
                 draft[3], draft[4]]
        o, l = oracle_for_seq(seq_d, q, [0, 0], 2, 4, 10)
        np.testing.assert_allclose(after.values, o, atol=1e-10)
        # 主流程 pos3 用的是 mine，而不是 draft[3]。
        rm = eng.query("main", 3, 10, q)
        seq_m = [eng.tokens["p0"], eng.tokens["p1"], draft[2], mine]
        om, lm = oracle_for_seq(seq_m, q, [0, 0], 2, 3, 10)
        np.testing.assert_allclose(rm.values, om, atol=1e-10)

    def test_inflight_query_lease_pins_pages(self):
        eng = HandoffEngine(pages=2, page_tokens=2, d=2, heads=[0, 0],
                            scratch_bytes=None, kv_heads=1,
                            auto_seal=False, compute_alternates=False)
        for i in range(2):
            eng.tokens[f"p{i}"] = {"k": [[float(i + 1), 0.0]],
                                   "v": [[0.0, float(i)]]}
            eng.append("main", i, f"p{i}")
        eng.hold("main", "h")
        q = np.array([[1.0, 0.0], [0.0, 1.0]])
        with eng.begin_query("h", 1, 2, q) as lease:
            eng.release("h")  # 租约仍固定该页
            self.assertEqual(eng.pool.occupied, 1)
            r = lease.run()
            self.assertTrue(r.ok)
            self.assertEqual(r.key_count, [2, 2])
        # 租约关闭后 main 仍是读者，页保持；引用计数一致。
        self.assertEqual(eng.pool.occupied, 1)
        for pid, page in eng.pool.pages.items():
            rc = sum(1 for v in eng.views.values()
                     for e in v.table if e and e[0] == pid)
            self.assertEqual(rc, page.refcount)

    def test_pool_exhaustion_atomic_and_release_recovers(self):
        eng = HandoffEngine(pages=2, page_tokens=2, d=1, heads=[0],
                            scratch_bytes=None, kv_heads=1,
                            auto_seal=False, compute_alternates=False)
        for i in range(2):
            eng.tokens[f"p{i}"] = {"k": [[float(i + 1)]],
                                   "v": [[float(-i)]]}
            self.assertTrue(eng.append("main", i, f"p{i}").ok)
        self.assertTrue(eng.fork("main", "A").ok)
        self.assertTrue(eng.fork("main", "B").ok)
        # A 用 COW 拿到第 2 页并写满 pos2,3。
        for pos in (2, 3):
            t = f"a{pos}"
            eng.tokens[t] = {"k": [[2.0]], "v": [[1.0]]}
            self.assertTrue(eng.append("A", pos, t).ok)
        # B 写 pos2 需要 COW，但两页已满 -> 资源拒绝。
        bt = {"k": [[3.0]], "v": [[2.0]]}
        eng.tokens["b2"] = bt
        r = eng.append("B", 2, "b2")
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "pages_exhausted")
        # 失败不消耗可见长度。
        self.assertEqual(eng.branches["B"].length, 2)
        # 仍可读。
        q = np.array([[1.0]])
        self.assertTrue(eng.query("B", 1, 2, q).ok)
        # 释放 A 的两页引用后有空闲名额，重试成功且可以继续。
        self.assertTrue(eng.release("A").ok)
        retry = eng.append("B", 2, "b2")
        self.assertTrue(retry.ok, retry.get("error"))
        self.assertEqual(eng.branches["B"].length, 3)

    def test_accept_never_leaks_rejected_token(self):
        eng = HandoffEngine(pages=6, page_tokens=4, d=2, heads=[0, 0],
                            scratch_bytes=None, kv_heads=1,
                            auto_seal=False, compute_alternates=False)
        rng = np.random.default_rng(4)
        for i in range(2):
            eng.tokens[f"p{i}"] = {
                "k": rng.standard_normal((1, 2)).tolist(),
                "v": rng.standard_normal((1, 2)).tolist()}
            eng.append("main", i, f"p{i}")
        eng.fork("main", "dr")
        draft_vals = []
        for pos in (2, 3):
            t = f"d{pos}"
            spec = {"k": rng.standard_normal((1, 2)).tolist(),
                    "v": rng.standard_normal((1, 2)).tolist()}
            eng.tokens[t] = spec
            draft_vals.append(spec["v"])
            self.assertTrue(eng.append("dr", pos, t).ok)
        # 只接收 pos2，pos3 被否决。
        self.assertTrue(eng.accept("main", "dr", 1).ok)
        q = rng.standard_normal((2, 2))
        # 即使把查询位置放到 3，可见上界也被长度 3 截断到 pos2。
        r = eng.query("main", 3, 10, q)
        self.assertNotIn(3, r.positions[0])
        self.assertEqual(max(r.positions[0]), 2)
        # 沿胜出前缀写入自己的 pos3，查询必须用新值而不是草稿 d3。
        new = {"k": rng.standard_normal((1, 2)).tolist(),
               "v": rng.standard_normal((1, 2)).tolist()}
        eng.tokens["win"] = new
        self.assertTrue(eng.append("main", 3, "win").ok)
        r2 = eng.query("main", 3, 10, q)
        seq = [eng.tokens["p0"], eng.tokens["p1"], eng.tokens["d2"], new]
        o, l = oracle_for_seq(seq, q, [0, 0], 2, 3, 10)
        np.testing.assert_allclose(r2.values, o, atol=1e-10)
        # 草稿本身未被 accept 修改，仍可见 d3。
        rd = eng.query("dr", 3, 10, q)
        self.assertIn(3, rd.positions[0])


class ScratchAndLongPromptTests(unittest.TestCase):
    def test_scratch_budget_rejects_too_small(self):
        eng = HandoffEngine(pages=4, page_tokens=3, d=2, heads=[0, 0],
                            scratch_bytes=32, kv_heads=1)
        eng.tokens["a"] = {"k": [[1.0, 0.0]], "v": [[0.0, 1.0]]}
        eng.append("main", 0, "a")
        r = eng.query("main", 0, 1, [[1.0, 2.0], [0.0, 0.0]])
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "scratch_exhausted")

    def test_scratch_chunking_matches_oracle(self):
        # 128 字节预算 -> 每 key64B -> 块宽 2（小于整页 3），结果仍正确。
        cap = load_sample()
        r = run_capture(cap, scratch_bytes=128, auto_seal=False)
        self.assertTrue(all(q.ok for q in r.queries))
        self.assertEqual(r.scratch.chunk_keys, 2)
        qA = np.array([[1, 2], [-1, 0]], float)
        o1, l1 = oracle_for_seq(
            tok_seq(cap, ["a", "b", "d"]), qA, [0, 0], 2, 2, 2)
        np.testing.assert_allclose(r.queries[0].values, o1, atol=1e-10)

    def test_long_prompt_bounded_tile(self):
        P, N, d, H, KV = 16, 1024, 8, 4, 2
        rng = np.random.default_rng(7)
        heads = [i % KV for i in range(H)]
        eng = HandoffEngine(pages=N + 2, page_tokens=P, d=d, heads=heads,
                            kv_heads=KV, scratch_bytes=10_000,
                            auto_seal=True, compute_alternates=False)
        for i in range(N):
            eng.tokens[f"t{i}"] = {
                "k": rng.standard_normal((KV, d)).tolist(),
                "v": rng.standard_normal((KV, d)).tolist()}
            self.assertTrue(eng.append("main", i, f"t{i}").ok)
        q = rng.standard_normal((H, d))
        r = eng.query("main", N - 1, 32, q)
        self.assertTrue(r.ok)
        self.assertEqual(sum(r.key_count), H * 32)
        # 分数临时块宽度不超过预算推出的 chunk，而非全部 1024 键。
        bpk = 8 * (2 * H + 2 * KV * d)
        chunk = min(P, 10_000 // bpk)
        self.assertLessEqual(chunk, P)
        self.assertEqual(r.scratch_required_bytes, bpk * chunk)
        self.assertLess(eng.scratch.peak_bytes, bpk * (N + 1))


class QuantizationTests(unittest.TestCase):
    def test_zero_block_scale_one(self):
        q, s, dq = quantize_symmetric_int8(np.zeros((3, 2)))
        self.assertEqual(s, 1.0)
        np.testing.assert_array_equal(q, 0)

    def test_scale_and_roundtrip_baseline(self):
        x = np.array([[1.0, -2.0], [3.0, 0.5]], dtype=np.float64)
        q, s, dq = quantize_symmetric_int8(x)
        self.assertAlmostEqual(s, 3.0 / 127.0, places=12)
        self.assertEqual(int(np.abs(q).max()), 127)
        # 页实际储存基准 = int8 * scale。
        np.testing.assert_allclose(dq, q.astype(float) * s)
        self.assertEqual(symmetric_int8_scale(x), 3.0 / 127.0)


class RandomOracleTests(unittest.TestCase):
    """跨页大小 / 窗口 / 交错的轻量随机对拍（float 路径）。"""

    def test_random_cases(self):
        import random

        def one(seed):
            rng = random.Random(seed)
            npr = np.random.default_rng(seed)
            P = rng.choice([1, 2, 3, 4, 6])
            Np = rng.choice([3, 4, 6, 10])
            H, KV, d = 2, 1, rng.choice([2, 4])
            heads = [0, 0]
            eng = HandoffEngine(pages=Np, page_tokens=P, d=d, heads=heads,
                                scratch_bytes=None, kv_heads=KV,
                                auto_seal=False, compute_alternates=False)
            tokens = {}
            eng.tokens = tokens

            def new_tok():
                tid = f"t{len(tokens)}"
                tokens[tid] = {
                    "k": np.round(npr.standard_normal((KV, d)), 3).tolist(),
                    "v": np.round(npr.standard_normal((KV, d)), 3).tolist()}
                return tid

            base = rng.randrange(0, 2 * P + 1)
            main_seq = []
            for p in range(base):
                t = new_tok()
                self.assertTrue(eng.append("main", p, t).ok)
                main_seq.append(tokens[t])
            self.assertTrue(eng.fork("main", "L").ok)
            seq_l = list(main_seq)
            for i in range(rng.randrange(1, 2 * P + 1)):
                t = new_tok()
                pos = len(seq_l)
                r = eng.append("L", pos, t)
                if r.ok:
                    seq_l.append(tokens[t])
                else:
                    self.assertEqual(r.error, "pages_exhausted")
                    self.assertEqual(len(seq_l), pos)
            if not seq_l:
                return True
            pos = rng.randrange(len(seq_l))
            w = rng.randrange(1, 3 * P + 1)
            q = npr.standard_normal((H, d))
            r = eng.query("L", pos, w, q)
            self.assertTrue(r.ok)
            o, l = oracle_for_seq(seq_l, q, heads, d, pos, w)
            np.testing.assert_allclose(r.values, o, atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(r.lse, l, atol=1e-8, rtol=1e-8)
            # 引用计数 / 占用不变量。
            self.assertLessEqual(eng.pool.occupied, Np)
            for pid, page in eng.pool.pages.items():
                rc = sum(1 for v in eng.views.values()
                         for e in v.table if e and e[0] == pid)
                self.assertEqual(rc, page.refcount)
            return True

        for seed in range(40):
            with self.subTest(seed=seed):
                self.assertTrue(one(seed))


if __name__ == "__main__":
    unittest.main()
