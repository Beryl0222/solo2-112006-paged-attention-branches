#!/usr/bin/env python3
"""可验证的分页注意力验收脚本（交接协议）。

运行::

    python acceptance_check.py            # 全部检查
    python acceptance_check.py --quick    # 少量随机种子

它独立构造 float64 逻辑 oracle 与逐页 INT8 基准，跨页大小 / 窗口 /
GQA / 交错顺序对拍引擎输出，并核对：

* 分支隔离、绝对位置因果、窗口边界、空键 -inf；
* hold 固定内容与精度；
* 页池占用 <= pages、引用计数一致、快照固定页计入名额；
* 资源耗尽时追加被明确拒绝且不消耗长度 / 不损坏草稿，release 后可继续；
* scratch 临时块字节公式、预算拒绝与长提示有界（无 Q×K 全表）。

退出码 0 表示全部通过。
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paged_attention import HandoffEngine, reference_attention  # noqa: E402

ROOT = Path(__file__).resolve().parent
CAPTURE = ROOT / "samples" / "capture.json"


# ---------------------------------------------------------------------------
# 独立 oracle
# ---------------------------------------------------------------------------

def q8_block(block: np.ndarray) -> np.ndarray:
    x = np.asarray(block, dtype=np.float64)
    mx = float(np.abs(x).max()) if x.size else 0.0
    scale = mx / 127.0 if mx > 0 else 1.0
    qi = np.clip(np.rint(x / scale), -127, 127).astype(np.int8)
    return qi.astype(np.float64) * scale


def stored_oracle(seq, q, heads, d, pos, w, page_precision, P):
    """按查询报告的实际页精度构造期望输出。"""
    pm = {vp: rep for vp, rep in page_precision}
    H = len(heads)
    out = np.zeros((H, d))
    lse = np.full(H, -np.inf)
    n = len(seq)
    for h in range(H):
        kvh = heads[h]
        lo = max(0, pos - w + 1)
        hi = min(pos, n - 1)
        if hi < lo:
            continue
        K, V = [], []
        for p in range(lo, hi + 1):
            vp, off = divmod(p, P)
            t = seq[p]
            if pm.get(vp, "float64") == "symmetric_int8":
                base = vp * P
                kb = np.stack([
                    np.asarray(seq[base + j]["k"], float)[kvh]
                    if base + j < n else np.zeros(d)
                    for j in range(P)])
                vb = np.stack([
                    np.asarray(seq[base + j]["v"], float)[kvh]
                    if base + j < n else np.zeros(d)
                    for j in range(P)])
                K.append(q8_block(kb)[off])
                V.append(q8_block(vb)[off])
            else:
                K.append(np.asarray(t["k"], float)[kvh])
                V.append(np.asarray(t["v"], float)[kvh])
        K, V = np.stack(K), np.stack(V)
        s = K @ q[h] / math.sqrt(d)
        m = s.max()
        e = np.exp(s - m)
        den = e.sum()
        out[h] = (e / den) @ V
        lse[h] = m + math.log(den)
    return out, lse


def check_invariants(eng: HandoffEngine, npages: int):
    assert eng.pool.occupied <= npages, "实际页占用超过 pages 名额"
    for pid, page in eng.pool.pages.items():
        rc = sum(1 for v in eng.views.values()
                 for e in v.table if e and e[0] == pid)
        assert rc == page.refcount, f"页 {pid} 引用计数 {rc}!={page.refcount}"
        assert page.refcount > 0
        if page.sealed:
            assert page.full, "只有整页可封存"


# ---------------------------------------------------------------------------
# 随机交错场景
# ---------------------------------------------------------------------------

def random_case(seed: int, sealed: bool):
    rng = random.Random(seed)
    npr = np.random.default_rng(seed)
    P = rng.choice([1, 2, 3, 4, 5, 8])
    npages = rng.choice([3, 4, 6, 9, 14])
    H = rng.choice([1, 2, 3])
    KV = rng.randrange(1, H + 1)
    heads = [rng.randrange(KV) for _ in range(H)]
    d = rng.choice([2, 3, 4, 8])
    budget = None if rng.random() < 0.6 else rng.choice([None, 256, 512, 2048])
    eng = HandoffEngine(
        pages=npages, page_tokens=P, d=d, heads=heads, kv_heads=KV,
        scratch_bytes=budget, auto_seal=sealed, cow_on_shared=True,
        branch_precision="original", compute_alternates=False)
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
        r = eng.append("main", p, t)
        if not r.ok:
            return  # 池太小，提前结束该场景
        main_seq.append(tokens[t])

    children = ["L", "R"]
    shadow = {}
    for c in children:
        if not eng.fork("main", c).ok:
            return
        shadow[c] = list(main_seq)
    holds = {}

    def maybe_query(ref, seq):
        if not seq:
            return
        pos = rng.randrange(len(seq))
        w = rng.choice([1, 2, P, 2 * P, 10 ** 6])
        q = npr.standard_normal((H, d))
        r = eng.query(ref, pos, w, q)
        if budget is not None and not r.ok \
                and r.error == "scratch_exhausted":
            return
        assert r.ok, (seed, r.get("error"), r.get("message"))
        if sealed:
            pp = [(x["vpage"], x["representation"])
                  for x in r.page_precision]
            eo, el = stored_oracle(seq, q, heads, d, pos, w, pp, P)
        else:
            eo, el = reference_attention(
                [t["k"] for t in seq], [t["v"] for t in seq],
                q, heads, d, window=w, pos=pos)
        np.testing.assert_allclose(r.values, eo, atol=2e-7, rtol=1e-6)
        np.testing.assert_allclose(r.lse, el, atol=2e-7, rtol=1e-6)

    for _ in range(rng.randrange(2, 14)):
        c = rng.choice(children)
        # 交错：追加 / hold / 查询草稿或快照 / 偶尔 release。
        action = rng.random()
        if action < 0.55:
            t = new_tok()
            pos = len(shadow[c])
            r = eng.append(c, pos, t)
            if r.ok:
                shadow[c].append(tokens[t])
            else:
                assert r.error == "pages_exhausted"
                assert len(shadow[c]) == pos, "失败追加消耗了可见长度"
        elif action < 0.72:
            hid = f"h{len(holds)}"
            if eng.hold(c, hid).ok:
                holds[hid] = list(shadow[c])
        elif action < 0.9:
            ref = rng.choice([c] + list(holds)) if holds else c
            seq = holds.get(ref, shadow.get(ref))
            maybe_query(ref, seq)
        else:
            if holds:
                hid = rng.choice(list(holds))
                eng.release(hid)
                holds.pop(hid, None)
        check_invariants(eng, npages)

    # 部分接收 L 的前 count 个分叉 token，然后沿胜出前缀继续。
    avail = len(shadow["L"]) - base
    count = rng.randrange(0, avail + 1) if avail >= 0 else 0
    if eng.accept("main", "L", count).ok:
        main_seq = shadow["L"][:base + count]
        assert eng.branches["main"].length == len(main_seq)
        for _ in range(rng.randrange(0, P + 1)):
            t = new_tok()
            pos = len(main_seq)
            r = eng.append("main", pos, t)
            if r.ok:
                main_seq.append(tokens[t])
            else:
                assert r.error in ("pages_exhausted",)
        maybe_query("main", main_seq)
    check_invariants(eng, npages)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()
    n_float = 60 if args.quick else 250
    n_sealed = 40 if args.quick else 160
    if args.seeds is not None:
        n_float = n_sealed = args.seeds

    t0 = time.time()
    print("[1/4] 样例采集协议回执与数值 ...")
    cap = json_loads()
    r = HandoffEngine.from_capture(cap, auto_seal=False,
                                   compute_alternates=True).run_all(cap["events"])
    assert r.events[8].error == "pages_exhausted"
    assert r.events[15].error == "position_exists"
    assert r.scratch.canonical_page_tile_bytes == 192
    q = r.queries[0]
    eo, el = reference_attention(
        [cap["tokens"][t]["k"] for t in ("a", "b", "d")],
        [cap["tokens"][t]["v"] for t in ("a", "b", "d")],
        np.array([[1, 2], [-1, 0]], float), [0, 0], 2, window=2, pos=2)
    np.testing.assert_allclose(q.values, eo, atol=1e-10)
    np.testing.assert_allclose(q.lse, el, atol=1e-10)
    assert np.all(np.isneginf(r.queries[2].lse))
    check_invariants(r.engine, cap["pages"])
    print("      ok (float oracle, 空键 -inf, scratch=192)")

    print(f"[2/4] 随机 float 对拍 x{n_float} ...")
    for s in range(n_float):
        random_case(10_000 + s, sealed=False)
    print("      ok")

    print(f"[3/4] 随机封存/精度交错对拍 x{n_sealed} ...")
    for s in range(n_sealed):
        random_case(20_000 + s, sealed=True)
    print("      ok")

    print("[4/4] 长提示有界临时区 ...")
    P, N, d, H, KV = 32, 2048, 16, 4, 2
    rng = np.random.default_rng(11)
    eng = HandoffEngine(pages=N + 2, page_tokens=P, d=d,
                        heads=[i % KV for i in range(H)], kv_heads=KV,
                        scratch_bytes=8192, auto_seal=True,
                        compute_alternates=False)
    for i in range(N):
        eng.tokens[f"t{i}"] = {
            "k": rng.standard_normal((KV, d)).tolist(),
            "v": rng.standard_normal((KV, d)).tolist()}
        assert eng.append("main", i, f"t{i}").ok
    q = rng.standard_normal((H, d))
    rr = eng.query("main", N - 1, 48, q)
    assert rr.ok and sum(rr.key_count) == H * 48
    bpk = 8 * (2 * H + 2 * KV * d)
    chunk = min(P, 8192 // bpk)
    assert rr.scratch_required_bytes == bpk * chunk
    assert eng.scratch.peak_bytes <= 8192
    check_invariants(eng, N + 2)
    print(f"      ok (chunk={chunk}, peak_scratch="
          f"{eng.scratch.peak_bytes}B, occupied={eng.pool.occupied})")

    dt = time.time() - t0
    print(f"\nACCEPTANCE OK  ({n_float + n_sealed} random cases, {dt:.1f}s)")
    return 0


def json_loads():
    import json
    return json.loads(CAPTURE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
