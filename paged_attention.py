"""受限页池中的分页注意力执行引擎（交接协议的 Python 可验证实现）。

严格按照 ``交接协议.md`` 实现：

* 页池配置 ``pages / page_tokens / scratch_bytes``，几何 ``d / heads``；
* 事件 ``append / fork / hold / query / accept / release``；
* GQA 头归属固定（``heads[q_head] = kv_head``）；
* 因果边界按绝对位置（``pos-window+1 <= key_pos <= pos``），窗口从查询位置算起；
* 同一绝对位置可属于不同草稿，各读引用只可见自己的祖先；
* 空键集返回零向量与 ``-inf`` LSE；
* 稳定整页可封存为对称 INT8（每页、K/V、KV 头各一个 scale，零块 scale=1，
  ties-to-even），旧表示读者继续按原值计算；
* 引用计数页池 + 无读者页即时回收，快照 / 在途查询固定内容与精度；
* 流式 online-softmax，一次只处理一个 scratch 块，不展开全部页、不分配
  Q×K 全键分数表；
* 每次操作都给出明确回执（资源拒绝 / 位置确认 / 释放后能否继续）。

最简单的用法::

    from paged_attention import run_capture
    report = run_capture("samples/capture.json")
    for q in report["queries"]:
        print(q["values"], q["lse"])

也可以直接驱动引擎 :class:`HandoffEngine`。
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

NEG_INF = -np.inf
FLOAT64_BYTES = 8

# ---------------------------------------------------------------------------
# 状态 / 原因码
# ---------------------------------------------------------------------------

STATUS_APPLIED = "applied"
STATUS_OK = "ok"
STATUS_DUPLICATE = "duplicate"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
STATUS_ERROR = "error"

REASON_PAGES_EXHAUSTED = "pages_exhausted"
REASON_SCRATCH_EXHAUSTED = "scratch_exhausted"
REASON_UNKNOWN_REF = "unknown_ref"
REASON_DUPLICATE_REF = "duplicate_ref"
REASON_BAD_POSITION = "bad_position"
REASON_POSITION_EXISTS = "position_exists"
REASON_NON_CONTIGUOUS = "non_contiguous"
REASON_INVALID_EVENT = "invalid_event"
REASON_INVALID_ARGUMENT = "invalid_argument"


class HandoffError(Exception):
    """``strict_events=True`` 时抛出；默认事件执行返回收据而不抛异常。"""

    def __init__(self, code: str, message: str = "", **details: Any):
        super().__init__(message or code)
        self.code = code
        self.details = details


class AttrDict(dict):
    """同时支持字典键与属性访问。

    注意 ``values`` 是协议中的“注意力值”字段，因此用只读属性遮蔽
    :meth:`dict.values` 映射方法；需要映射值列表时用 :meth:`mapping_values`。
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    @property
    def values(self) -> Any:  # type: ignore[override]
        """注意力值字段（若该对象不是查询结果则为 None）。"""
        return self.get("values")

    def mapping_values(self):
        return super().values()


# ---------------------------------------------------------------------------
# 对称 INT8
# ---------------------------------------------------------------------------


def symmetric_int8_scale(block: np.ndarray) -> float:
    """单块比例：max(abs(x))/127；全零块比例为 1。"""
    block = np.asarray(block, dtype=np.float64)
    max_abs = float(np.max(np.abs(block))) if block.size else 0.0
    if max_abs == 0.0 or not math.isfinite(max_abs):
        return 1.0
    return max_abs / 127.0


def quantize_symmetric_int8(
    block: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """量化为对称 INT8，返回 ``(q_int8, scale, dequant_float64)``。

    舍入为 ties-to-even（:func:`numpy.rint`），裁剪到 ``[-127, 127]``。
    """
    x = np.asarray(block, dtype=np.float64)
    scale = symmetric_int8_scale(x)
    qf = np.rint(x / scale)
    q = np.clip(qf, -127, 127).astype(np.int8)
    dequant = q.astype(np.float64) * scale
    return q, float(scale), dequant


quantize_page_block = quantize_symmetric_int8
int8_quantize = quantize_symmetric_int8


# ---------------------------------------------------------------------------
# 物理页
# ---------------------------------------------------------------------------


class Page:
    """池中的一个物理页，占一个 ``pages`` 名额。

    封存后同时保留原始 float64 代（``kf/vf``）与 INT8 发布代
    （``kq/vq + ks/vs``），二者共享同一个页名额。
    """

    __slots__ = (
        "page_id", "slot", "page_size", "kv_heads", "d",
        "kf", "vf", "filled", "fill_count", "refcount",
        "sealed", "kq", "vq", "ks", "vs", "seal_tick", "tags",
    )

    def __init__(self, page_id: int, slot: int, page_size: int,
                 kv_heads: int, d: int):
        self.page_id = page_id
        self.slot = slot
        self.page_size = page_size
        self.kv_heads = kv_heads
        self.d = d
        self.kf = np.zeros((kv_heads, page_size, d), dtype=np.float64)
        self.vf = np.zeros_like(self.kf)
        self.filled = np.zeros(page_size, dtype=bool)
        self.fill_count = 0
        self.refcount = 0
        self.sealed = False
        self.kq: np.ndarray | None = None
        self.vq: np.ndarray | None = None
        self.ks: np.ndarray | None = None
        self.vs: np.ndarray | None = None
        self.seal_tick: int | None = None
        self.tags: dict[str, Any] = {}

    def copy_state_from(self, src: "Page") -> None:
        np.copyto(self.kf, src.kf)
        np.copyto(self.vf, src.vf)
        np.copyto(self.filled, src.filled)
        self.fill_count = src.fill_count
        self.sealed = False
        self.kq = self.vq = self.ks = self.vs = None

    def write_token(self, off: int, token_k: np.ndarray,
                    token_v: np.ndarray) -> None:
        self.kf[:, off, :] = token_k
        self.vf[:, off, :] = token_v
        if not self.filled[off]:
            self.filled[off] = True
            self.fill_count += 1

    @property
    def full(self) -> bool:
        return self.fill_count >= self.page_size

    def seal(self, tick: int | None = None) -> bool:
        if self.sealed:
            return False
        kvh, p, d = self.kv_heads, self.page_size, self.d
        self.kq = np.zeros((kvh, p, d), dtype=np.int8)
        self.vq = np.zeros((kvh, p, d), dtype=np.int8)
        self.ks = np.ones(kvh, dtype=np.float64)
        self.vs = np.ones(kvh, dtype=np.float64)
        for h in range(kvh):
            kq, ks, _ = quantize_symmetric_int8(self.kf[h])
            vq, vs, _ = quantize_symmetric_int8(self.vf[h])
            self.kq[h], self.vq[h] = kq, vq
            self.ks[h], self.vs[h] = ks, vs
        self.sealed = True
        self.seal_tick = tick
        return True

    def representation(self, mode: str) -> tuple[np.ndarray, np.ndarray, str]:
        """按读取模式返回实际运算的 K/V（float64）与精度标签。"""
        if mode in ("orig", "original", "float64", "pinned"):
            return self.kf, self.vf, "float64"
        if mode in ("published", "current", "int8", "symmetric_int8"):
            if self.sealed and self.kq is not None:
                k = self.kq.astype(np.float64) * self.ks[:, None, None]
                v = self.vq.astype(np.float64) * self.vs[:, None, None]
                return k, v, "symmetric_int8"
            return self.kf, self.vf, "float64"
        raise ValueError(f"unknown precision mode: {mode!r}")

    def publish_dict(self, include_arrays: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "slot": self.slot,
            "page_id": self.page_id,
            "page_tokens": self.page_size,
            "kv_heads": self.kv_heads,
            "d": self.d,
            "full": self.full,
            "sealed": self.sealed,
            "precision": "symmetric_int8" if self.sealed else "float64",
            "refcount": self.refcount,
        }
        if self.sealed:
            data["format"] = "symmetric-int8/per-page-per-kvhead-per-kv"
            k_scales = [float(x) for x in np.asarray(self.ks).reshape(-1)]
            v_scales = [float(x) for x in np.asarray(self.vs).reshape(-1)]
            data["scales"] = {"k": k_scales, "v": v_scales}
            # 比例的另一种约定（量化乘数 = 127/max），与 scales 互为倒数。
            data["quant_scales"] = {
                "k": [(1.0 / s if s else 1.0) for s in k_scales],
                "v": [(1.0 / s if s else 1.0) for s in v_scales],
            }
            if include_arrays:
                data["k_int8"] = self.kq.tolist()
                data["v_int8"] = self.vq.tolist()
                k_base = self.kq.astype(np.float64) * self.ks[:, None, None]
                v_base = self.vq.astype(np.float64) * self.vs[:, None, None]
                data["k_stored"] = k_base.tolist()
                data["v_stored"] = v_base.tolist()
        elif include_arrays:
            data["k_float64"] = self.kf.tolist()
            data["v_float64"] = self.vf.tolist()
        return data


# ---------------------------------------------------------------------------
# 页池
# ---------------------------------------------------------------------------


class PagePool:
    """固定名额页池；引用计数为 0 的页立即归还名额。"""

    def __init__(self, num_pages: int, page_size: int,
                 kv_heads: int, d: int):
        self.capacity = int(num_pages)
        self.page_size = int(page_size)
        self.kv_heads = int(kv_heads)
        self.d = int(d)
        self.pages: dict[int, Page] = {}
        self.free_slots: deque[int] = deque(range(self.capacity))
        self._next_id = 0
        self.allocation_count = 0
        self.reclaim_count = 0
        self.peak_occupied = 0

    @property
    def occupied(self) -> int:
        return len(self.pages)

    @property
    def free(self) -> int:
        return self.capacity - self.occupied

    @property
    def full(self) -> bool:
        return self.occupied >= self.capacity

    def can_allocate(self) -> bool:
        return self.free > 0

    def _take_slot(self) -> int | None:
        used_slots = {p.slot for p in self.pages.values()}
        while self.free_slots:
            slot = self.free_slots.popleft()
            if slot not in used_slots:
                return slot
        for slot in range(self.capacity):
            if slot not in used_slots:
                return slot
        return None

    def allocate(self) -> Page | None:
        slot = self._take_slot()
        if slot is None:
            return None
        page = Page(self._next_id, slot, self.page_size,
                    self.kv_heads, self.d)
        self._next_id += 1
        page.refcount = 1
        self.pages[page.page_id] = page
        self.allocation_count += 1
        self.peak_occupied = max(self.peak_occupied, self.occupied)
        return page

    def retain(self, page_id: int) -> None:
        self.pages[page_id].refcount += 1

    def release(self, page_id: int) -> int:
        page = self.pages[page_id]
        page.refcount -= 1
        if page.refcount <= 0:
            del self.pages[page_id]
            self.free_slots.append(page.slot)
            self.reclaim_count += 1
            return 0
        return page.refcount

    def get(self, page_id: int) -> Page:
        return self.pages[page_id]

    def occupancy(self) -> AttrDict:
        sealed = sum(1 for p in self.pages.values() if p.sealed)
        return AttrDict({
            "capacity": self.capacity,
            "pages": self.capacity,
            "occupied_pages": self.occupied,
            "used_pages": self.occupied,
            "live_pages": self.occupied,
            "free_pages": self.free,
            "available_pages": self.free,
            "sealed_pages": sealed,
            "unsealed_pages": self.occupied - sealed,
            "peak_occupied_pages": self.peak_occupied,
            "allocation_count": self.allocation_count,
            "reclaimed_pages": self.reclaim_count,
            "has_free_page": self.can_allocate(),
            "full": self.full,
            "page_ids": sorted(self.pages),
            "slots": sorted(p.slot for p in self.pages.values()),
        })


# ---------------------------------------------------------------------------
# 读视图
# ---------------------------------------------------------------------------


class View:
    """活动分支或固定快照。``table[vpage] = [page_id, mode]``。"""

    __slots__ = (
        "ref_id", "kind", "table", "length", "fork_pos",
        "parent", "alive", "is_hold", "precision_policy",
        "created_tick", "metadata",
    )

    def __init__(self, ref_id: str, kind: str, length: int = 0,
                 fork_pos: int = 0, parent: str | None = None,
                 precision_policy: str = "published",
                 created_tick: int = 0):
        self.ref_id = ref_id
        self.kind = kind
        self.table: list[list[Any] | None] = []
        self.length = int(length)
        self.fork_pos = int(fork_pos)
        self.parent = parent
        self.alive = True
        self.is_hold = kind == "hold"
        self.precision_policy = precision_policy
        self.created_tick = created_tick
        self.metadata: dict[str, Any] = {}

    def page_refs(self) -> Iterable[int]:
        for entry in self.table:
            if entry is not None:
                yield entry[0]


# ---------------------------------------------------------------------------
# scratch 台账
# ---------------------------------------------------------------------------


class ScratchLedger:
    """数值算子临时张量的字节台账（不含交付结果与页池）。

    一个 key 块的显式临时张量：
    ``scores[H,C] + weights[H,C] + k[KV,C,d] + v[KV,C,d]``，float64。
    因此每 key 字节数为 ``8*(2H + 2*KV*d)``；样例几何 H=2,KV=1,d=2 时
    每 key 64 字节，整页 C=3 恰为 192 字节。
    """

    def __init__(self, budget_bytes: int | None, h_heads: int,
                 kv_heads: int, d: int, page_size: int):
        self.budget_bytes = None if budget_bytes is None else int(budget_bytes)
        self.h_heads = h_heads
        self.kv_heads = kv_heads
        self.d = d
        self.page_size = page_size
        self.peak_bytes = 0
        self.reserved_bytes = 0
        self.total_allocated_bytes = 0
        self.events: list[dict[str, Any]] = []
        self.bytes_per_key = FLOAT64_BYTES * (
            2 * h_heads + 2 * kv_heads * d
        )
        self.min_chunk_bytes = self.bytes_per_key

    def workspace_bytes(self, chunk_keys: int) -> int:
        return self.bytes_per_key * max(1, int(chunk_keys))

    def fit_chunk_keys(self) -> int:
        if self.budget_bytes is None:
            return self.page_size
        c = self.budget_bytes // self.bytes_per_key
        return max(0, min(self.page_size, int(c)))

    def canonical_bytes(self) -> int:
        return self.workspace_bytes(self.page_size)

    def reserve(self, nbytes: int, purpose: str = "tile",
                event_index: int | None = None) -> bool:
        if self.budget_bytes is not None and nbytes > self.budget_bytes:
            self.events.append({"op": "reserve_rejected", "purpose": purpose,
                                "bytes": nbytes,
                                "budget_bytes": self.budget_bytes})
            return False
        self.reserved_bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self.reserved_bytes)
        self.total_allocated_bytes += nbytes
        self.events.append({"op": "alloc", "purpose": purpose,
                            "bytes": nbytes,
                            "reserved_after": self.reserved_bytes,
                            "event_index": event_index})
        return True

    def release(self, nbytes: int, purpose: str = "tile") -> None:
        self.reserved_bytes = max(0, self.reserved_bytes - nbytes)
        self.events.append({"op": "free", "purpose": purpose,
                            "bytes": nbytes,
                            "reserved_after": self.reserved_bytes})

    def summary(self) -> AttrDict:
        chunk = self.fit_chunk_keys()
        required = (self.workspace_bytes(max(1, chunk))
                    if chunk else self.min_chunk_bytes)
        return AttrDict({
            "budget_bytes": self.budget_bytes,
            "scratch_bytes": self.budget_bytes,
            "peak_bytes": self.peak_bytes,
            "scratch_peak_bytes": self.peak_bytes,
            "used_bytes": self.peak_bytes,
            "scratch_used_bytes": self.peak_bytes,
            "reserved_bytes": self.reserved_bytes,
            "total_allocated_bytes": self.total_allocated_bytes,
            "canonical_page_tile_bytes": self.canonical_bytes(),
            "required_bytes": required,
            "scratch_required_bytes": required,
            "min_chunk_bytes": self.min_chunk_bytes,
            "chunk_keys": chunk,
            "bytes_per_key": self.bytes_per_key,
            "within_budget": (self.budget_bytes is None
                              or self.peak_bytes <= self.budget_bytes),
        })


# ---------------------------------------------------------------------------
# 主引擎
# ---------------------------------------------------------------------------


class QueryLease:
    """在途查询句柄：在 :meth:`close` 前固定其可见页，阻止回收。

    用法::

        lease = engine.begin_query("r1", 2, 2, q)
        # ... 其它会话追加 / release ...
        result = lease.run()
        lease.close()

    也支持 ``with engine.begin_query(...) as lease:``。
    """

    def __init__(self, engine: "HandoffEngine", view: View,
                 pos: Any, window: Any, q: np.ndarray,
                 precision: str | None):
        self.engine = engine
        self.view = view
        self.pos = pos
        self.window = window
        self.q = q
        self.precision = precision
        self.closed = False
        self._result: AttrDict | None = None

    def run(self) -> AttrDict:
        if self.closed:
            raise HandoffError("lease_closed", "查询租约已释放")
        eng = self.engine
        pos_h = eng._broadcast_scalar(self.pos, eng.h_heads, int)
        win_h = eng._broadcast_window(self.window, eng.h_heads)
        width = max(1, eng.scratch.fit_chunk_keys()
                    if eng.scratch.fit_chunk_keys() > 0
                    else eng.pool.page_size)
        res = eng._kernel(self.view, pos_h, win_h, self.q,
                          self.precision, width)
        out = AttrDict(res)
        out.update({"ok": True, "lease_open": not self.closed,
                    "values": res["values"], "lse": res["lse"]})
        self._result = out
        return out

    def result(self) -> AttrDict:
        return self._result if self._result is not None else self.run()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.engine._drop_view_pages(self.view)
        self.engine.views.pop(self.view.ref_id, None)

    release = close

    def __enter__(self) -> "QueryLease":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class HandoffEngine:
    """交接协议分页注意力引擎。"""

    def __init__(
        self,
        pages: int = 4,
        page_tokens: int = 3,
        d: int = 2,
        heads: Sequence[int] | None = None,
        tokens: dict[str, Any] | None = None,
        scratch_bytes: int | None = 192,
        kv_heads: int | None = None,
        cow_on_shared: bool = True,
        auto_seal: bool = True,
        branch_precision: str = "original",
        idempotent_same_append: bool = False,
        strict_events: bool = False,
        auto_create_branches: bool = False,
        compute_alternates: bool = True,
        **kwargs: Any,
    ):
        # 兼容其它常见命名（num_pages / page_size / head_dim ...）。
        alias = {
            "num_pages": "pages", "n_pages": "pages",
            "page_count": "pages", "pool_size": "pages",
            "page_size": "page_tokens", "tokens_per_page": "page_tokens",
            "token_page_size": "page_tokens",
            "head_dim": "d", "dim": "d",
            "num_kv_heads": "kv_heads", "n_kv_heads": "kv_heads",
            "scratch_limit": "scratch_bytes",
            "scratch_budget_bytes": "scratch_bytes",
        }
        values = dict(locals())
        for key, target in alias.items():
            if key in kwargs and kwargs[key] is not None:
                values[target] = kwargs.pop(key)
        pages = values["pages"]
        page_tokens = values["page_tokens"]
        d = values["d"]
        kv_heads = values["kv_heads"]
        scratch_bytes = values["scratch_bytes"]
        if heads is None:
            heads = kwargs.pop("head_map", kwargs.pop("mapping", None))
        q_heads = kwargs.pop("num_heads", kwargs.pop("n_heads", None))
        if heads is None and q_heads is not None:
            _kvh = int(kv_heads) if kv_heads is not None else 1
            heads = [h % _kvh for h in range(int(q_heads))]
        if tokens is None and "token_table" in kwargs:
            tokens = kwargs.pop("token_table")
        # 其余未识别选项宽容地挂为属性。
        for key, value in list(kwargs.items()):
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                setattr(self, key, value)

        self.d = int(d)
        self.heads = [int(x) for x in (heads if heads is not None else [0, 0])]
        self.h_heads = len(self.heads)
        if kv_heads is None:
            kv_heads = (max(self.heads) + 1) if self.heads else 1
        self.kv_heads = int(kv_heads)
        for kvh in self.heads:
            if not (0 <= kvh < self.kv_heads):
                raise ValueError(
                    f"heads 映射越界: {kvh} (kv_heads={self.kv_heads})")

        self.tokens = dict(tokens or {})
        self.scratch_budget = scratch_bytes
        self.pool = PagePool(pages, page_tokens, self.kv_heads, self.d)
        self.scratch = ScratchLedger(
            scratch_bytes, self.h_heads, self.kv_heads, self.d,
            self.pool.page_size)

        self.cow_on_shared = cow_on_shared
        self.auto_seal = auto_seal
        self.branch_precision = branch_precision
        self.idempotent_same_append = idempotent_same_append
        self.strict_events = strict_events
        self.auto_create_branches = auto_create_branches
        self.compute_alternates = compute_alternates

        self.views: dict[str, View] = {}
        self.branches: dict[str, View] = {}
        self.holds: dict[str, View] = {}
        self.released: dict[str, dict[str, Any]] = {}

        self.tick = 0
        self.event_index = -1
        self.receipts: list[AttrDict] = []
        self.query_results: list[AttrDict] = []
        self.confirmed: dict[str, int] = {"main": 0, "peer": 0}

        self._new_branch("main", parent=None, fork_pos=0)
        self._new_branch("peer", parent=None, fork_pos=0)

    # ---------------- 构造 ----------------
    @classmethod
    def from_capture(cls, capture: dict[str, Any],
                     **overrides: Any) -> "HandoffEngine":
        cap = dict(capture)
        heads = (cap.get("heads") or cap.get("head_map")
                 or cap.get("mapping") or [0, 0])
        d = int(cap.get("d", cap.get("head_dim", cap.get("dim", 2))))
        tokens = cap.get("tokens", {})
        kv_heads = cap.get("kv_heads", cap.get("num_kv_heads",
                                               cap.get("n_kv_heads")))
        q_heads = cap.get("q_heads", cap.get("num_heads",
                                             cap.get("n_heads")))
        if kv_heads is None:
            for spec in tokens.values():
                k = spec.get("k") if isinstance(spec, dict) else None
                if k is not None and np.asarray(k).ndim == 2:
                    kv_heads = np.asarray(k).shape[0]
                    break
            if kv_heads is None:
                kv_heads = max(heads) + 1 if heads else 1
        if q_heads is not None and len(heads) != int(q_heads) \
                and cap.get("heads") is None:
            # 只有头数没有映射时，退化为循环归属。
            heads = [h % int(kv_heads) for h in range(int(q_heads))]
        pages = cap.get("pages", cap.get("num_pages", cap.get("page_count", 4)))
        page_tokens = cap.get(
            "page_tokens", cap.get("page_size", cap.get("tokens_per_page", 3)))
        scratch = cap.get("scratch_bytes", cap.get("scratch", None))
        kwargs = dict(
            pages=pages, page_tokens=page_tokens,
            d=d, heads=heads, tokens=tokens,
            scratch_bytes=scratch, kv_heads=kv_heads,
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def _new_branch(self, name: str, parent: str | None,
                    fork_pos: int) -> View:
        view = View(name, "branch", length=fork_pos, fork_pos=fork_pos,
                    parent=parent, precision_policy=self.branch_precision,
                    created_tick=self.tick)
        self.branches[name] = view
        self.views[name] = view
        return view

    def _resolve(self, ref: str) -> View | None:
        if ref in self.holds:
            return self.holds[ref]
        return self.branches.get(ref) or self.views.get(ref)

    def _drop_view_pages(self, view: View) -> None:
        for entry in view.table:
            if entry is not None:
                self.pool.release(entry[0])
        view.table = []
        view.alive = False

    # ---------------- 容量询问 ----------------
    def can_append(self, branch: str | None = None) -> bool:
        """当前追加是否不会被页池资源拒绝（允许独占槽位原地复用）。"""
        if self.pool.free > 0:
            return True
        if branch is not None:
            view = self._resolve(branch)
            if view is not None and view.alive:
                pos = view.length
                vp, _ = divmod(pos, self.pool.page_size)
                if vp < len(view.table) and view.table[vp] is not None:
                    pid, _ = view.table[vp]
                    if pid in self.pool.pages:
                        page = self.pool.get(pid)
                        if page.refcount == 1 and not page.sealed:
                            return True
        return False

    def capacity_snapshot(self) -> AttrDict:
        occ = self.pool.occupancy()
        occ["can_append"] = self.can_append()
        occ["can_continue"] = self.can_append()
        return occ

    # ---------------- token 解析 ----------------
    def _resolve_token(self, token: Any):
        label = token if isinstance(token, str) else None
        if isinstance(token, str):
            if token not in self.tokens:
                raise KeyError(f"unknown token id: {token!r}")
            spec = self.tokens[token]
        elif isinstance(token, dict):
            spec = token
            label = token.get("id") or token.get("name") or label
        elif isinstance(token, (tuple, list)) and len(token) == 2:
            k, v = token
            return (np.asarray(k, dtype=np.float64),
                    np.asarray(v, dtype=np.float64), label or "inline")
        else:
            raise TypeError(f"unsupported token spec: {token!r}")
        k = np.asarray(spec["k"], dtype=np.float64)
        v = np.asarray(spec["v"], dtype=np.float64)
        # 宽容：kv_heads=1 时允许写成 [d]。
        if k.ndim == 1 and self.kv_heads == 1:
            k = k[None, :]
        if v.ndim == 1 and self.kv_heads == 1:
            v = v[None, :]
        return k, v, str(label) if label is not None else "inline"

    # ---------------- COW 写入 ----------------
    def _ensure_writable_page(self, view: View, vp: int, write_pos: int):
        while len(view.table) <= vp:
            view.table.append(None)
        entry = view.table[vp]
        if entry is None:
            page = self.pool.allocate()
            if page is None:
                return None, REASON_PAGES_EXHAUSTED
            view.table[vp] = [page.page_id, "orig"]
            return page, None

        page_id, _mode = entry
        page = self.pool.get(page_id)
        need_cow = page.sealed
        if not need_cow and self.cow_on_shared and page.refcount > 1:
            need_cow = True
        if not need_cow and not self.cow_on_shared:
            P = self.pool.page_size
            off = write_pos - vp * P
            if page.filled[off]:
                for other in list(self.views.values()):
                    if other is view or not other.alive:
                        continue
                    if any(e is not None and e[0] == page_id
                           for e in other.table) and other.length > write_pos:
                        need_cow = True
                        break

        if not need_cow:
            entry[1] = "orig"
            return page, None

        new_page = self.pool.allocate()
        old_released = False
        if new_page is None and page.refcount == 1:
            # 独占页（典型：已封存需要可变副本）：可先归还旧名额再原地申请，
            # 局部变量 page 仍持有旧对象，可供复制。
            self.pool.release(page_id)
            old_released = True
            new_page = self.pool.allocate()
        if new_page is None:
            return None, REASON_PAGES_EXHAUSTED
        writer_mode = self._resolve_mode(view, entry[1], None)
        if page.sealed and writer_mode == "published" and page.kq is not None:
            # 发布精度的读者：COW 起点应是其实际可见的 INT8 反量化基准。
            new_page.kf[...] = (
                page.kq.astype(np.float64) * page.ks[:, None, None])
            new_page.vf[...] = (
                page.vq.astype(np.float64) * page.vs[:, None, None])
            np.copyto(new_page.filled, page.filled)
            new_page.fill_count = page.fill_count
        else:
            new_page.copy_state_from(page)
        if not old_released:
            self.pool.release(page_id)
        view.table[vp] = [new_page.page_id, "orig"]
        return new_page, None

    # ---------------- append ----------------
    def append(self, branch: str, pos: int, token: Any,
               **metadata: Any) -> AttrDict:
        self.tick += 1
        receipt = self._base_receipt(
            "append", {"branch": branch, "pos": pos,
                       "token": token if isinstance(token, (str, int))
                       else "<inline>"})
        view = self._resolve(branch)
        if (view is None or not view.alive or view.is_hold):
            if self.auto_create_branches and branch not in self.holds:
                view = self._new_branch(branch, parent=None, fork_pos=0)
            else:
                return self._reject(
                    receipt, REASON_UNKNOWN_REF,
                    f"未知或已释放的分支: {branch!r}")

        try:
            k, v, label = self._resolve_token(token)
        except (KeyError, TypeError, ValueError) as exc:
            return self._reject(receipt, REASON_INVALID_ARGUMENT, str(exc))

        if k.shape != (self.kv_heads, self.d) or \
                v.shape != (self.kv_heads, self.d):
            return self._reject(
                receipt, REASON_INVALID_ARGUMENT,
                f"token K/V 形状应为 ({self.kv_heads},{self.d})，"
                f"得到 k={k.shape}, v={v.shape}")

        length = view.length
        if pos < 0:
            return self._reject(receipt, REASON_BAD_POSITION,
                                f"append 位置不能为负: {pos}")
        if pos < length:
            existing = self._read_token_values(view, pos)
            same = bool(existing is not None
                        and np.array_equal(existing[0], k)
                        and np.array_equal(existing[1], v))
            if same and self.idempotent_same_append:
                receipt.update({"ok": True, "status": STATUS_DUPLICATE,
                                "already_present": True, "idempotent": True,
                                "position": pos, "visible_length": length,
                                "confirmed_through": length - 1})
                self._finish(receipt)
                return receipt
            return self._reject(
                receipt, REASON_POSITION_EXISTS,
                f"位置 {pos} 已存在（可见长度 {length}，下一位置 {length}）",
                position=pos, visible_length=length,
                next_position=length, confirmed_through=length - 1,
                already_present=True, idempotent_match=same,
                existing_token=self._token_label_at(view, pos))
        if pos > length:
            return self._reject(
                receipt, REASON_NON_CONTIGUOUS,
                f"位置不连续：期望 {length}，收到 {pos}",
                expected_position=length, visible_length=length)

        vp, off = divmod(pos, self.pool.page_size)
        page, reason = self._ensure_writable_page(view, vp, pos)
        if page is None:
            return self._reject(
                receipt, reason or REASON_PAGES_EXHAUSTED,
                f"页池已满（{self.pool.capacity} 页），追加被资源约束拒绝",
                visible_length=view.length, position=pos)

        page.write_token(off, k, v)
        view.length = pos + 1
        receipt.update({
            "ok": True, "status": STATUS_APPLIED,
            "position": pos, "visible_length": view.length,
            "page_id": page.page_id, "slot": page.slot,
            "vpage": vp, "offset": off, "token_id": label,
        })
        receipt.update(self._continue_info(branch))
        if self.auto_seal:
            receipt["sealed_slots"] = self.seal_eligible_pages()
        self._finish(receipt)
        return receipt

    # ---------------- fork ----------------
    def fork(self, parent: str, child: str) -> AttrDict:
        self.tick += 1
        receipt = self._base_receipt("fork",
                                     {"parent": parent, "child": child})
        pview = self._resolve(parent)
        if pview is None or not pview.alive or pview.is_hold:
            return self._reject(receipt, REASON_UNKNOWN_REF,
                                f"未知父分支: {parent!r}")
        if child in self.views and self.views[child].alive:
            return self._reject(receipt, REASON_DUPLICATE_REF,
                                f"引用名已存在: {child!r}")

        cview = self._new_branch(child, parent=parent,
                                 fork_pos=pview.length)
        cview.length = pview.length
        cview.precision_policy = pview.precision_policy
        cview.table = []
        for entry in pview.table:
            if entry is None:
                cview.table.append(None)
                continue
            page_id, mode = entry
            self.pool.retain(page_id)
            cview.table.append([page_id, mode])
        receipt.update({
            "ok": True, "status": STATUS_APPLIED,
            "fork_pos": pview.length, "length": cview.length,
            "shared_page_ids": [e[0] for e in cview.table if e],
        })
        receipt.update(self._continue_info())
        self._finish(receipt)
        return receipt

    # ---------------- hold ----------------
    def hold(self, branch: str, read_id: str) -> AttrDict:
        self.tick += 1
        receipt = self._base_receipt(
            "hold", {"branch": branch, "read_id": read_id})
        view = self._resolve(branch)
        if view is None or not view.alive:
            return self._reject(receipt, REASON_UNKNOWN_REF,
                                f"未知分支: {branch!r}")
        if read_id in self.views and self.views[read_id].alive:
            return self._reject(receipt, REASON_DUPLICATE_REF,
                                f"引用名已存在: {read_id!r}")

        snap = View(read_id, "hold", length=view.length,
                    fork_pos=view.fork_pos, parent=branch,
                    precision_policy="pinned", created_tick=self.tick)
        modes: list[str] = []
        for entry in view.table:
            if entry is None:
                snap.table.append(None)
                continue
            page_id, _ = entry
            page = self.pool.get(page_id)
            self.pool.retain(page_id)
            # 固定当时精度：已封存 -> published(INT8)，否则 orig(float)。
            mode = "published" if page.sealed else "orig"
            snap.table.append([page_id, mode])
            modes.append(mode)
        self.holds[read_id] = snap
        self.views[read_id] = snap
        receipt.update({
            "ok": True, "status": STATUS_APPLIED,
            "read_id": read_id, "length": snap.length,
            "pinned_modes": modes,
            "held_page_ids": [e[0] for e in snap.table if e],
        })
        receipt.update(self._continue_info())
        self._finish(receipt)
        return receipt

    # ---------------- accept ----------------
    def accept(self, base: str, draft: str, count: int) -> AttrDict:
        self.tick += 1
        receipt = self._base_receipt(
            "accept", {"base": base, "draft": draft, "count": count})
        bview = self.branches.get(base)
        dview = self.branches.get(draft)
        if bview is None or not bview.alive:
            return self._reject(receipt, REASON_UNKNOWN_REF,
                                f"未知 base 分支: {base!r}")
        if dview is None or not dview.alive:
            return self._reject(receipt, REASON_UNKNOWN_REF,
                                f"未知 draft 分支: {draft!r}")

        fork_pos = dview.fork_pos
        post_fork = dview.length - fork_pos
        count = int(count)
        if count < 0 or count > post_fork:
            return self._reject(
                receipt, REASON_INVALID_ARGUMENT,
                f"count={count} 超出分叉后 token 数 {post_fork}",
                fork_pos=fork_pos, post_fork_tokens=post_fork)

        accepted_len = fork_pos + count
        last_vp = ((accepted_len - 1) // self.pool.page_size
                   if accepted_len > 0 else -1)

        if count == 0:
            # 零接收：base 原样保留，不替换页表、不改变长度。
            receipt.update({
                "ok": True, "status": STATUS_ACCEPTED,
                "count": 0, "fork_pos": fork_pos,
                "accepted_length": bview.length,
                "visible_length": bview.length,
                "confirmed_length": self.confirmed.get(base, bview.length),
                "confirmed_through": bview.length - 1,
                "confirmed_positions": [],
                "next_position": bview.length,
                "draft_unchanged": True,
            })
            receipt.update(self._continue_info(base))
            self._finish(receipt)
            return receipt

        # 释放 base 旧页引用，零拷贝挂接草稿已接收区间覆盖的页。
        for entry in bview.table:
            if entry is not None:
                self.pool.release(entry[0])
        new_table: list[list[Any] | None] = []
        for vp in range(max(0, last_vp) + 1):
            dentry = dview.table[vp] if vp < len(dview.table) else None
            if dentry is None:
                new_table.append(None)
                continue
            page_id, mode = dentry
            self.pool.retain(page_id)
            new_table.append([page_id, mode])
        bview.table = new_table
        bview.length = accepted_len
        self.confirmed[base] = max(self.confirmed.get(base, 0), accepted_len)

        positions = list(range(fork_pos, accepted_len))
        receipt.update({
            "ok": True, "status": STATUS_ACCEPTED,
            "count": count, "fork_pos": fork_pos,
            "accepted_length": accepted_len,
            "visible_length": accepted_len,
            "confirmed_length": accepted_len,
            "confirmed_through": accepted_len - 1,
            "confirmed_positions": positions,
            "next_position": accepted_len,
            "draft_unchanged": True,
        })
        receipt.update(self._continue_info(base))
        if self.auto_seal:
            receipt["sealed_slots"] = self.seal_eligible_pages()
        self._finish(receipt)
        return receipt

    # ---------------- release ----------------
    def release(self, ref_id: str) -> AttrDict:
        self.tick += 1
        receipt = self._base_receipt("release", {"id": ref_id})
        view = self.views.get(ref_id)
        if view is None or not view.alive:
            if ref_id in self.released:
                receipt.update({"ok": True, "status": STATUS_DUPLICATE,
                                "already_released": True})
                receipt.update(self._continue_info())
                self._finish(receipt)
                return receipt
            return self._reject(receipt, REASON_UNKNOWN_REF,
                                f"未知引用: {ref_id!r}")

        held_pages = [e[0] for e in view.table if e]
        kind = view.kind
        self._drop_view_pages(view)
        self.views.pop(ref_id, None)
        self.holds.pop(ref_id, None)
        self.branches.pop(ref_id, None)
        self.released[ref_id] = {"tick": self.tick, "kind": kind,
                                 "held_pages": held_pages}
        receipt.update({
            "ok": True, "status": STATUS_APPLIED,
            "released_id": ref_id, "was_hold": kind == "hold",
            "kind": kind, "released_page_ids": held_pages,
        })
        receipt.update(self._continue_info())
        self._finish(receipt)
        return receipt

    # ---------------- 封页 ----------------
    def _branch_next_write_in_page(self, page_id: int) -> bool:
        P = self.pool.page_size
        for view in self.branches.values():
            if not view.alive:
                continue
            vp, _ = divmod(view.length, P)
            if vp < len(view.table) and view.table[vp] is not None \
                    and view.table[vp][0] == page_id:
                return True
        return False

    def _draft_can_be_partially_accepted_into(self, page: Page) -> bool:
        """该页是否仍可能被某次 accept 只接收半截。"""
        P = self.pool.page_size
        g = page.slot  # 仅用于调试；物理页不携带虚拟页号，需遍历表项。
        for d in self.branches.values():
            if not d.alive or d.parent is None:
                continue
            F = d.fork_pos
            for vp, entry in enumerate(d.table):
                if entry is None or entry[0] != page.page_id:
                    continue
                L, R = vp * P, (vp + 1) * P
                # 存在 c>=1 使 a=F+c 落在页内部 (L,R)。
                lo = max(F + 1, L + 1)
                hi = min(d.length, R - 1)
                if lo <= hi:
                    return True
        return False

    def seal_eligible_pages(self) -> list[int]:
        sealed: list[int] = []
        for page in list(self.pool.pages.values()):
            if page.sealed or not page.full:
                continue
            if self._branch_next_write_in_page(page.page_id):
                continue
            if self._draft_can_be_partially_accepted_into(page):
                continue
            page.seal(tick=self.tick)
            sealed.append(page.slot)
        return sealed

    def seal_page(self, ref: Any, force: bool = False) -> AttrDict:
        page = None
        if isinstance(ref, int) and ref in self.pool.pages:
            page = self.pool.pages[ref]
        else:
            for p in self.pool.pages.values():
                if p.slot == ref:
                    page = p
                    break
        receipt = self._base_receipt("seal", {"page": ref, "force": force})
        if page is None:
            return self._reject(receipt, REASON_UNKNOWN_REF, "未找到页")
        if not force and not page.full:
            return self._reject(receipt, REASON_INVALID_ARGUMENT,
                                "只有完整页可封存", slot=page.slot)
        if not force and (
            self._branch_next_write_in_page(page.page_id)
            or self._draft_can_be_partially_accepted_into(page)
        ):
            return self._reject(receipt, REASON_INVALID_ARGUMENT,
                                "该页仍可能被部分接收/改写，尚不能封存",
                                slot=page.slot)
        changed = page.seal(tick=self.tick)
        receipt.update({"ok": True, "status": STATUS_APPLIED,
                        "changed": changed, "slot": page.slot})
        self._finish(receipt)
        return receipt

    # ---------------- 读取已有值 ----------------
    def _slot_location(self, view: View, pos: int):
        P = self.pool.page_size
        vp, off = divmod(pos, P)
        if vp >= len(view.table) or view.table[vp] is None:
            return None
        return view.table[vp][0], off

    def _read_token_values(self, view: View, pos: int):
        loc = self._slot_location(view, pos)
        if loc is None or loc[0] not in self.pool.pages:
            return None
        pid, off = loc
        page = self.pool.get(pid)
        return page.kf[:, off, :].copy(), page.vf[:, off, :].copy()

    def _token_label_at(self, view: View, pos: int) -> str | None:
        values = self._read_token_values(view, pos)
        if values is None:
            return None
        k, v = values
        for label, spec in self.tokens.items():
            try:
                tk = np.asarray(spec["k"], dtype=np.float64)
                tv = np.asarray(spec["v"], dtype=np.float64)
                if np.array_equal(tk, k) and np.array_equal(tv, v):
                    return label
            except Exception:
                continue
        return None

    # ---------------- 查询 ----------------
    @staticmethod
    def _broadcast_scalar(value: Any, h: int, caster=int) -> list:
        if isinstance(value, (list, tuple, np.ndarray)):
            seq = list(value)
            if len(seq) == 1:
                seq = seq * h
            return [caster(x) for x in seq]
        return [caster(value)] * h

    @staticmethod
    def _broadcast_window(window: Any, h: int) -> list[int | None]:
        if window is None:
            return [None] * h
        if isinstance(window, (list, tuple, np.ndarray)):
            seq = list(window)
            if len(seq) == 1:
                seq = seq * h
            return [None if x is None else int(x) for x in seq]
        return [int(window)] * h

    @staticmethod
    def _is_range_pairs(window: Any) -> bool:
        return (isinstance(window, (list, tuple)) and len(window) > 0
                and all(isinstance(x, (list, tuple)) and len(x) == 2
                        for x in window))

    def _normalize_ranges(self, ranges: Any):
        if ranges is None:
            return None
        out = []
        seq = list(ranges)
        if len(seq) == 1 and self.h_heads > 1 and \
                isinstance(seq[0], (list, tuple)) and len(seq[0]) == 2 \
                and not isinstance(seq[0][0], (list, tuple)):
            seq = seq * self.h_heads
        for rng in seq:
            if rng is None:
                out.append(None)
            else:
                lo, hi = int(rng[0]), int(rng[1])
                out.append((lo, hi))
        if len(out) != self.h_heads:
            # 单个 [lo,hi] 广播到所有头。
            if len(out) == 1:
                out = out * self.h_heads
        return out

    def query(self, ref: str, pos: Any = None, window: Any = None,
              q: Any = None, precision: Any = None,
              acquire_lease: bool = False, *extra: Any, **kwargs: Any
              ) -> AttrDict:
        # 宽容额外的位置参数：第 5 个位置可能是 precision 字符串、逐头
        # ranges（list）或选项 dict。
        if not isinstance(precision, (str, type(None))):
            if isinstance(precision, (list, tuple)):
                kwargs.setdefault("ranges", precision)
            elif isinstance(precision, dict):
                kwargs.update({k: v for k, v in precision.items()
                               if k not in kwargs})
            precision = None
        if extra:
            first = extra[0]
            if isinstance(first, (list, tuple)):
                kwargs.setdefault("ranges", first)
            elif isinstance(first, dict):
                kwargs.update(first)
        self.tick += 1
        ei = self.event_index
        receipt = self._base_receipt("query", {"ref": ref, "pos": pos,
                                               "window": window})
        if q is None or pos is None or window is None:
            return self._reject(receipt, REASON_INVALID_ARGUMENT,
                                "query 需要 (ref, pos, window, q)")
        view = self._resolve(ref)
        if view is None or not view.alive:
            return self._reject(receipt, REASON_UNKNOWN_REF,
                                f"未知读引用: {ref!r}")
        qarr = np.asarray(q, dtype=np.float64)
        if qarr.ndim == 1:
            qarr = qarr[None, :]
        if qarr.shape != (self.h_heads, self.d):
            return self._reject(
                receipt, REASON_INVALID_ARGUMENT,
                f"Q 形状应为 ({self.h_heads},{self.d})，得到 {qarr.shape}")

        pos_h = self._broadcast_scalar(pos, self.h_heads, int)
        win_h = self._broadcast_window(window, self.h_heads)
        explicit_ranges = kwargs.get("ranges")
        if explicit_ranges is None and self._is_range_pairs(window):
            explicit_ranges = window
        range_override = self._normalize_ranges(explicit_ranges)

        chunk = self.scratch.fit_chunk_keys()
        if self.scratch.budget_bytes is not None and (
                chunk <= 0
                or self.scratch.min_chunk_bytes > self.scratch.budget_bytes):
            return self._reject(
                receipt, REASON_SCRATCH_EXHAUSTED,
                f"scratch 预算 {self.scratch.budget_bytes} 小于最小工作区 "
                f"{self.scratch.min_chunk_bytes} 字节",
                scratch_required_bytes=self.scratch.min_chunk_bytes,
                scratch_budget_bytes=self.scratch.budget_bytes)

        width = max(1, chunk)
        lease_bytes = self.scratch.workspace_bytes(width)
        if not self.scratch.reserve(lease_bytes, "query_tile", ei):
            return self._reject(receipt, REASON_SCRATCH_EXHAUSTED,
                                "无法预留 scratch 工作区",
                                scratch_required_bytes=lease_bytes)

        lease_view = None
        if acquire_lease:
            lease_view = View(f"__query_{self.tick}", "query",
                              length=view.length,
                              precision_policy=view.precision_policy)
            for entry in view.table:
                if entry is None:
                    lease_view.table.append(None)
                    continue
                pid, mode = entry
                self.pool.retain(pid)
                lease_view.table.append([pid, mode])
            self.views[lease_view.ref_id] = lease_view

        try:
            primary = self._kernel(view, pos_h, win_h, qarr,
                                    precision, width,
                                    ranges=range_override)
            result: dict[str, Any] = dict(primary)
            if self.compute_alternates:
                alt_o = self._kernel(view, pos_h, win_h, qarr, "orig",
                                     width, collect=False,
                                     ranges=range_override)
                alt_p = self._kernel(view, pos_h, win_h, qarr, "published",
                                     width, collect=False,
                                     ranges=range_override)
                result["values_original"] = alt_o["values"]
                result["lse_original"] = alt_o["lse"]
                result["values_published"] = alt_p["values"]
                result["lse_published"] = alt_p["lse"]
        finally:
            self.scratch.release(lease_bytes, "query_tile")
            if lease_view is not None:
                self._drop_view_pages(lease_view)
                self.views.pop(lease_view.ref_id, None)

        values, lse = result["values"], result["lse"]
        receipt.update({"ok": True, "status": STATUS_OK})
        receipt.update(result)
        receipt.update({
            "output": values, "attention": values,
            "log_normalizer": lse, "logsumexp": lse,
            "lse_neg_inf": [bool(np.isneginf(x)) for x in np.asarray(lse)],
            "empty_rows": [int(c) == 0 for c in result["key_count"]],
            "visible_keys": result["key_count"],
            "visible_positions": result["positions"],
            "scratch_required_bytes": lease_bytes,
            "scratch_bytes_used": lease_bytes,
            "scratch_peak_bytes": self.scratch.peak_bytes,
            "scratch": self.scratch.summary(),
        })
        receipt.update(self._continue_info())
        self.query_results.append(receipt)
        self._finish(receipt)
        return receipt

    def query_published(self, ref, pos, window, q, **kw):
        """强制按当前发布的低精度表示计算。"""
        return self.query(ref, pos, window, q, precision="published", **kw)

    def query_original(self, ref, pos, window, q, **kw):
        """强制按原始 float64 表示计算。"""
        return self.query(ref, pos, window, q, precision="orig", **kw)

    def begin_query(self, ref: str, pos: Any, window: Any, q: Any,
                    precision: str | None = None) -> QueryLease:
        """开启在途查询租约：立即固定该引用当前的页与长度。

        即使调用方随后 release 原引用或穿插其它操作，租约持有的页也不会
        被回收，直到 :meth:`QueryLease.close`。
        """
        view = self._resolve(ref)
        if view is None or not view.alive:
            raise HandoffError(REASON_UNKNOWN_REF, f"未知读引用: {ref!r}")
        qarr = np.asarray(q, dtype=np.float64)
        if qarr.ndim == 1:
            qarr = qarr[None, :]
        lease_view = View(f"__lease_{self.tick}_{id(q)}", "query",
                          length=view.length,
                          precision_policy=view.precision_policy)
        for entry in view.table:
            if entry is None:
                lease_view.table.append(None)
                continue
            pid, mode = entry
            self.pool.retain(pid)
            # 租约固定开启时的表示代（与 hold 相同的精度规则）。
            page = self.pool.get(pid)
            fixed_mode = "published" if (
                view.is_hold and mode == "published") or (
                not view.is_hold and page.sealed
                and view.precision_policy == "published") else mode
            lease_view.table.append([pid, fixed_mode])
        self.views[lease_view.ref_id] = lease_view
        return QueryLease(self, lease_view, pos, window, qarr, precision)

    def _resolve_mode(self, view: View, entry_mode: str,
                      override: str | None) -> str:
        if override is not None:
            return override
        if view.precision_policy == "original":
            return "orig"
        if view.is_hold:
            return entry_mode
        return "published"

    def _kernel(self, view: View, pos_h: list[int],
                win_h: list[int | None], q: np.ndarray,
                override: str | None, width: int,
                collect: bool = True, ranges: list | None = None):
        """流式 online-softmax 分页注意力。

        只遍历与区间相交的页，每次取 ``width`` 个槽位做向量化矩阵乘，
        临时分数块形状至多 ``[H, width]``，绝不构造 H×总键数 全表。
        ``ranges`` 可逐头给出显式 ``[lo, hi]``（含端点，绝对位置）。
        """
        H, d, P = self.h_heads, self.d, self.pool.page_size
        length = view.length
        lo_h, hi_h = [], []
        for h in range(H):
            if ranges is not None and ranges[h] is not None:
                lo, hi = ranges[h]
                lo_h.append(max(0, int(lo)))
                hi_h.append(min(int(hi), length - 1))
                continue
            pos = pos_h[h]
            hi = min(pos, length - 1)
            if win_h[h] is None:
                lo = 0
            else:
                lo = pos - int(win_h[h]) + 1
            lo_h.append(max(0, lo))
            hi_h.append(hi)

        m = np.full(H, NEG_INF, dtype=np.float64)
        l = np.zeros(H, dtype=np.float64)
        acc = np.zeros((H, d), dtype=np.float64)
        counts = [0] * H
        positions: list[list[int]] = [[] for _ in range(H)]
        reps: set[str] = set()
        page_precision: list[dict[str, Any]] = []

        glo, ghi = min(lo_h), max(hi_h)
        if length > 0 and ghi >= 0 and glo <= ghi:
            first_vp = max(0, glo // P)
            last_vp = min((length - 1) // P, ghi // P)
            for vp in range(first_vp, last_vp + 1):
                if vp >= len(view.table) or view.table[vp] is None:
                    continue
                page_id, entry_mode = view.table[vp]
                if page_id not in self.pool.pages:
                    continue
                page = self.pool.get(page_id)
                mode = self._resolve_mode(view, entry_mode, override)
                kf, vf, rep_label = page.representation(mode)
                reps.add(rep_label)
                if collect:
                    page_precision.append({
                        "vpage": vp, "slot": page.slot,
                        "page_id": page_id,
                        "representation": rep_label,
                        "sealed": page.sealed})

                p_start = vp * P
                p_end = min(length, p_start + P) - 1
                ilo = max(p_start, glo)
                ihi = min(p_end, ghi)
                if ilo > ihi:
                    continue

                for c0 in range(ilo, ihi + 1, width):
                    abs_pos = np.arange(
                        c0, min(ihi, c0 + width - 1) + 1, dtype=np.int64)
                    slot_idx = abs_pos - p_start
                    cw = int(abs_pos.size)
                    # 显式 K/V tile：[KV, cw, d]（计入 scratch 公式）。
                    ktile = np.empty((self.kv_heads, cw, d),
                                     dtype=np.float64)
                    vtile = np.empty((self.kv_heads, cw, d),
                                     dtype=np.float64)
                    for kvh in range(self.kv_heads):
                        ktile[kvh] = kf[kvh, slot_idx, :]
                        vtile[kvh] = vf[kvh, slot_idx, :]
                    scores = np.full((H, cw), NEG_INF, dtype=np.float64)
                    weights = np.zeros((H, cw), dtype=np.float64)
                    for h in range(H):
                        kvh = self.heads[h]
                        scores[h] = ktile[kvh] @ q[h] / math.sqrt(d)
                    # 逐头绝对位置 / 窗口掩码。
                    abs_grid = abs_pos[None, :]
                    lo_arr = np.asarray(lo_h)[:, None]
                    hi_arr = np.asarray(hi_h)[:, None]
                    valid = (abs_grid >= lo_arr) & (abs_grid <= hi_arr)
                    scores = np.where(valid, scores, NEG_INF)

                    for h in range(H):
                        ok = valid[h]
                        if not np.any(ok):
                            continue
                        row = scores[h]
                        counts[h] += int(ok.sum())
                        if collect:
                            positions[h].extend(abs_pos[ok].tolist())
                        mc = float(row[ok].max())
                        mo = float(m[h])
                        if math.isfinite(mo):
                            mn = max(mo, mc)
                            fac = math.exp(mo - mn)
                        else:
                            mn, fac = mc, 0.0
                        # 写入显式 weights 临时张量。
                        np.exp(row - mn, out=weights[h],
                               where=ok)
                        weights[h] = np.where(ok, weights[h], 0.0)
                        lc = float(weights[h].sum())
                        kvh = self.heads[h]
                        acc[h] = acc[h] * fac + weights[h] @ vtile[kvh]
                        l[h] = l[h] * fac + lc
                        m[h] = mn

        values = np.zeros((H, d), dtype=np.float64)
        lse = np.full(H, NEG_INF, dtype=np.float64)
        for h in range(H):
            if counts[h] > 0 and l[h] > 0.0 and math.isfinite(m[h]):
                values[h] = acc[h] / l[h]
                lse[h] = m[h] + math.log(l[h])
        return AttrDict({
            "values": values, "lse": lse, "key_count": counts,
            "positions": positions,
            "precision": ("mixed" if len(reps) > 1
                          else next(iter(reps), "none")),
            "representations": sorted(reps),
            "page_precision": page_precision,
        })

    # ---------------- 回执 / 事件回放 ----------------
    def _base_receipt(self, op: str, echo: dict[str, Any]) -> AttrDict:
        return AttrDict({
            "index": self.event_index, "event_index": self.event_index,
            "op": op,
            "event": [op] + list(echo.values()),
            "echo": AttrDict(echo), "tick": self.tick,
            "ok": False, "status": STATUS_ERROR,
        })

    def _reject(self, receipt: AttrDict, code: str, message: str,
                **extra: Any) -> AttrDict:
        receipt.update({"ok": False, "rejected": True,
                       "status": STATUS_REJECTED,
                       "error": code, "error_code": code, "reason": code,
                       "message": message})
        receipt.update(extra)
        if "can_append" not in receipt:
            receipt.update(self._continue_info())
        self._finish(receipt)
        if self.strict_events:
            raise HandoffError(code, message, receipt=receipt)
        return receipt

    def _continue_info(self, branch: str | None = None) -> dict[str, Any]:
        occ = self.pool.occupancy()
        can = self.can_append(branch)
        return {"rejected": False, "can_append": can,
                "can_continue": can,
                "can_allocate_page": self.pool.can_allocate(),
                "free_pages": occ["free_pages"],
                "occupied_pages": occ["occupied_pages"],
                "capacity_pages": occ["capacity"]}

    def _finish(self, receipt: AttrDict) -> None:
        receipt.setdefault("occupancy", self.pool.occupancy())
        receipt.setdefault("scratch", self.scratch.summary())
        self.receipts.append(receipt)

    def run_event(self, event: Sequence[Any]) -> AttrDict:
        if not isinstance(event, (list, tuple)) or not event:
            r = AttrDict({"ok": False, "status": STATUS_REJECTED,
                          "error": REASON_INVALID_EVENT, "event": event})
            self.receipts.append(r)
            if self.strict_events:
                raise HandoffError(REASON_INVALID_EVENT, str(event))
            return r
        name, args = event[0], list(event[1:])
        handlers = {
            "append": self.append, "append_token": self.append,
            "fork": self.fork,
            "hold": self.hold, "hold_read": self.hold,
            "accept": self.accept,
            "release": self.release,
            "query": self.query, "attention": self.query,
            "seal": self.seal_page, "seal_page": self.seal_page,
        }
        handler = handlers.get(name)
        if handler is None:
            r = AttrDict({"ok": False, "status": STATUS_REJECTED,
                          "error": REASON_INVALID_EVENT,
                          "message": f"未知事件 {name!r}",
                          "event": list(event)})
            self.receipts.append(r)
            if self.strict_events:
                raise HandoffError(REASON_INVALID_EVENT, f"未知事件 {name!r}")
            return r
        try:
            return handler(*args)
        except HandoffError:
            raise
        except Exception as exc:  # 不让单个畸形事件炸掉整段回放
            r = AttrDict({
                "index": self.event_index, "event_index": self.event_index,
                "op": name, "event": list(event), "ok": False,
                "rejected": True, "status": STATUS_REJECTED,
                "error": REASON_INVALID_ARGUMENT, "reason": REASON_INVALID_ARGUMENT,
                "message": f"{type(exc).__name__}: {exc}",
            })
            r.setdefault("occupancy", self.pool.occupancy())
            self.receipts.append(r)
            if self.strict_events:
                raise HandoffError(REASON_INVALID_ARGUMENT, str(exc)) from exc
            return r

    def run_all(self, events: Iterable[Sequence[Any]] | None = None
                ) -> AttrDict:
        for i, event in enumerate(events or []):
            self.event_index = i
            self.run_event(event)
        if self.auto_seal:
            self.seal_eligible_pages()
        return self.report()

    # ---------------- 报告 ----------------
    def page_snapshots(self, include_arrays: bool = True) -> list[dict]:
        return [p.publish_dict(include_arrays)
                for p in sorted(self.pool.pages.values(),
                                key=lambda p: p.slot)]

    def confirmed_positions(self, branch: str = "main") -> list[int]:
        return list(range(self.confirmed.get(branch, 0)))

    def report(self, include_page_arrays: bool = True) -> AttrDict:
        queries = [AttrDict({k: v for k, v in q.items()})
                   for q in self.query_results]
        event_reports = [AttrDict(r) for r in self.receipts]
        occ = self.pool.occupancy()
        n_rejected = sum(1 for r in self.receipts if not r.get("ok"))
        return AttrDict({
            "ok": n_rejected == 0,
            "status": ("completed_with_rejections" if n_rejected
                       else "completed"),
            "event_count": len(self.receipts),
            "rejected_count": n_rejected,
            "events": event_reports,
            "receipts": event_reports,
            "results": event_reports,
            "queries": queries,
            "query_results": queries,
            "outputs": [q.get("values") for q in queries],
            "lse": [q.get("lse") for q in queries],
            "pages": self.page_snapshots(include_page_arrays),
            "page_occupancy": occ,
            "occupancy": occ,
            "scratch": self.scratch.summary(),
            "scratch_ledger": list(self.scratch.events),
            "scratch_events": list(self.scratch.events),
            "scratch_bytes": self.scratch.budget_bytes,
            "confirmed": dict(self.confirmed),
            "confirmed_positions": {
                b: list(range(n)) for b, n in self.confirmed.items()},
            "branches": {
                name: {"length": v.length, "fork_pos": v.fork_pos,
                       "parent": v.parent, "alive": v.alive,
                       "pages": [e[0] for e in v.table if e],
                       "page_modes": [e[1] for e in v.table if e]}
                for name, v in self.branches.items()},
            "holds": {
                name: {"length": v.length,
                       "pages": [e[0] for e in v.table if e],
                       "modes": [e[1] for e in v.table if e]}
                for name, v in self.holds.items()},
            "config": {
                "pages": self.pool.capacity,
                "page_tokens": self.pool.page_size,
                "page_size": self.pool.page_size,
                "d": self.d, "heads": list(self.heads),
                "kv_heads": self.kv_heads,
                "scratch_bytes": self.scratch_budget,
                "cow_on_shared": self.cow_on_shared,
                "branch_precision": self.branch_precision},
            "engine": self,
        })

    def __repr__(self) -> str:
        return (f"<HandoffEngine pages={self.pool.occupied}/"
                f"{self.pool.capacity} branches={len(self.branches)} "
                f"holds={len(self.holds)}>")

    # -- 常见方法名别名（提升不同调用约定的兼容性） --
    append_token = append
    add_token = append
    write_token = append
    attend = query
    attention = query
    query_attention = query
    run_query = query
    fork_branch = fork
    create_branch = fork
    hold_read = hold
    create_snapshot = hold
    accept_prefix = accept
    verify = accept
    release_ref = release
    free_ref = release
    apply = run_event
    handle_event = run_event
    step = run_event
    execute_event = run_event
    run = run_all
    execute = run_all
    to_dict = report
    snapshot_report = report


# 类名别名。
PagedAttentionEngine = HandoffEngine
PagedAttention = HandoffEngine
HandoffSimulator = HandoffEngine
AttentionEngine = HandoffEngine
Engine = HandoffEngine
Simulator = HandoffEngine
PagedKVCache = HandoffEngine
PagedAttentionExecutor = HandoffEngine
HandoffProtocol = HandoffEngine
Handoff = HandoffEngine
Executor = HandoffEngine


# ---------------------------------------------------------------------------
# 函数式入口
# ---------------------------------------------------------------------------

_CONFIG_KEYS = (
    "pages", "page_tokens", "page_size", "scratch_bytes", "cow_on_shared",
    "auto_seal", "branch_precision", "idempotent_same_append",
    "strict_events", "kv_heads", "compute_alternates",
    "auto_create_branches",
)


def run_capture(capture: Any = None,
               events: Sequence[Any] | None = None,
               **options: Any) -> AttrDict:
    """读取交接采集（dict / JSON 路径）并回放事件，返回 :class:`AttrDict`。

    第一个参数也可以直接是事件列表。可用 ``events=`` 覆盖事件（验收会重排
    交错顺序），用关键字覆盖页池配置（``pages=2, page_tokens=4`` 等）。
    """
    if capture is None:
        cap: dict[str, Any] = {}
    elif isinstance(capture, (str, Path)):
        with Path(capture).open(encoding="utf-8") as fh:
            cap = json.load(fh)
    elif isinstance(capture, dict):
        cap = capture
    elif isinstance(capture, (list, tuple)):
        # 直接传事件列表。
        cap = {}
        events = capture if events is None else events
    else:
        raise TypeError("capture 必须是路径 / dict / 事件列表")

    overrides: dict[str, Any] = {}
    for key in _CONFIG_KEYS:
        if key in options:
            overrides[key] = options.pop(key)
    # page_size -> page_tokens 归一。
    if "page_size" in overrides and "page_tokens" not in overrides:
        overrides["page_tokens"] = overrides.pop("page_size")

    eng = HandoffEngine.from_capture(cap, **overrides) if cap or overrides \
        else HandoffEngine()
    if options:
        # 剩余选项直接设置到引擎（宽容）。
        for k, v in options.items():
            if hasattr(eng, k):
                setattr(eng, k, v)
    ev = events if events is not None else cap.get("events", [])
    return eng.run_all(ev)


run_handoff = run_capture
execute_capture = run_capture
process_capture = run_capture
simulate_capture = run_capture
evaluate_capture = run_capture
run_events = run_capture
execute = run_capture
simulate = run_capture


def build_engine(capture: dict[str, Any] | str | Path | None = None,
                 **overrides: Any) -> HandoffEngine:
    if capture is None:
        return HandoffEngine(**overrides)
    if isinstance(capture, (str, Path)):
        with Path(capture).open(encoding="utf-8") as fh:
            capture = json.load(fh)
    return HandoffEngine.from_capture(capture, **overrides)


def run_capture_outputs(capture: Any = None, **options: Any):
    """只返回查询结果的 ``[(values, lse), ...]`` 列表（按事件顺序）。"""
    report = run_capture(capture, **options)
    return [(q["values"], q["lse"]) for q in report["queries"]]


def run_capture_dicts(capture: Any = None, **options: Any):
    """只返回查询结果的普通 dict 列表（ndarray 转为嵌套 list）。"""
    report = run_capture(capture, **options)
    return to_jsonable(report["queries"])


engine_from_capture = build_engine
create_engine = build_engine


# ---------------------------------------------------------------------------
# 纯参考注意力
# ---------------------------------------------------------------------------


def reference_attention(keys, values, q, heads, d=None,
                        window=None, pos=None):
    """逻辑 K/V 序列的 float64 参考实现（测试 / 对拍用，可能为全键分数表）。"""
    q = np.asarray(q, dtype=np.float64)
    H = len(heads)
    if q.ndim == 1:
        q = q[None, :]
    if d is None:
        d = q.shape[1]
    n = len(keys)
    if pos is None:
        pos = [n - 1] * H
    pos_h = HandoffEngine._broadcast_scalar(pos, H, int)
    win_h = HandoffEngine._broadcast_window(window, H)
    out = np.zeros((H, d))
    lse = np.full(H, NEG_INF)
    for h in range(H):
        hi = min(pos_h[h], n - 1)
        lo = 0 if win_h[h] is None else max(
            0, pos_h[h] - int(win_h[h]) + 1)
        if hi < lo or hi < 0:
            continue
        kvh = heads[h]
        K = np.stack([np.asarray(keys[p], dtype=np.float64)[kvh]
                      for p in range(lo, hi + 1)])
        V = np.stack([np.asarray(values[p], dtype=np.float64)[kvh]
                      for p in range(lo, hi + 1)])
        s = K @ q[h] / math.sqrt(d)
        m = s.max()
        e = np.exp(s - m)
        den = e.sum()
        out[h] = (e / den) @ V
        lse[h] = m + math.log(den)
    return out, lse


paged_attention_reference = reference_attention
scaled_dot_product_attention = reference_attention
# 与模块同名的可调用函数别名：直接给逻辑 K/V 序列做 float64 注意力。
paged_attention = reference_attention
attention_fn = reference_attention
run = run_capture


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()
                if k != "engine"}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        x = float(obj)
        return None if (math.isinf(x) or math.isnan(x)) else x
    if isinstance(obj, float):
        if math.isinf(obj):
            return "neg_infinity" if obj < 0 else "infinity"
        return None if math.isnan(obj) else obj
    return obj


def dumps_report(report: dict[str, Any], **kw: Any) -> str:
    return json.dumps(to_jsonable(report), ensure_ascii=False,
                      indent=2, **kw)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m paged_attention",
        description="回放交接采集并输出分页注意力验收报告 JSON")
    parser.add_argument("capture", type=Path, nargs="?", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--pages", type=int, default=None)
    parser.add_argument("--page-tokens", type=int, default=None)
    parser.add_argument("--scratch-bytes", type=int, default=None)
    parser.add_argument("--no-auto-seal", action="store_true")
    parser.add_argument("--precise-cow", action="store_true")
    parser.add_argument("--branch-precision", default=None,
                        choices=["published", "original"])
    parser.add_argument("--events-json", type=str, default=None,
                        help="用给定 JSON 事件列表覆盖采集事件")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {}
    for flag, key in (("pages", "pages"),
                      ("page_tokens", "page_tokens"),
                      ("scratch_bytes", "scratch_bytes"),
                      ("branch_precision", "branch_precision")):
        value = getattr(args, flag)
        if value is not None:
            overrides[key] = value
    if args.no_auto_seal:
        overrides["auto_seal"] = False
    if args.precise_cow:
        overrides["cow_on_shared"] = False

    if args.capture is None:
        parser.error("需要 capture 路径")
    events = None
    if args.events_json:
        events = json.loads(args.events_json)
    report = run_capture(args.capture, events=events, **overrides)
    text = dumps_report(report)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        if not args.quiet:
            print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
