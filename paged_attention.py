"""受限页池中的投机分页注意力引擎。

按 ``交接协议.md`` 实现：

* 固定容量页池，整页 copy-on-write，无读者的页可回收；
* 多分支/分叉/部分接收，被否决的草稿 token 绝不进入后续因果范围；
* ``hold`` 固定快照内容与精度，封存（INT8）走多版本表示，老读者仍按原值计算；
* GQA 头映射、按绝对位置的窗口因果边界；
* 流式（在线 softmax）逐头计算，不随页数分配临时分数表，``scratch_bytes``
  只统计数值算子的临时张量；
* 资源约束通过结构化结果显式回报（拒绝不产生副作用）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class ProtocolError(ValueError):
    """违反交接协议（位置不连续、未知引用等），属于调用方程序错误。"""


# ---------------------------------------------------------------------------
# 临时区记账：只统计数值算子的临时张量（不含交付结果与页池本身）
# ---------------------------------------------------------------------------


class ScratchTracker:
    """记录数值算子临时张量的当前/峰值字节占用。

    视图、零维标量与 Python 数值不计；交付结果和页池存储不经此记账。
    """

    def __init__(self) -> None:
        self.current_bytes = 0
        self.peak_bytes = 0

    def allocate(self, array: np.ndarray) -> np.ndarray:
        self.current_bytes += array.nbytes
        if self.current_bytes > self.peak_bytes:
            self.peak_bytes = self.current_bytes
        return array

    def free(self, array: np.ndarray) -> None:
        self.current_bytes -= array.nbytes


# ---------------------------------------------------------------------------
# 页与分支
# ---------------------------------------------------------------------------


@dataclass
class Page:
    """页池中的一页。

    ``fp_k/fp_v`` 为原始 64 位表示（封存后为 ``None``）；封存后以
    对称 INT8 码点与逐 (K/V, kv 头) 比例表示，``stored_*`` 是按实际
    存储值重建的数值基准。
    """

    pid: int
    page_tokens: int
    kv_heads: int
    d: int
    rows: int = 0
    fp_k: Optional[np.ndarray] = None
    fp_v: Optional[np.ndarray] = None
    sealed: bool = False
    int_k: Optional[np.ndarray] = None
    int_v: Optional[np.ndarray] = None
    scale_k: Optional[np.ndarray] = None
    scale_v: Optional[np.ndarray] = None
    referrers: set = field(default_factory=set)
    successor: Optional[int] = None  # CoW 封存后，新读者应改用的页

    def __post_init__(self) -> None:
        if self.fp_k is None:
            self.fp_k = np.zeros(
                (self.page_tokens, self.kv_heads, self.d), dtype=np.float64
            )
            self.fp_v = np.zeros_like(self.fp_k)


@dataclass
class Branch:
    name: str
    confirmed_len: int  # 已确认（可被因果查询看到）的长度
    page_table: list = field(default_factory=list)  # 逻辑页号 -> pid
    token_ids: list = field(default_factory=list)  # 每个绝对位置的 token 标识
    fork_len: int = 0  # 分叉时继承的长度
    is_draft: bool = False  # 分叉产生的草稿：分叉点后内容仍是未决候选
    alive: bool = True
    is_hold: bool = False


# ---------------------------------------------------------------------------
# INT8 封存
# ---------------------------------------------------------------------------


def quantize_block(values: np.ndarray, tracker: ScratchTracker) -> tuple[np.ndarray, float]:
    """对称 INT8 量化：最大绝对值 / 127 取比例，零块比例 1，向偶数舍入。

    返回 (int8 码点, 比例)。输入为一维 fp64 数组。
    """
    amax = float(np.max(np.abs(values)))
    scale = 1.0 if amax == 0.0 else amax / 127.0
    scaled = tracker.allocate(np.rint(values / scale).astype(np.float64))
    codes = np.clip(scaled, -128.0, 127.0).astype(np.int8)
    tracker.free(scaled)
    return codes, scale


def dequantize(codes: np.ndarray, scale: float) -> np.ndarray:
    return codes.astype(np.float64) * scale


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------


class PagedAttentionEngine:
    def __init__(
        self,
        pages: int,
        page_tokens: int,
        scratch_bytes: int,
        d: int,
        heads: list[int],
        tokens: dict,
    ) -> None:
        if pages <= 0 or page_tokens <= 0:
            raise ProtocolError("页池容量与页大小必须为正")
        self.capacity = int(pages)
        self.page_tokens = int(page_tokens)
        self.scratch_limit = int(scratch_bytes)
        self.d = int(d)
        self.heads = [int(h) for h in heads]
        self.q_heads = len(self.heads)
        self.kv_heads = max(self.heads) + 1
        self.tokens = dict(tokens)
        # 校验 token 几何
        for token_id, payload in self.tokens.items():
            k = np.asarray(payload["k"], dtype=np.float64)
            v = np.asarray(payload["v"], dtype=np.float64)
            if k.shape != (self.kv_heads, self.d) or v.shape != (self.kv_heads, self.d):
                raise ProtocolError(f"token {token_id!r} 的 K/V 几何不符合协议")

        self.slots: list[Optional[Page]] = [None] * self.capacity
        self.branches: dict[str, Branch] = {
            "main": Branch("main", 0),
            "peer": Branch("peer", 0),
        }
        self._next_pid = 0
        self.tracker = ScratchTracker()
        self.last_rejection: Optional[dict] = None
        self.confirmed_positions: dict[str, list[int]] = {}

    # -- 基础工具 ----------------------------------------------------------

    def _allocate_page(self) -> Optional[Page]:
        """先取空槽，再回收无读者的页。"""
        for slot, page in enumerate(self.slots):
            if page is None:
                return self._new_page(slot)
        victim: Optional[Page] = None
        for page in self.slots:
            # 无读者即不可达，无论是否为封存前代页都可回收
            if page is not None and not page.referrers:
                victim = page
                break
        if victim is None:
            return None
        slot = self.slots.index(victim)
        self.slots[slot] = None
        return self._new_page(slot)

    def _new_page(self, slot: int) -> Page:
        page = Page(
            pid=self._next_pid,
            page_tokens=self.page_tokens,
            kv_heads=self.kv_heads,
            d=self.d,
        )
        self._next_pid += 1
        self.slots[slot] = page
        return page

    def _get_page(self, pid: int) -> Page:
        for page in self.slots:
            if page is not None and page.pid == pid:
                return page
        raise ProtocolError(f"引用了不存在的页 {pid}")

    def _require_branch(self, name: str, expect_alive: bool = True) -> Branch:
        branch = self.branches.get(name)
        if branch is None:
            raise ProtocolError(f"未知分支/引用 {name!r}")
        if expect_alive and not branch.alive:
            raise ProtocolError(f"引用 {name!r} 已释放")
        return branch

    def _token_kv(self, token_id: str) -> tuple[np.ndarray, np.ndarray]:
        if token_id not in self.tokens:
            raise ProtocolError(f"未知 token {token_id!r}")
        payload = self.tokens[token_id]
        return (
            np.asarray(payload["k"], dtype=np.float64),
            np.asarray(payload["v"], dtype=np.float64),
        )

    def _stored_kv(self, page: Page, row: int) -> tuple[np.ndarray, np.ndarray]:
        """该读者实际可见的存储表示。"""
        if page.sealed:
            k = np.stack(
                [dequantize(page.int_k[row, h], page.scale_k[h]) for h in range(self.kv_heads)]
            )
            v = np.stack(
                [dequantize(page.int_v[row, h], page.scale_v[h]) for h in range(self.kv_heads)]
            )
            return k, v
        return page.fp_k[row], page.fp_v[row]

    # -- 事件入口 ----------------------------------------------------------

    def execute(self, event: list) -> dict:
        kind = event[0]
        handler = getattr(self, f"_event_{kind}", None)
        if handler is None:
            raise ProtocolError(f"未知事件类型 {kind!r}")
        result = handler(*event[1:])
        result.setdefault("event", kind)
        result.setdefault("rejected", False)
        self.last_rejection = result if result["rejected"] else None
        return result

    def run(self, events: list[list]) -> list[dict]:
        return [self.execute(event) for event in events]

    # -- append ------------------------------------------------------------

    def _event_append(self, branch_name: str, pos: int, token_id: str) -> dict:
        branch = self._require_branch(branch_name)
        pos = int(pos)
        if pos < len(branch.token_ids):
            # 同位置重放：幂等成功，不得重复消耗资源
            if branch.token_ids[pos] == token_id:
                return {"branch": branch_name, "pos": pos, "duplicate": True}
            raise ProtocolError(f"位置 {pos} 已被不同 token 占用")
        if pos != len(branch.token_ids):
            raise ProtocolError(
                f"append 位置不连续：期望 {len(branch.token_ids)}，收到 {pos}"
            )

        logical_page = pos // self.page_tokens
        row_in_page = pos % self.page_tokens

        if logical_page < len(branch.page_table) and branch.page_table[logical_page] >= 0:
            # 写入已有页（半页追加）。共享或已封存 => 整页 CoW，失败整体拒绝，
            # 不消耗可见长度、不改动仍在读的快照。
            page = self._get_page(branch.page_table[logical_page])
            if len(page.referrers) > 1 or page.sealed:
                cloned = self._clone_page_for(
                    branch, page, min(page.rows, row_in_page + 1)
                )
                if cloned is None:
                    return self._reject(
                        "append",
                        reason="page_pool_full",
                        branch=branch_name,
                        pos=pos,
                    )
                page = cloned
        else:
            # 新逻辑页：可能落在稀疏页表的空洞上（-1），也可能是新增页
            page = self._allocate_page()
            if page is None:
                return self._reject(
                    "append",
                    reason="page_pool_full",
                    branch=branch_name,
                    pos=pos,
                )
            page.referrers.add(branch.name)
            if logical_page < len(branch.page_table):
                branch.page_table[logical_page] = page.pid  # 填补空洞
            else:
                while len(branch.page_table) < logical_page:
                    branch.page_table.append(-1)
                branch.page_table.append(page.pid)

        k, v = self._token_kv(token_id)
        page.fp_k[row_in_page] = k
        page.fp_v[row_in_page] = v
        page.rows = max(page.rows, row_in_page + 1)
        branch.token_ids.append(token_id)
        if branch.confirmed_len < len(branch.token_ids):
            branch.confirmed_len = len(branch.token_ids)

        result = {
            "branch": branch_name,
            "pos": pos,
            "page": page.pid,
            "visible_len": len(branch.token_ids),
        }
        self._seal_sweep()
        return result

    def _clone_page_for(self, branch: Branch, src: Page, rows: int) -> Optional[Page]:
        page = self._allocate_page()
        if page is None:
            return None
        if src.sealed:
            # 按源读者可见的实际存储值复制
            page.fp_k[:rows] = np.stack(
                [self._stored_kv(src, r)[0] for r in range(rows)]
            )
            page.fp_v[:rows] = np.stack(
                [self._stored_kv(src, r)[1] for r in range(rows)]
            )
        else:
            page.fp_k[:rows] = src.fp_k[:rows]
            page.fp_v[:rows] = src.fp_v[:rows]
        page.rows = rows
        src.referrers.discard(branch.name)
        page.referrers.add(branch.name)
        logical = next(idx for idx, pid in enumerate(branch.page_table) if pid == src.pid)
        branch.page_table[logical] = page.pid
        return page

    # -- fork --------------------------------------------------------------

    def _event_fork(self, parent_name: str, child_name: str) -> dict:
        parent = self._require_branch(parent_name)
        if child_name in self.branches:
            raise ProtocolError(f"分支 {child_name!r} 已存在")
        child = Branch(
            name=child_name,
            confirmed_len=parent.confirmed_len,
            page_table=list(parent.page_table),
            token_ids=list(parent.token_ids),
            fork_len=parent.confirmed_len,
            is_draft=True,
        )
        for pid in child.page_table:
            if pid >= 0:
                self._get_page(pid).referrers.add(child_name)
        self.branches[child_name] = child
        return {"parent": parent_name, "child": child_name, "at": parent.confirmed_len}

    # -- hold --------------------------------------------------------------

    def _event_hold(self, branch_name: str, read_id: str) -> dict:
        if read_id in self.branches:
            raise ProtocolError(f"读引用 {read_id!r} 已存在")
        source = self._require_branch(branch_name)
        hold = Branch(
            name=read_id,
            confirmed_len=source.confirmed_len,
            page_table=list(source.page_table),
            token_ids=list(source.token_ids),
            # 继承源分叉点以正确描述页内是否含草稿；is_hold 使其不参与未决判定
            fork_len=source.fork_len,
            is_hold=True,
        )
        for pid in hold.page_table:
            if pid >= 0:
                self._get_page(pid).referrers.add(read_id)
        self.branches[read_id] = hold
        return {"read_id": read_id, "source": branch_name, "fixed_len": hold.confirmed_len}

    # -- accept ------------------------------------------------------------

    def _event_accept(self, base_name: str, draft_name: str, count: int) -> dict:
        base = self._require_branch(base_name)
        draft = self._require_branch(draft_name)
        count = int(count)
        if count < 0:
            raise ProtocolError("accept count 不能为负")
        if count == 0:
            return {"base": base_name, "accepted": 0, "confirmed_len": base.confirmed_len}
        start = base.confirmed_len
        end = start + count
        if end > draft.confirmed_len:
            raise ProtocolError("接收数量超出草稿已确认长度")
        if draft.token_ids[:start] != base.token_ids[:start]:
            raise ProtocolError("草稿与基线前缀不一致，无法接收")

        first_page = start // self.page_tokens
        last_page = (end - 1) // self.page_tokens
        # 接收只沿胜出前缀推进，不复制页内容，因此不需要新槽位：
        #   - 基线已持有同一逻辑页 => 共同祖先，保持不动；
        #   - 否则直接采纳草稿在该逻辑页的页对象（CoW 保证祖先行与基线数值
        #     完全一致），把基线路由接上去并增加一个读者。
        # 被否决的草稿行虽物理保留在页内，但不在基线确认范围内，对因果不可见；
        # 基线日后在该半页追加时由 append 的整页 CoW 隔离，快照读者不受影响。
        adopted = []
        for logical in range(first_page, last_page + 1):
            src = self._get_page(draft.page_table[logical])
            if logical < len(base.page_table) and base.page_table[logical] == src.pid:
                continue  # 共同祖先
            # 页表可能稀疏（基线跳过的整页之后才接收），用 -1 占位补齐空洞
            while len(base.page_table) <= logical:
                base.page_table.append(-1)
            base.page_table[logical] = src.pid
            src.referrers.add(base.name)
            adopted.append(src.pid)

        base.token_ids = list(draft.token_ids[:end])
        base.confirmed_len = end
        self.confirmed_positions.setdefault(base_name, [])
        for pos in range(start, end):
            if pos not in self.confirmed_positions[base_name]:
                self.confirmed_positions[base_name].append(pos)

        result = {
            "base": base_name,
            "draft": draft_name,
            "accepted": count,
            "range": [start, end - 1],
            "confirmed_len": end,
            "adopted_pages": adopted,
        }
        self._seal_sweep()
        return result

    # -- release -----------------------------------------------------------

    def _event_release(self, ref_name: str) -> dict:
        ref = self._require_branch(ref_name)
        freed_pages = []
        for pid in ref.page_table:
            if pid < 0:
                continue  # 稀疏页表空洞，无页可释放
            page = self._get_page(pid)
            page.referrers.discard(ref_name)
            if not page.referrers:
                freed_pages.append(pid)
        ref.alive = False
        ref.page_table = []
        ref.token_ids = []
        result = {
            "id": ref_name,
            "reclaimable_pages": sorted(freed_pages),
            # 引用一经释放即失效，后续再以该 id 操作属于协议错误；
            # 但无读者页已进入可回收集合，新的 append/fork 可立即复用其槽位。
            "can_continue": False,
            "free_slots_after": sum(1 for p in self.slots if p is None),
        }
        self._seal_sweep()
        return result

    # -- query -------------------------------------------------------------

    def _event_query(
        self, ref_name: str, pos: int, window: int, q: list
    ) -> dict:
        ref = self._require_branch(ref_name)
        pos = int(pos)
        window = int(window)
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.shape != (self.q_heads, self.d):
            raise ProtocolError("Q 几何应为 [q_heads, d]")
        if window <= 0:
            raise ProtocolError("window 必须为正")

        output = np.zeros((self.q_heads, self.d), dtype=np.float64)
        lse = np.full(self.q_heads, -np.inf, dtype=np.float64)

        if pos < 0 or pos >= ref.confirmed_len:
            # 无可用键：零向量 + 负无穷 LSE
            return {
                "ref": ref_name,
                "pos": pos,
                "output": output,
                "lse": lse,
                "empty": True,
                "visible_keys": 0,
            }

        lo = max(0, pos - window + 1)
        visible = 0
        for qh in range(self.q_heads):
            kvh = self.heads[qh]
            m = -np.inf
            acc = self.tracker.allocate(np.zeros(self.d, dtype=np.float64))
            denom = 0.0
            for key_pos in range(lo, pos + 1):
                if key_pos >= ref.confirmed_len:
                    continue
                logical = key_pos // self.page_tokens
                row = key_pos % self.page_tokens
                if logical >= len(ref.page_table) or ref.page_table[logical] < 0:
                    continue  # 稀疏页表空洞：该位置对当前引用不可见
                page = self._get_page(ref.page_table[logical])
                k_row, v_row = self._stored_kv(page, row)
                key = k_row[kvh]
                val = v_row[kvh]
                score = float(np.dot(q_arr[qh], key) / math.sqrt(self.d))
                if score > m:
                    correction = math.exp(m - score)
                    denom = denom * correction + 1.0
                    acc *= correction
                    m = score
                else:
                    denom += math.exp(score - m)
                # 逐元素累加，避免分配完整分数表/广播临时张量
                for j in range(self.d):
                    acc[j] += math.exp(score - m) * val[j]
                visible += 1 if qh == 0 else 0
            if denom > 0.0:
                output[qh] = acc / denom
                lse[qh] = m + math.log(denom)
            self.tracker.free(acc)

        return {
            "ref": ref_name,
            "pos": pos,
            "output": output,
            "lse": lse,
            "empty": False,
            "visible_keys": visible,
            "scratch_bytes_peak": self.tracker.peak_bytes,
        }

    # -- 封存 --------------------------------------------------------------

    def _page_is_stable_full(self, page: Page) -> bool:
        """整页稳定才可封存，三个条件同时成立：

        1. 页已满（半页仍可能被追加改写有效成员）；
        2. 每个引用者都能看到页内全部位置（部分接收只推进半截时不满足）；
        3. 页内没有任何未决草稿 token——即不存在仍有机会被部分接收或放弃、
           从而改写该页有效成员的分叉内容。
        """
        if page.sealed or page.rows < self.page_tokens or page.successor is not None:
            return False
        for ref_name in page.referrers:
            ref = self.branches[ref_name]
            # 页表位置即逻辑页号，页表值是 pid；页表可能稀疏，必须枚举
            logical = next(
                (idx for idx, pid in enumerate(ref.page_table) if pid == page.pid),
                None,
            )
            if logical is None:
                return False
            page_start = logical * self.page_tokens
            end_pos = (logical + 1) * self.page_tokens
            if ref.confirmed_len < end_pos:
                return False  # 条件 2：引用者还看不全整页（部分接收只到半截）
            # 条件 3：页内不得含未决草稿。只有存活的草稿分支才携带未决内容
            # （hold 固定快照、不参与判定；根会话本身就是定稿序列）。若该页
            # 与分叉后区间 [fork_len, confirmed_len) 相交，则仍有被部分
            # 接收/放弃的机会，不能封存。
            if ref.is_draft and not ref.is_hold:
                if page_start < ref.confirmed_len and end_pos > ref.fork_len:
                    return False
        return True

    def _seal_sweep(self) -> None:
        for page in list(self.slots):
            if page is None or not self._page_is_stable_full(page):
                continue
            branch_refs = [r for r in page.referrers if not self.branches[r].is_hold]
            if len(page.referrers) == 1 and branch_refs:
                self._seal_in_place(page)
            elif branch_refs:
                # 多读者或快照在持：CoW 出新版本，仅改分支读者，老读者保留原表示
                owner_name = branch_refs[0]
                owner = self.branches[owner_name]
                logical = next(
                    idx for idx, pid in enumerate(owner.page_table) if pid == page.pid
                )
                sealed_page = self._seal_as_new_page(page)
                if sealed_page is None:
                    continue  # 池满：保留 fp，下次再试，绝不改变老读者数值
                page.successor = sealed_page.pid
                owner.page_table[logical] = sealed_page.pid
                page.referrers.discard(owner_name)
                sealed_page.referrers.add(owner_name)
            # 仅剩 hold 读者时不动：快照固定的就是封存前的表示

    def _write_int8_payload(self, page: Page) -> None:
        int_k = np.zeros((self.page_tokens, self.kv_heads, self.d), dtype=np.int8)
        int_v = np.zeros_like(int_k)
        scale_k = np.ones(self.kv_heads, dtype=np.float64)
        scale_v = np.ones(self.kv_heads, dtype=np.float64)
        for h in range(self.kv_heads):
            int_k[:, h], scale_k[h] = quantize_block(page.fp_k[:, h].copy(), self.tracker)
            int_v[:, h], scale_v[h] = quantize_block(page.fp_v[:, h].copy(), self.tracker)
        page.int_k, page.int_v = int_k, int_v
        page.scale_k, page.scale_v = scale_k, scale_v
        page.sealed = True
        page.fp_k = page.fp_v = None

    def _seal_in_place(self, page: Page) -> None:
        self._write_int8_payload(page)

    def _seal_as_new_page(self, src: Page) -> Optional[Page]:
        page = self._allocate_page()
        if page is None:
            return None
        page.fp_k[:] = src.fp_k
        page.fp_v[:] = src.fp_v
        page.rows = src.rows
        self._write_int8_payload(page)
        return page

    # -- 报告 --------------------------------------------------------------

    def _reject(self, event: str, reason: str, **context) -> dict:
        rejection = {
            "event": event,
            "rejected": True,
            "reason": reason,
            "scratch_bytes_peak": self.tracker.peak_bytes,
            **context,
        }
        self.last_rejection = rejection
        return rejection

    def snapshot(self) -> dict:
        """按发布格式给出临时区与实际页占用。"""
        pages_report = []
        for slot, page in enumerate(self.slots):
            if page is None:
                pages_report.append({"slot": slot, "occupied": False})
                continue
            entry = {
                "slot": slot,
                "occupied": True,
                "pid": page.pid,
                "rows": page.rows,
                "sealed": page.sealed,
                "referrers": sorted(page.referrers),
                "successor": page.successor,
            }
            if page.sealed:
                entry["int_k"] = page.int_k.tolist()
                entry["int_v"] = page.int_v.tolist()
                entry["scale_k"] = page.scale_k.tolist()
                entry["scale_v"] = page.scale_v.tolist()
                entry["stored_k"] = np.stack(
                    [
                        [
                            dequantize(page.int_k[r, h], page.scale_k[h])
                            for h in range(self.kv_heads)
                        ]
                        for r in range(page.rows)
                    ]
                ).tolist()
                entry["stored_v"] = np.stack(
                    [
                        [
                            dequantize(page.int_v[r, h], page.scale_v[h])
                            for h in range(self.kv_heads)
                        ]
                        for r in range(page.rows)
                    ]
                ).tolist()
            else:
                entry["fp_k"] = page.fp_k[: page.rows].tolist()
                entry["fp_v"] = page.fp_v[: page.rows].tolist()
            pages_report.append(entry)

        return {
            "capacity": self.capacity,
            "page_tokens": self.page_tokens,
            "used_pages": sum(1 for p in self.slots if p is not None),
            "free_slots": sum(1 for p in self.slots if p is None),
            "scratch_bytes_peak": self.tracker.peak_bytes,
            "scratch_bytes_limit": self.scratch_limit,
            "scratch_within_limit": self.tracker.peak_bytes <= self.scratch_limit,
            "last_rejection": self.last_rejection,
            "branches": {
                name: {
                    "alive": b.alive,
                    "is_hold": b.is_hold,
                    "confirmed_len": b.confirmed_len,
                    "page_table": list(b.page_table),
                    "token_ids": list(b.token_ids),
                }
                for name, b in self.branches.items()
            },
            "confirmed_positions": {k: list(v) for k, v in self.confirmed_positions.items()},
            "pages": pages_report,
        }


def run_capture(capture: dict) -> tuple[PagedAttentionEngine, list[dict]]:
    """读取一次交接采集并按序执行，返回引擎与逐事件结果。"""
    engine = PagedAttentionEngine(
        pages=capture["pages"],
        page_tokens=capture["page_tokens"],
        scratch_bytes=capture["scratch_bytes"],
        d=capture["d"],
        heads=capture["heads"],
        tokens=capture["tokens"],
    )
    results = engine.run(capture["events"])
    return engine, results


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and np.isneginf(value):
        return "-Infinity"
    raise TypeError(f"无法序列化 {type(value)!r}")


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="受限页池投机分页注意力引擎")
    parser.add_argument("capture", type=Path, help="交接采集 JSON")
    parser.add_argument("--report", action="store_true", help="末尾输出完整占用/临时区报告")
    args = parser.parse_args(argv)

    with args.capture.open(encoding="utf-8") as stream:
        capture = json.load(stream)
    engine, results = run_capture(capture)
    for index, (event, result) in enumerate(zip(capture["events"], results), start=1):
        printable = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in result.items()
        }
        print(f"{index:02} {event[0]:<7} {json.dumps(printable, ensure_ascii=False)}")
    if args.report:
        print(json.dumps(engine.snapshot(), ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
