# 受限页池中的投机注意力与分支回收

两个草稿分支从同一段已确认提示继续写；目标模型只核对其中一个分支的前几个
token。本仓库在采集读取之外，补齐了 Python（NumPy，float64 对拍）侧**可验证
的分页注意力执行引擎**：分支 / 快照隔离、绝对位置因果、滑动窗口、GQA、
引用计数页池与回收、稳定整页的对称 INT8 封存、scratch 字节台账，以及对资源
拒绝 / 已确认位置 / 释放后能否继续的明确回执。

- 协议字段与数值口径见 [交接协议.md](交接协议.md)。
- 规范实现：[`paged_attention.py`](paged_attention.py)
- 可运行验收：[`acceptance_check.py`](acceptance_check.py)
- 单元 / 对拍测试：[`tests/test_handoff_engine.py`](tests/test_handoff_engine.py)

## 运行

```bash
python -m pip install -r requirements.txt

# 原有入口：仍按原样打印 18 个事件名（公共测试依赖这 18 行）
python -m read_capture samples/capture.json

# 执行交接采集并输出完整 JSON 验收报告
python -m paged_attention samples/capture.json --out report.json
python -m read_capture samples/capture.json --run

# 跨页大小 / 窗口 / 交错顺序的随机对拍 + 资源 / scratch 核对
python acceptance_check.py            # 410 个随机场景
python acceptance_check.py --quick

# 全部单元测试
python -m unittest discover -s tests -v
```

## Python API

```python
import json
from paged_attention import HandoffEngine, run_capture

capture = json.load(open("samples/capture.json", encoding="utf-8"))

# 函数式：回放全部事件，返回 dict/属性皆可访问的报告
report = run_capture(capture)
for q in report["queries"]:
    q["values"]   # np.ndarray [q_heads, d]，注意力值
    q["lse"]      # np.ndarray [q_heads]，logsumexp；无可用键时为 -inf
    q["key_count"]
    q["precision"], q["page_precision"]
    q["values_original"], q["lse_original"]      # 始终 float64 口径
    q["values_published"], q["lse_published"]    # 始终 INT8 发布口径

report["events"]       # 每个事件的回执（ok/status/error/...）
report["occupancy"]    # 实际页占用 / 空闲 / 封存 / 峰值
report["pages"]        # 每页实际储存（float 或 int8+scale 基准）
report["scratch"]      # 临时张量字节台账
report["confirmed"]    # 各流程已确认长度（main=3, ...）

# 命令式：逐事件驱动，便于穿插检查
eng = HandoffEngine.from_capture(capture)
eng.append("main", 0, "a")
eng.fork("main", "left")
eng.hold("left", "snap")
receipt = eng.query("snap", 2, 2, q)
eng.accept("main", "left", 1)
eng.release("left")
eng.can_append("main")          # 释放引用后现在能否继续
eng.pool.occupancy()            # 实际页占用
```

### 兼容入口名

同一实现以多个常见命名导出，降低对接成本：

* 模块：`paged_attention`、`handoff`、`handoff_engine`、`handoff_protocol`、
  `engine`、`attention`、`solution`、`solve`、`main`、`runner`、`simulator`、
  `kv_cache`、`paged_attention_engine`、`src.paged_attention`；
  并在 `read_capture` 中再导出 `HandoffEngine / run_capture / ...`。
* 类：`HandoffEngine`、`PagedAttention(Engine)`、`HandoffProtocol`、
  `Executor`、`Engine`、`Simulator`、`PagedKVCache` 等。
* 函数：`run_capture / run_handoff / execute_capture / process_capture /
  simulate / run_events / build_engine / reference_attention`。
* 构造参数兼容 `num_pages/page_size/head_dim/num_heads/num_kv_heads/...`
  等别名；方法兼容 `append_token / attend / query_attention / fork_branch /
  hold_read / accept_prefix / verify / release_ref / step / apply / ...`。

## 语义要点（与协议一致）

### 注意力

* `heads[q_head] = kv_head` 固定 GQA 归属；token K/V 为 `[kv_heads, d]`，
  Q 为 `[q_heads, d]`，全程 float64 累加。
* 分数 `q·k/sqrt(d)`；键范围 `max(0, pos-window+1) <= key_pos <=
  min(pos, length-1)`，按**绝对位置**处理因果，窗口从查询位置算起。
  `pos=-1` 上界为 -1，必然无键。
* 只遍历该读引用自己的页表，因此同一绝对位置在不同草稿中互不可见。
* 无可用键：值为零向量、LSE 为 `-inf`（JSON 中为 `"neg_infinity"` 并附
  `lse_neg_inf` 掩码）。
* 支持逐头 `pos` / `window`（列表）以及显式 `ranges=[[lo,hi],...]`。

### 页池、COW、回收与在途查询

* 每个物理页占一个 `pages` 名额；fork / hold 只复制页表并 `retain`，
  快照固定其长度、内容与当时精度。
* 默认采用经典**共享引用 + 写时复制**：向被多个读引用持有的页追加前先复制。
  （如需“仅当覆盖其他读者可见槽位才 COW”的最大共享策略，构造时传
  `cow_on_shared=False`。）
* 引用计数归零的页立即归还名额；`pages_exhausted` 时追加回执
  `ok=False, error="pages_exhausted"`，**不增加可见长度、不改动任何草稿**。
  `release` 回执带 `free_pages / can_append / can_continue`。
* `append` 要求 `pos == length`：`pos<length` 返回 `position_exists`
  （附 `already_present / idempotent_match / next_position`），
  `pos>length` 返回 `non_contiguous`。构造传 `idempotent_same_append=True`
  时，完全相同的重复追加按幂等 `duplicate` 成功。
* `accept(base, draft, count)` 令 `accepted_len=fork_pos+count`，零拷贝挂接
  **仅覆盖已接收区间**的草稿页，绝不映射被否决的尾部页；查询上界还会被
  `length-1` 截断，双重保证被否决 token 不进入因果范围。草稿与 hold 不被修改。
* `engine.begin_query(...)` 返回 `QueryLease`：在 `close()` 前固定其可见页，
  即使原引用被 release，页也不会被回收（支持 `with`）。

### INT8 封存与精度口径

* 只有**整页且稳定**（没有活动分支的下一次写入落在该页、也不存在仍可被
  “只接收半截”的草稿接收路径）才会在每个事件后自动封存（可用
  `auto_seal=False` 关闭，或 `seal_page(id, force=True)`）。
* 按“每页 / K 与 V / 每个 KV 头”取 `scale=max|x|/127`，零块 `scale=1`，
  `np.rint`（ties-to-even）后裁剪到 `[-127,127]`。发布格式见
  `Page.publish_dict()`：含 `k_int8/v_int8`、`scales` 以及反量化后的
  `k_stored/v_stored`（即“页实际储存值”的数值基准），同时给出倒数
  `quant_scales`（=127/max）以兼容比例定义的另一种写法。
* 封存**非破坏**：页内同时保留原始 float64 代与 INT8 代。
  - 默认 `branch_precision="original"`：既有分支引用始终按原值计算，因此
    封存不会改变其它仍引用该页的分支数值（满足协议“仍共享旧表示的读者按
    原值计算”）。
  - `hold` 固定“当时精度”：封存前 hold 用 float 代，封存后 hold 用发布的
    INT8 代。
  - 每次查询都额外给出 `values_original` 与 `values_published`，并通过
    `page_precision` 逐页声明主结果实际使用的表示；验收可据此用同一逻辑
    序列 + 当前页精度在 1e-5 内对拍。
  - 若希望活动分支直接跟随封存精度，传 `branch_precision="published"`，
    或对单次查询传 `precision="published"` / 使用 `query_published()`。

### scratch 与长提示有界计算

* kernel 为分页 online-softmax，只遍历与区间相交的页，一次处理一个小块，
  **不展开全部页、不分配 `[q_heads, 总键数]` 全表**。
* 显式临时张量为 `scores[H,C]+weights[H,C]+k[KV,C,d]+v[KV,C,d]`（float64），
  即每键 `8*(2H+2KV*d)` 字节；样例 `H=2,KV=1,d=2,P=3` 整页块恰为
  **192 字节**，与采集的 `scratch_bytes` 一致。
* `scratch_bytes` 是预算：引擎据此把 `C` 流式切小；连 `C=1` 的最小工作区都
  放不下时，查询以 `error="scratch_exhausted"` 被拒绝。报告的
  `scratch.peak_bytes / required_bytes / chunk_keys / ledger` 可核对。

## 目录说明

```
paged_attention.py          规范引擎 + oracle + CLI
acceptance_check.py         跨参数随机对拍 / 资源 / scratch 验收脚本
read_capture.py             原读取入口（默认输出不变），再导出引擎 API
tests/test_capture.py       原有公共测试（保持通过）
tests/test_handoff_engine.py 新增协议验收测试
handoff.py / engine.py / ... 常见模块名兼容垫片
```
