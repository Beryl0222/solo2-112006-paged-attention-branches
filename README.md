# 受限页池中的投机注意力与分支回收

`samples/capture.json` 是离线合成的交接采集，包含原始数据和接收顺序。[交接协议](交接协议.md) 描述各字段及上下游共同使用的数值口径。当前入口可读取采集文件并列出操作顺序。

`paged_attention.py` 是按协议实现的 Python 侧分页注意力执行引擎，覆盖多草稿分叉、部分接收、页池回收与 INT8 封存：

- **整页 copy-on-write**：`fork`/`hold` 共享页对象，任何分支在共享页或已封存页上的追加都先整页复制，失败则整体拒绝、无副作用，仍在读的快照数值不变。
- **因果隔离**：被否决的草稿 token 物理上可能仍在某页里，但不在胜出序列的确认范围内，查询只遍历当前读引用自己祖先的确认位置，绝不混入另一条序列。
- **零分配接收**：`accept` 沿胜出前缀采纳草稿页（CoW 已保证祖先行数值一致），池满也能完成接收；部分接收只推进基线，草稿与快照不动。
- **封存多版本**：只有整页写满、所有读者可见整页、且不含未决草稿 token 的页才能封存为对称 INT8（逐 K/V、逐头比例，向偶数舍入）。封存走 CoW，老读者继续按封存前的 fp 表示计算；以页实际存储值作为数值基准。
- **流式注意力**：逐头在线 softmax，不随页数展开 Q×K 分数表，`scratch_bytes` 只统计数值算子临时张量，峰值与可见键数量无关。
- **显式回报**：资源约束通过结构化 `rejected`/`reason` 返回；`release` 回报可回收页与引用是否可继续；`snapshot()` 按发布格式给出临时区峰值与逐槽实际占用。

环境使用 Python 3.11 和锁定的 NumPy。首次在本地准备虚拟环境后运行以下命令。

```bash
python -m pip install -r requirements.txt
python -m read_capture samples/capture.json
python -m paged_attention samples/capture.json          # 逐事件结果
python -m paged_attention samples/capture.json --report # 附页池/临时区报告
python -m unittest discover -s tests -v
```

## 引擎接口

```python
from paged_attention import PagedAttentionEngine, run_capture

engine = PagedAttentionEngine(
    pages=4, page_tokens=3, scratch_bytes=192,
    d=2, heads=[0, 0], tokens=capture["tokens"],
)
result = engine.execute(["query", "main", 3, 2, q])  # output / lse / empty
snapshot = engine.snapshot()                          # 占用、封存、临时区
```

每个事件返回一个 dict：`append`/`accept` 成功时带可见长度与确认位置，资源不足时 `rejected=True` 且 `reason="page_pool_full"`；`query` 返回逐头注意力值 `output`、对数归一化常数 `lse`，无可用键时为零向量与 `-inf` LSE；`release` 返回 `reclaimable_pages` 与 `can_continue=False`。

