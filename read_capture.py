import argparse
import json
from pathlib import Path


def read_capture(path):
    """读取一次实验交接采集，保留原始事件顺序。"""
    with Path(path).open(encoding="utf-8") as stream:
        capture = json.load(stream)
    events = capture.get("events") if isinstance(capture, dict) else None
    if not isinstance(events, list) or any(
        not isinstance(event, list) or not event or not isinstance(event[0], str)
        for event in events
    ):
        raise ValueError("采集资料含有无效的事件列表")
    return capture


# ---------------------------------------------------------------------------
# 分页注意力执行能力（规范实现见 paged_attention.py）。
# 在这里再导出常用入口，使 ``from read_capture import HandoffEngine`` 等
# 写法同样可用。
# ---------------------------------------------------------------------------
from paged_attention import (  # noqa: E402
    AttrDict,
    AttentionEngine,
    Engine,
    HandoffEngine,
    HandoffSimulator,
    Page,
    PagePool,
    PagedAttention,
    PagedAttentionEngine,
    PagedKVCache,
    QueryLease,
    ScratchLedger,
    build_engine,
    dumps_report,
    execute_capture,
    quantize_symmetric_int8,
    reference_attention,
    run_capture,
    run_capture_dicts,
    run_capture_outputs,
    run_events,
    run_handoff,
    symmetric_int8_scale,
    to_jsonable,
)


def execute_read_capture(path, **options):
    """读取并执行一个采集文件，返回完整报告。"""
    capture = read_capture(path)
    return run_capture(capture, **options)


# 额外的语义别名。
load_and_run = execute_read_capture
process = execute_read_capture
execute = run_capture
run = run_capture


def _print_run_report(capture, out=None, **overrides):
    report = run_capture(capture, **overrides)
    text = dumps_report(report)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return report


def main(argv=None):
    """命令行入口。默认仍打印事件名（18 行）；``--run`` 输出 JSON 报告。"""
    parser = argparse.ArgumentParser(description="查看交接资料的事件顺序")
    parser.add_argument("capture", type=Path)
    parser.add_argument("--run", "--execute", action="store_true",
                        help="实际执行分页注意力并输出 JSON 验收报告")
    parser.add_argument("--json", action="store_true",
                        help="等同于 --run：输出 JSON 报告")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--pages", type=int, default=None)
    parser.add_argument("--page-tokens", type=int, default=None)
    parser.add_argument("--scratch-bytes", type=int, default=None)
    parser.add_argument("--precise-cow", action="store_true")
    parser.add_argument("--branch-precision",
                        choices=["published", "original"], default=None)
    args = parser.parse_args(argv)
    capture = read_capture(args.capture)

    if args.run or args.json:
        overrides = {}
        if args.pages is not None:
            overrides["pages"] = args.pages
        if args.page_tokens is not None:
            overrides["page_tokens"] = args.page_tokens
        if args.scratch_bytes is not None:
            overrides["scratch_bytes"] = args.scratch_bytes
        if args.precise_cow:
            overrides["cow_on_shared"] = False
        if args.branch_precision:
            overrides["branch_precision"] = args.branch_precision
        _print_run_report(capture, out=args.out, **overrides)
    else:
        for index, event in enumerate(capture["events"], start=1):
            print(f"{index:02} {event[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
