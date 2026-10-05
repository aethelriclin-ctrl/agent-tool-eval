"""调用链日志：把 Agent 每一步记成一行 JSON。

为什么要记：
  · 出问题时能回答"第几步、哪一次工具调用出的错"，而不是只看最终答案；
  · 每行带 case_id，既能整体看，也能 grep 出某一题。

输出：仓库根目录的 trace.jsonl（文件名带时间戳，不覆盖）
"""
import json
import os
import datetime

# trace.jsonl 放仓库根目录（和 results_*.json 一起）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_PATH = os.path.join(
    BASE_DIR, "trace_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".jsonl")


class TraceLogger:
    """每调一次 log() 就往文件里追加一行。"""

    def __init__(self, case_id=None, path=TRACE_PATH):
        self.case_id = case_id
        self.path = path
        self.step = 0

    def log(self, kind, *, tool_name=None, args=None, elapsed_ms=None,
            ok=True, result_len=None, extra=None):
        """kind: 'llm' | 'tool' | 'final'"""
        self.step += 1
        row = {
            "step": self.step,
            "case_id": self.case_id,
            "type": kind,
            "tool_name": tool_name,
            "args": args,
            "elapsed_ms": elapsed_ms,
            "ok": ok,
            "result_len": result_len,
        }
        if extra:
            row.update(extra)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

   