"""LangGraph 版 Agent —— 与 agent_eval.py 的手写版做对照。
纪律：工具实现/判分器都不改，只换"谁在跑这个循环"。
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_eval import MODEL, TOOL_IMPL                      # 复用原实现

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph_impl.trace_logger import TraceLogger          # ← 新增


# 工具名必须和原版一致（judge 是按名字判断的），所以函数名就叫原名
@tool
def calculator(expression: str) -> str:
    """计算数学表达式的值。凡是需要做算术运算时都必须使用它，不要自己心算。"""
    return TOOL_IMPL["calculator"](expression)


@tool
def get_current_time() -> str:
    """获取当前的日期和时间。当问题涉及'现在''今天''几点'等时间信息时使用。"""
    return TOOL_IMPL["get_current_time"]()


@tool
def search_knowledge(query: str) -> str:
    """检索公司内部知识库，用于回答关于公司政策、流程、规则的问题。"""
    return TOOL_IMPL["search_knowledge"](query)

def run_agent_lg(question, max_rounds=4, case_id=None):
    """返回 (answer, call_log)。行为与手写版对齐，但每一步都有 trace。"""
    logger = TraceLogger(case_id=case_id)

    llm = ChatOpenAI(model=MODEL, temperature=0,
                     api_key=os.environ.get("DEEPSEEK_API_KEY"),
                     base_url="https://api.deepseek.com")
    llm_with_tools = llm.bind_tools([calculator, get_current_time, search_knowledge])
    tool_map = {"calculator": calculator,
                "get_current_time": get_current_time,
                "search_knowledge": search_knowledge}

    messages = [HumanMessage(content=question)]
    call_log = []

    for rnd in range(max_rounds):
        t0 = time.time()
        ai = llm_with_tools.invoke(messages)
        ms = int((time.time() - t0) * 1000)

        logger.log("llm", elapsed_ms=ms, ok=True,
                   result_len=len(ai.content or ""),
                   extra={"round": rnd + 1,
                          "n_tool_calls": len(ai.tool_calls or [])})
        messages.append(ai)

        if not ai.tool_calls:
            logger.log("final", ok=True, result_len=len(ai.content or ""),
                       extra={"answer": (ai.content or "")[:200]})
            return ai.content or "", call_log

        for tc in ai.tool_calls:
            t0 = time.time()
            try:
                res = tool_map[tc["name"]].invoke(tc["args"])
                ok = True
            except Exception as e:
                res, ok = f"错误：{e}", False
            ms = int((time.time() - t0) * 1000)

            logger.log("tool", tool_name=tc["name"], args=tc["args"],
                       elapsed_ms=ms, ok=ok, result_len=len(str(res)),
                       extra={"round": rnd + 1})
            call_log.append({"name": tc["name"], "args": tc["args"], "result": str(res)})
            messages.append(ToolMessage(content=str(res), tool_call_id=tc["id"]))

    logger.log("final", ok=False, extra={"reason": "达到最大轮数"})
    return "（达到最大轮数仍未给出最终回答）", call_log

if __name__ == "__main__":
    a, c = run_agent_lg("37 乘以 48 等于多少？", case_id="单测")
    print("回答:", a)
    for x in c:
        print("  调用:", x)