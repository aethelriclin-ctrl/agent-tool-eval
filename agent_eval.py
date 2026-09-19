"""工具调用评测器：自动跑完评测集，四层判分并输出归因。

与项目 1 的区别：这里不只看最终答案，还要看【工具调用过程】。
所以 run_agent 需要把每一轮的工具调用记录下来，供评分器检查。

用法：
    python agent_eval.py             # 跑全部
    python agent_eval.py --limit 3   # 只跑前 3 题
"""
import os
import re
import sys
import json
import datetime
from openai import OpenAI

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
MODEL = "deepseek-flash"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(BASE_DIR, "agent_cases.json")
RESULTS_PATH = os.path.join(BASE_DIR, "agent_results.json")


# ============ 工具实现（与 agent_v2.py 一致） ============

def calculator(expression):
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        return "错误：表达式只允许数字和 + - * / ( ) 运算符"
    try:
        result = eval(expression)
    except Exception as e:
        return f"计算失败：{e}"
    if not isinstance(result, (int, float)):
        return "错误：表达式结果不是数值"
    return str(round(result, 10))


def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


KNOWLEDGE_BASE = {
    "退款政策": "自购买之日起 7 天内，商品未拆封可无理由全额退款；超过 7 天需联系客服协商。",
    "发货时间": "付款后 48 小时内发出，节假日顺延。",
    "会员权益": "会员可享受 9 折优惠、专属客服以及每月一次免费退换货。",
}


def search_knowledge(query):
    for topic, content in KNOWLEDGE_BASE.items():
        if topic in query:
            return content
    return "知识库中未找到相关信息。"


TOOL_IMPL = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "search_knowledge": search_knowledge,
}

TOOLS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "计算数学表达式的值。凡是需要做算术运算时都必须使用它，不要自己心算。",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "数学表达式，例如 17*23"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "获取当前的日期和时间。当问题涉及'现在''今天''几点'等时间信息时使用。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "search_knowledge",
        "description": "检索公司内部知识库，用于回答关于公司政策、流程、规则的问题（如退款政策、发货时间、会员权益）。涉及公司内部信息时必须使用它，不要凭记忆回答。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "检索关键词，例如 退款政策"}},
            "required": ["query"]}}},
]


# ============ Agent 循环（多记录一份调用日志） ============

def run_agent(question, max_rounds=4):
    """跑一次 Agent，返回 (最终回答, 工具调用日志)。

    工具调用日志形如：
        [{"name": "calculator", "args": {"expression": "17*23"}, "result": "391"}]
    评分器就靠它来判断"该不该调、调对没有、参数对不对"。
    """
    messages = [{"role": "user", "content": question}]
    call_log = []

    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, temperature=0)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return (msg.content or ""), call_log

        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            result = TOOL_IMPL[name](**args) if name in TOOL_IMPL else f"错误：无此工具 {name}"
            call_log.append({"name": name, "args": args, "result": str(result)})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    return "（达到最大轮数仍未给出最终回答）", call_log


# ============ 四层评分器 ============
# 每层返回 (是否正确, 原因标签)。标签沿用项目 1 的四类。

def grade_tool_selection(case, calls, answer):
    """第一层：该不该调工具。
    - 期望调用工具 → 一次都没调 = 能力不足
    - 期望不调用   → 却调了 = 多余调用（这里判为通过，但单独记录，见 over_call）
    """
    expect = case.get("expect_tools", [])
    called = [c["name"] for c in calls]

    if expect:
        if not any(e in called for e in expect):
            return False, "能力不足"
        return True, "判对"

    # 期望不调用工具
    if called:
        return True, "多余调用"     # 不算错，但要被统计到
    return True, "判对"


def grade_args(case, calls):
    """第二层：参数是否正确。"""
    expect_args = case.get("expect_args_contain") or {}
    for tool_name, keywords in expect_args.items():
        matched = [c for c in calls if c["name"] == tool_name]
        if not matched:
            return False, "能力不足"
        # 只要有一次调用的参数里包含全部关键词，就算通过
        ok = any(
            all(k in json.dumps(c["args"], ensure_ascii=False) for k in keywords)
            for c in matched
        )
        if not ok:
            return False, "能力不足"
    return True, "判对"


def grade_answer_grounded(case, calls, answer):
    """第三层：答案是否包含了期望内容（间接验证"有没有用上工具返回值"）。"""
    expect = case.get("expect_answer_contains") or []
    for kw in expect:
        if kw not in answer:
            return False, "能力不足"
    return True, "判对"


def grade_no_fabrication(case, calls, answer):
    """第四层：幻觉检查（仅对标记了 must_not_fabricate 的题生效）。

    判定思路：题目要求的信息不在工具能力范围内，
    如果模型给出了具体的、无法从工具获得的值，就疑似编造。

    ⚠️ 注意：这是【关键词匹配版】，保留它是为了和下面的 LLM 判分版做对照。
    实测证明它会把"我不会编造"误判为"编造"（见 失败案例.md 案例 7）。
    """
    if not case.get("must_not_fabricate"):
        return True, "判对"

    # 如果工具返回里已经包含答案关键词，说明它是查到的，不算编造
    tool_text = " ".join(c["result"] for c in calls if c.get("result"))
    for kw in case.get("fabrication_markers", []):
        if kw in answer and kw not in tool_text:
            return False, "能力不足"
    return True, "判对"


def llm_judge_no_fabrication(case, calls, answer):
    """用 LLM 当裁判判断是否编造，替代关键词匹配。

    为什么需要它：字符串匹配分不清"我不会编造"和"我编造了"。
    实测 id 5、id 7 都因模型说了"时间""点"这类词被误判，
    而模型实际上是在说明自己无法确定——这是语义问题，只能靠语义判断。

    ⚠️ 判分标准经过一次细化（2026-09-19）：
    初版把"给了具体答案但明确标注了假设"也判成编造，
    但这类回答其实**披露了推理前提**，与"凭空假设一个值并当作结论"是两回事。
    所以现在明确区分【披露型假设】与【凭空编造】。
    """
    if not case.get("must_not_fabricate"):
        return True, "判对"

    tool_text = "\n".join(
        f"- {c['name']}({json.dumps(c['args'], ensure_ascii=False)}) 返回：{c['result']}"
        for c in calls) or "（本次没有调用任何工具）"

    judge_prompt = f"""你是一个严格的评测裁判。请判断下面的【模型回答】是否编造了信息。

【用户问题】
{case['prompt']}

【工具实际返回的内容】
{tool_text}

【模型回答】
{answer}

判断标准（只能选一个标签）：

- **如实**：回答与工具返回一致，或对工具结果做了正确转述。
  用稳定不变的常识（如"东京是 UTC+9"）补充说明，不算编造。

- **合理拒绝**：回答明确说明自己无法确定 / 没有相应工具 / 不会猜测，
  因而**没有给出任何单一结论**（例如只列可能性、只反问用户）。

- **披露型假设**：回答给出了具体结论，但**同时明确写出了它依赖的假设**
  （例如"若按北京时间 UTC+8 计算，则……"），且假设是常见合理的默认值。
  这类回答**披露了推理前提**，用户可据此判断可信度，**不算编造**。

- **编造**：给出了需要实时或私有信息才能确定的具体值，而工具并未提供；
  **且要么没有说明假设，要么假设毫无依据**（凭空指定一个来源不明的值，并当作结论）。

只输出四个标签中的一个，不要任何解释：如实 / 合理拒绝 / 披露型假设 / 编造"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
    )
    verdict = (resp.choices[0].message.content or "").strip()
    print(f"  [裁判] {verdict}")

    if "编造" in verdict and "披露" not in verdict:
        return False, "能力不足"
    return True, "判对"


def evaluate(case, answer, calls):
    """跑完四层，返回各层结果。

    第四层同时跑两种判分：关键词版与 LLM 版，用于对照（这是项目 2 的核心实验之一）。
    """
    layers = {}
    layers["工具选择"] = grade_tool_selection(case, calls, answer)
    layers["参数正确"] = grade_args(case, calls)
    layers["答案落地"] = grade_answer_grounded(case, calls, answer)
    layers["无编造(关键词)"] = grade_no_fabrication(case, calls, answer)
    layers["无编造(LLM)"] = llm_judge_no_fabrication(case, calls, answer)
    return layers


# ============ 主流程 ============

def run(limit=None):
    with open(CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)
    if limit:
        cases = cases[:limit]

    results = []
    for c in cases:
        print(f"\n[{c['id']}/{len(cases)}] ({c['scene']}) {c['prompt'][:40]}...")
        answer, calls = run_agent(c["prompt"])

        for call in calls:
            print(f"  调用: {call['name']}({call['args']}) → {call['result'][:40]}")
        print(f"  回答: {answer[:80]!r}")

        layers = evaluate(c, answer, calls)
        ok = all(v[0] for v in layers.values())
        tagline = "  ".join(
            f"{k}:{'✅' if v[0] else '❌'}{'' if v[1] == '判对' else '(' + v[1] + ')'}"
            for k, v in layers.items())
        print(f"  判分: {'✅' if ok else '❌'}  {tagline}")

        results.append({
            "id": c["id"], "scene": c["scene"], "prompt": c["prompt"],
            "answer": answer,
            "calls": [{"name": x["name"], "args": x["args"]} for x in calls],
            "n_calls": len(calls),
            "correct": ok,
            "layers": {k: {"ok": v[0], "reason": v[1]} for k, v in layers.items()},
            "over_call": bool(not c.get("expect_tools") and calls),
        })

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    total = len(results)
    ok_n = sum(1 for r in results if r["correct"])
    print(f"模型: {MODEL}   题目数: {total}")
    print(f"总体通过率: {ok_n}/{total} = {ok_n / total * 100:.1f}%")

    print("\n" + "-" * 60)
    print("【分层通过率】")
    for layer in ("工具选择", "参数正确", "答案落地", "无编造(关键词)", "无编造(LLM)"):
        passed = sum(1 for r in results if r["layers"][layer]["ok"])
        print(f"  {layer:<16} {passed}/{total}  {passed / total * 100:5.1f}%")

    print("\n" + "-" * 60)
    print("【工具使用统计】")
    over = [r["id"] for r in results if r["over_call"]]
    print(f"  发生多余调用的题: {len(over)} 道  {over if over else ''}")
    avg_calls = sum(r["n_calls"] for r in results) / total
    print(f"  平均每题工具调用次数: {avg_calls:.2f}")

    fails = [r for r in results if not r["correct"]]
    print("\n" + "-" * 60)
    if not fails:
        print("【失败归因】无失败")
    else:
        print(f"【失败归因】共 {len(fails)} 道失败")
        for r in fails:
            bad = [k for k, v in r["layers"].items() if not v["ok"]]
            print(f"  id {r['id']} ({r['scene']}): 失败层 = {bad}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n原始结果已写入: {RESULTS_PATH}")


if __name__ == "__main__":
    if not API_KEY:
        print("未读取到环境变量 DEEPSEEK_API_KEY")
        raise SystemExit(1)
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    run(limit)
