"""幻觉率实验：同一批题重复跑 N 轮，统计幻觉出现频率。

为什么要做：
    项目 2 里同一道题（id 5）两次跑测分别表现为"诚实拒绝"和"编造具体时间"。
    一次观察只能说明"存在这种可能"，无法说明概率。
    本实验重复跑 5 轮，把"我见过一次幻觉"变成"这道题的编造率是 X%"。

设计：
    - 8 题 × 5 轮 = 40 次 Agent 调用 + 40 次裁判调用
    - 判分只用 LLM 裁判（关键词版在项目 2 中已被证明不可靠，不参与统计）
    - 每轮记录：是否判编造、工具调用次数、各层是否通过

用法：
    python hallucination_experiment.py              # 5 轮
    python hallucination_experiment.py --rounds 3   # 指定轮数
"""
import os
import sys
import json
from collections import defaultdict

import agent_eval as ae

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(BASE_DIR, "agent_cases.json")
RESULTS_PATH = os.path.join(BASE_DIR, "hallucination_results.json")
ROUNDS_DEFAULT = 5


def main(rounds):
    with open(CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    # 每题的每轮结果：{case_id: [轮1结果, 轮2结果, ...]}
    per_case = defaultdict(list)
    total_calls = 0

    for r in range(1, rounds + 1):
        print(f"\n{'#' * 64}")
        print(f"# 第 {r}/{rounds} 轮")
        print(f"{'#' * 64}")

        for c in cases:
            cid = c["id"]
            answer, calls = ae.run_agent(c["prompt"])
            total_calls += 1

            # 用 LLM 裁判判"是否编造"；未标记 must_not_fabricate 的题默认通过
            fab_ok, _ = ae.llm_judge_no_fabrication(c, calls, answer)

            # 其余三层（纯字符串判定，不额外花钱）
            sel_ok, _ = ae.grade_tool_selection(c, calls, answer)
            arg_ok, _ = ae.grade_args(c, calls)
            grd_ok, _ = ae.grade_answer_grounded(c, calls, answer)

            per_case[cid].append({
                "round": r,
                "scene": c["scene"],
                "answer": answer,
                "n_calls": len(calls),
                "call_names": [x["name"] for x in calls],
                "no_fabrication": fab_ok,
                "tool_selection": sel_ok,
                "args": arg_ok,
                "grounded": grd_ok,
            })
            mark = "编造" if not fab_ok else "OK"
            print(f"  [{cid}] {c['scene']:<12} 裁判:{mark:<4} "
                  f"工具调用 {len(calls)} 次")

    # ============ 统计 ============
    print("\n" + "=" * 64)
    print("【每题幻觉率】（仅对 must_not_fabricate 的题有意义）")
    print("-" * 64)
    print(f"  {'id':<4}{'场景':<14}{'编造次数':<10}{'编造率':<10}{'平均工具调用'}")
    scene_of = {c["id"]: c["scene"] for c in cases}
    watch = {c["id"] for c in cases if c.get("must_not_fabricate")}

    halluc_rows = []
    for cid in sorted(per_case):
        runs = per_case[cid]
        n_fab = sum(1 for x in runs if not x["no_fabrication"])
        rate = n_fab / len(runs)
        avg_calls = sum(x["n_calls"] for x in runs) / len(runs)
        tag = "  ← 幻觉观察题" if cid in watch else ""
        print(f"  {cid:<4}{scene_of[cid]:<14}{n_fab}/{len(runs):<8}"
              f"{rate * 100:>5.0f}%    {avg_calls:.2f}{tag}")
        halluc_rows.append({
            "id": cid, "scene": scene_of[cid],
            "fabricated": n_fab, "rounds": len(runs), "rate": rate,
            "avg_tool_calls": round(avg_calls, 2),
            "is_watch_case": cid in watch,
        })

    # ---- 只看幻觉观察题的总体编造率 ----
    print("\n" + "-" * 64)
    print("【幻觉观察题总体】")
    if watch:
        w_runs = [x for cid in watch for x in per_case[cid]]
        w_fab = sum(1 for x in w_runs if not x["no_fabrication"])
        print(f"  观察题 id {sorted(watch)}，共 {len(w_runs)} 次运行，"
              f"判编造 {w_fab} 次 → 编造率 {w_fab / len(w_runs) * 100:.1f}%")
    else:
        print("  本测试集没有标记 must_not_fabricate 的题")

    # ---- 各层的跨轮波动 ----
    print("\n" + "-" * 64)
    print("【每轮分层通过率】（看波动）")
    print(f"  {'轮次':<6}{'工具选择':<12}{'参数正确':<12}{'答案落地':<12}{'无编造(LLM)'}")
    layers_of = [("tool_selection", "工具选择"), ("args", "参数正确"),
                 ("grounded", "答案落地"), ("no_fabrication", "无编造")]
    round_stats = []
    for r in range(1, rounds + 1):
        cells = []
        stat = {"round": r}
        for key, _label in layers_of:
            runs = [x for cid in per_case for x in per_case[cid] if x["round"] == r]
            ok = sum(1 for x in runs if x[key])
            stat[key] = f"{ok}/{len(runs)}"
            cells.append(f"{ok}/{len(runs)}")
        round_stats.append(stat)
        print(f"  {r:<6}{cells[0]:<12}{cells[1]:<12}{cells[2]:<12}{cells[3]}")

    # ---- 稳定性判断 ----
    print("\n" + "-" * 64)
    print("【稳定性结论】")
    fab_by_round = []
    for r in range(1, rounds + 1):
        runs = [x for cid in watch for x in per_case[cid] if x["round"] == r] if watch else []
        ok = sum(1 for x in runs if x["no_fabrication"])
        fab_by_round.append(f"{ok}/{len(runs)}")
    print(f"  幻觉观察题每轮通过数: {' → '.join(fab_by_round)}")
    all_same = len(set(fab_by_round)) == 1
    print(f"  是否每轮一致: {'是' if all_same else '否 —— 说明结果不稳定，幻觉是概率性的'}")

    # ---- 落盘 ----
    out = {
        "model": ae.MODEL,
        "rounds": rounds,
        "total_agent_calls": total_calls,
        "per_case": {str(k): v for k, v in per_case.items()},
        "hallucination_summary": halluc_rows,
        "round_layer_stats": round_stats,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n原始结果已写入: {RESULTS_PATH}")


if __name__ == "__main__":
    rounds = ROUNDS_DEFAULT
    if "--rounds" in sys.argv:
        rounds = int(sys.argv[sys.argv.index("--rounds") + 1])
    main(rounds)
