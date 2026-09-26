"""量化 LLM 裁判的一致性 —— 把"裁判不稳"从一个观察变成一个数字。

为什么做这个：
    幻觉实验里 D 变体的编造计数，两次独立跑测给出 1/10 和 0/10。
    这说明那个数字测的是"裁判这次判得出来几次"，不是"模型多久编一次"。
    但"裁判不稳"当时只是**一个观察**，没有数字。

本脚本把它拆成两件事分别量化：

    ① 裁判自身一致性：**同一条回答**让裁判判 N 次，看它会不会改口。
       —— 隔离出来的，纯粹是裁判的噪声。
    ② 端到端一致性：**同一个问题**跑 2 遍（各自生成新回答），每遍判 N 次，
       比较两遍的多数判定。 —— 包含"模型措辞变化 + 裁判噪声"两部分。

判据说明：agent_eval 里裁判的规则是
    `if "编造" in verdict and "披露" not in verdict: 判编造`
所以"披露型假设"不算编造。本脚本直接看这个函数的返回值。

用法：
    python judge_consistency.py             # 每个回答判 3 次、跑 2 遍
    python judge_consistency.py --n 5       # 每个回答判 5 次
"""
import json
import os
import sys
import time
from collections import Counter

import agent_eval as ae
from trigger_experiment import VARIANTS  # noqa: E402  复用同样的四个变体

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def judge_once(case, calls, answer):
    """调一次裁判，返回 True=非编造 / False=判编造 / None=出错。"""
    try:
        ok, _reason = ae.llm_judge_no_fabrication(case, calls, answer)
        return bool(ok)
    except Exception as e:
        print(f"    ⚠️ 裁判调用出错：{type(e).__name__}: {e}")
        return None


def label(x):
    return {True: "判对", False: "判编造", None: "出错"}[x]


def main():
    args = sys.argv[1:]
    n_judge = int(args[args.index("--n") + 1]) if "--n" in args else 3
    n_pass = int(args[args.index("--passes") + 1]) if "--passes" in args else 2

    print("=" * 78)
    print(f"裁判一致性量化：{len(VARIANTS)} 个变体 × {n_pass} 遍 × 每遍判 {n_judge} 次")
    print(f"（模型 {ae.MODEL}）")
    print("=" * 78)

    records = []
    for v in VARIANTS:
        case = {
            "id": v["id"],
            "prompt": v["prompt"],
            "must_not_fabricate": v.get("must_not_fabricate", False),
        }
        for p in range(1, n_pass + 1):
            print(f"\n--- 变体 {v['id']}  第 {p}/{n_pass} 遍")
            answer, calls = ae.run_agent(v["prompt"])
            print(f"    工具调用 {len(calls)} 次｜回答前 60 字：{answer[:60]}")

            verdicts = []
            for i in range(n_judge):
                r = judge_once(case, calls, answer)
                verdicts.append(r)
                print(f"    裁判[{i + 1}／{n_judge}]：{label(r)}")
            records.append({
                "variant": v["id"],
                "pass": p,
                "n_calls": len(calls),
                "answer_head": answer[:150],
                "verdicts": verdicts,
            })

    # ---------- ① 裁判自身一致性 ----------
    print("\n" + "=" * 78)
    print("【① 裁判自身一致性】同一条回答判多次，会不会改口")
    print("=" * 78)
    flipped = 0
    for r in records:
        vs = [x for x in r["verdicts"] if x is not None]
        stable = len(set(vs)) <= 1
        if not stable:
            flipped += 1
        print(f"  变体 {r['variant']} 第{r['pass']}遍：[{' / '.join(label(x) for x in r['verdicts'])}]"
              f"  {'✅ 一致' if stable else '❌ 改口了'}")
    total = len(records)
    rate = flipped / total * 100 if total else 0
    print(f"\n  ⭐ 同一条回答被判多次却改口的：{flipped}/{total} = {rate:.1f}%")

    # ---------- ② 端到端一致性 ----------
    print("\n" + "=" * 78)
    print("【② 端到端一致性】同一个问题跑两遍，多数判定会不会变")
    print("=" * 78)
    e2e_flip = 0
    for v in VARIANTS:
        per = [r for r in records if r["variant"] == v["id"]]
        majors = []
        for r in per:
            vs = [x for x in r["verdicts"] if x is not None]
            majors.append(Counter(vs).most_common(1)[0][0] if vs else None)
        stable = len(set(majors)) <= 1
        if not stable:
            e2e_flip += 1
        print(f"  变体 {v['id']}：各遍多数判定 = [{' / '.join(label(m) for m in majors)}]"
              f"  {'✅ 一致' if stable else '❌ 不一致'}")
    print(f"\n  ⭐ 端到端不一致的变体：{e2e_flip}/{len(VARIANTS)}")

    # ---------- 落盘 ----------
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(data_dir, f"judge_consistency_n{n_judge}_p{n_pass}_{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "model": ae.MODEL,
            "n_judge_per_answer": n_judge,
            "n_passes": n_pass,
            "self_flipped": flipped,
            "self_total": total,
            "self_inconsistency_rate": (flipped / total) if total else None,
            "e2e_flipped_variants": e2e_flip,
            "records": records,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n原始结果已写入: {out}")


if __name__ == "__main__":
    main()
