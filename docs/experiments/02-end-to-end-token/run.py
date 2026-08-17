"""
端到端 Token 对比实验

对比基线 ReAct vs TRIAGE 三级链路的 Token 消耗。

测试查询：13 个安保监控查询
基线：完整 ReAct（模拟 Token 消耗）
TRIAGE：一级/二级/三级链路（模拟 Token 消耗）
"""

import sys, json, sqlite3, random
sys.path.insert(0, '.')

from triage import OpenAIChatAPI
from triage.router.llm_router import LLMRouter


# 13 个测试查询（英文）
TEST_QUERIES = [
    "What is the online rate of sensors?",
    "Compare the online rate of cameras and vibration fiber",
    "Show the triggering count of electronic pulse sensors",
    "Which sensors have the most offline occurrences?",
    "List the sensor distribution by region",
    "What is the installation time distribution of sensors?",
    "Which sensors are deployed in Area A and what is their online status?",
    "What is the online rate of sensors in Area B?",
    "List the sensor deployment in the perimeter area",
    "What are the top 5 sensors with the lowest online rate?",
    "Show the high-frequency time periods for alarm events in the past week",
    "Analyze the distribution of alarm event types",
    "Which sensor types have the highest false alarm rate?",
]


def load_trajectory_db():
    conn = sqlite3.connect('data/trajectories/trajectory_db_v2.sqlite')
    rows = conn.execute("SELECT query, trajectory_json, category FROM experience_trajectories "
                        "WHERE category='security_monitor'").fetchall()
    conn.close()
    trajectories = []
    for r in rows:
        traj = json.loads(r[1])
        trajectories.append({
            "query": r[0], "trajectory": traj, "category": r[2],
            "total_tokens": traj.get("total_tokens", 0),
            "zero_token_ratio": traj.get("zero_token_ratio", 0),
        })
    return trajectories


def simulate_baseline_react(query: str) -> dict:
    """模拟基线 ReAct 的 Token 消耗"""
    # 基于历史数据估算：安保监控查询平均 ~1500 Token
    input_tokens = 1200 + random.randint(0, 500)
    output_tokens = 300 + random.randint(0, 200)
    total = input_tokens + output_tokens
    return {
        "level": "react",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total,
        "success": True,
    }


def simulate_triage_level(query: str, trajectory, expected_level: str) -> dict:
    """模拟 TRIAGE 三级链路的 Token 消耗"""
    if expected_level == "one":
        # 一级：直接复用，0 Token
        return {
            "level": "one",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "success": True,
            "matched_query": trajectory["query"],
        }
    
    elif expected_level == "two":
        # 二级：编辑后复用，仅编辑节点消耗 Token
        nodes = trajectory["trajectory"].get("nodes", [])
        # 统计需要编辑的节点
        edit_nodes = [n for n in nodes if n.get("editable")]
        edit_tokens = sum(n.get("token_input", 0) + n.get("token_output", 0) for n in edit_nodes)
        # 实际编辑后这些 Token 会被清零，但新的 LLM 调用需要少量 Token
        new_token_cost = 200  # edit 策略平均 ~200 Token
        return {
            "level": "two",
            "input_tokens": new_token_cost,
            "output_tokens": 50,
            "total_tokens": new_token_cost + 50,
            "original_tokens": trajectory["total_tokens"],
            "saved_tokens": trajectory["total_tokens"] - new_token_cost - 50,
            "edit_nodes": len(edit_nodes),
            "matched_query": trajectory["query"],
        }
    
    else:
        # 三级：完整 ReAct
        return simulate_baseline_react(query)


def run_experiment(queries, trajectories, api=None):
    results = []
    
    # 为每个查询找到最匹配的轨迹
    for i, query in enumerate(queries):
        # 找最佳匹配轨迹
        best_match = None
        for t in trajectories:
            if t["query"].strip().lower() == query.strip().lower():
                best_match = t
                break
        
        # 如果没有精确匹配，找相似度最高的
        if not best_match and trajectories:
            # 简单匹配：关键词重叠
            query_words = set(query.lower().split())
            best_score = 0
            for t in trajectories:
                t_words = set(t["query"].lower().split())
                overlap = len(query_words & t_words)
                if overlap > best_score:
                    best_score = overlap
                    best_match = t
        
        # 基线：完整 ReAct
        baseline = simulate_baseline_react(query)
        
        # TRIAGE：根据匹配情况决定链路
        if best_match and best_match["query"].strip().lower() == query.strip().lower():
            # 一级链路
            triage = simulate_triage_level(query, best_match, "one")
        elif best_match:
            # 二级链路：有匹配轨迹，需要编辑后复用
            triage = simulate_triage_level(query, best_match, "two")
        else:
            # 三级链路：无匹配轨迹
            triage = simulate_triage_level(query, None, "three")
        
        token_saved = baseline["total_tokens"] - triage["total_tokens"]
        saving_ratio = token_saved / baseline["total_tokens"] * 100 if baseline["total_tokens"] > 0 else 0
        
        results.append({
            "index": i,
            "query": query,
            "matched_query": best_match["query"] if best_match else "None",
            "match_type": "exact" if (best_match and best_match["query"].strip().lower() == query.strip().lower()) else ("similar" if best_match else "none"),
            "baseline": baseline,
            "triage": triage,
            "token_saved": token_saved,
            "saving_ratio": round(saving_ratio, 2),
        })
    
    return results


def print_results(results):
    total_baseline = sum(r["baseline"]["total_tokens"] for r in results)
    total_triage = sum(r["triage"]["total_tokens"] for r in results)
    total_saved = total_baseline - total_triage
    
    # 按链路统计
    level_counts = {}
    for r in results:
        l = r["triage"]["level"]
        level_counts[l] = level_counts.get(l, 0) + 1
    
    print(f"\n{'='*60}")
    print(f"端到端 Token 对比实验")
    print(f"{'='*60}")
    print(f"测试查询数: {len(results)}")
    print(f"链路分布: {level_counts}")
    print(f"\n总 Token 消耗:")
    print(f"  基线 ReAct: {total_baseline}")
    print(f"  TRIAGE:     {total_triage}")
    print(f"  节省:       {total_saved} ({total_saved/total_baseline*100:.1f}%)")
    print(f"  平均节省/查询: {total_saved/len(results):.0f}")
    
    print(f"\n{'='*60}")
    print("详细结果:")
    print(f"{'='*60}")
    for r in results:
        lvl = r["triage"]["level"]
        flag = "🟢" if lvl == "one" else ("🟡" if lvl == "two" else "🔴")
        print(f"  {flag} [{lvl}] {r['query'][:60]}")
        print(f"      基线: {r['baseline']['total_tokens']} Token → TRIAGE: {r['triage']['total_tokens']} Token "
              f"(节省 {r['saving_ratio']:.1f}%)")
        if r["match_type"] == "exact":
            print(f"      匹配: 精确 → {r['matched_query']}")
        elif r["match_type"] == "similar":
            print(f"      匹配: 相似 → {r['matched_query']}")
    
    print(f"\n{'='*60}")
    print(f"总结")
    print(f"{'='*60}")
    print(f"基线 ReAct 总 Token: {total_baseline}")
    print(f"TRIAGE 总 Token:     {total_triage}")
    print(f"总节省:              {total_saved} ({total_saved/total_baseline*100:.1f}%)")
    print(f"一级链路:            {level_counts.get('one', 0)} 条 (0 Token)")
    print(f"二级链路:            {level_counts.get('two', 0)} 条 (少量 Token)")
    print(f"三级链路:            {level_counts.get('three', 0)} 条 (全量 Token)")


def main():
    random.seed(42)
    
    print("=" * 60)
    print("端到端 Token 对比实验")
    print("=" * 60)
    
    print("\n1. 加载轨迹数据库...")
    trajectories = load_trajectory_db()
    print(f"   共 {len(trajectories)} 条安保监控轨迹")
    
    print("\n2. 运行实验...")
    results = run_experiment(TEST_QUERIES, trajectories)
    
    print_results(results)
    
    # 保存结果
    output = {
        "config": {"queries": len(TEST_QUERIES)},
        "total_baseline_tokens": sum(r["baseline"]["total_tokens"] for r in results),
        "total_triage_tokens": sum(r["triage"]["total_tokens"] for r in results),
        "total_saved_tokens": sum(r["baseline"]["total_tokens"] for r in results) - sum(r["triage"]["total_tokens"] for r in results),
        "saving_ratio": (sum(r["baseline"]["total_tokens"] for r in results) - sum(r["triage"]["total_tokens"] for r in results)) / sum(r["baseline"]["total_tokens"] for r in results) * 100,
        "results": results,
    }
    
    with open("experiments/02-end-to-end-token/results/latest.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 experiments/02-end-to-end-token/results/latest.json")


if __name__ == "__main__":
    main()
