"""
LLM 路由准确率实验

先跑规则路由基线（不消耗 Token），再跑 LLM 路由（需要网络）。
"""

import sys, json, sqlite3, random, time
sys.path.insert(0, '.')

from triage import OpenAIChatAPI
from triage.router.llm_router import LLMRouter, RouterDecision


def load_trajectory_db():
    conn = sqlite3.connect('data/trajectories/trajectory_db_v2.sqlite')
    rows = conn.execute("SELECT query, trajectory_json, category FROM experience_trajectories").fetchall()
    conn.close()
    trajectories = []
    for r in rows:
        traj = json.loads(r[1])
        trajectories.append({"query": r[0], "trajectory": traj, "category": r[2]})
    return trajectories


def build_test_cases(trajectories):
    by_category = {}
    for t in trajectories:
        cat = t["category"]
        by_category.setdefault(cat, []).append(t)
    
    test_cases = []
    security = by_category.get("security_monitor", [])
    
    # 一级测试：完全相同
    for t in security[:10]:
        test_cases.append({
            "query": t["query"], "retrieved": [t],
            "expected_level": "one", "type": "exact_match",
        })
    
    # 二级测试：相似但参数不同
    param_variants = [
        ("各类传感器的在线率是多少？请分析", "各类摄像头的在线率是多少？请分析"),
        ("哪些传感器离线最多？", "哪些摄像头离线最多？"),
        ("A区有哪些传感器？在线情况如何", "A区有哪些摄像头？在线情况如何"),
        ("B区传感器的在线率统计", "A区传感器的在线率统计"),
        ("在线率最低的传感器排名", "在线率最高的传感器排名"),
        ("告警事件类型分布分析", "告警事件区域分布分析"),
    ]
    for orig_query, variant_query in param_variants:
        matched = [t for t in security if t["query"] == orig_query]
        if matched:
            test_cases.append({
                "query": variant_query, "retrieved": matched,
                "expected_level": "two", "type": "param_variant",
                "original_query": orig_query,
            })
    
    # 三级测试：跨类别
    other_cats = [c for c in by_category.keys() if c != "security_monitor"]
    random.seed(42)
    for cat in other_cats[:5]:
        samples = by_category[cat][:2]
        for s in samples:
            test_cases.append({
                "query": s["query"], "retrieved": security[:3],
                "expected_level": "three", "type": "cross_category",
                "query_category": cat,
            })
    
    return test_cases


def run_rules_only(test_cases):
    """只跑规则路由（不消耗 Token）"""
    router = LLMRouter(None)  # 不需要 API
    results = []
    for i, case in enumerate(test_cases):
        query = case["query"]
        retrieved = case["retrieved"]
        expected = case["expected_level"]
        
        rule_decision = router.route_no_llm(query, retrieved, threshold=0.85)
        
        results.append({
            "index": i, "query": query[:80],
            "type": case["type"], "expected": expected,
            "rule_level": rule_decision.level,
            "rule_correct": rule_decision.level == expected,
            "rule_confidence": rule_decision.confidence,
        })
    return results


def run_llm_router(test_cases, api, max_retries=3):
    """跑 LLM 路由（需要网络，带重试）"""
    router = LLMRouter(api, top_k=3)
    results = []
    
    for i, case in enumerate(test_cases):
        query = case["query"]
        retrieved = case["retrieved"]
        expected = case["expected_level"]
        
        print(f"  [{i+1}/{len(test_cases)}] {query[:50]}...", end=" ", flush=True)
        
        for attempt in range(max_retries):
            try:
                decision = router.route(query, retrieved)
                results.append({
                    "index": i, "query": query[:80],
                    "type": case["type"], "expected": expected,
                    "llm_level": decision.level,
                    "llm_correct": decision.level == expected,
                    "llm_confidence": decision.confidence,
                    "llm_reasoning": decision.reasoning[:100],
                })
                flag = "✅" if decision.level == expected else "❌"
                print(f"{flag} (expect={expected}, got={decision.level})")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"⏳ (retry {attempt+1}/{max_retries}, wait {wait}s)...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    print(f"❌ (failed: {str(e)[:50]})")
                    results.append({
                        "index": i, "query": query[:80],
                        "type": case["type"], "expected": expected,
                        "llm_level": "error", "llm_correct": False,
                        "llm_error": str(e)[:100],
                    })
    
    return results


def print_comparison(rule_results, llm_results=None):
    print(f"\n{'='*60}")
    print(f"规则路由结果（0 Token 消耗）")
    print(f"{'='*60}")
    total = len(rule_results)
    rule_correct = sum(1 for r in rule_results if r["rule_correct"])
    print(f"总测试数: {total}")
    print(f"规则路由准确率: {rule_correct}/{total} = {rule_correct/total*100:.1f}%")
    
    for test_type in sorted(set(r["type"] for r in rule_results)):
        subset = [r for r in rule_results if r["type"] == test_type]
        c = sum(1 for r in subset if r["rule_correct"])
        print(f"  [{test_type}] {c}/{len(subset)} = {c/len(subset)*100:.1f}%")
    
    if llm_results:
        print(f"\n{'='*60}")
        print(f"LLM 路由结果（消耗 Token）")
        print(f"{'='*60}")
        llm_correct = sum(1 for r in llm_results if r.get("llm_correct"))
        print(f"LLM 路由准确率: {llm_correct}/{len(llm_results)} = {llm_correct/len(llm_results)*100:.1f}%")
        
        for test_type in sorted(set(r["type"] for r in llm_results)):
            subset = [r for r in llm_results if r["type"] == test_type]
            c = sum(1 for r in subset if r.get("llm_correct"))
            print(f"  [{test_type}] {c}/{len(subset)} = {c/len(subset)*100:.1f}%")
        
        # 对比
        print(f"\n{'='*60}")
        print(f"对比总结")
        print(f"{'='*60}")
        print(f"规则路由: {rule_correct}/{total} = {rule_correct/total*100:.1f}% (0 Token)")
        print(f"LLM 路由: {llm_correct}/{len(llm_results)} = {llm_correct/len(llm_results)*100:.1f}% (消耗 Token)")
    
    print(f"\n详细结果:")
    for r in rule_results[:30]:
        flag = "✅" if r["rule_correct"] else "❌"
        llm_flag = ""
        if llm_results:
            lr = next((x for x in llm_results if x["index"] == r["index"]), None)
            if lr:
                llm_flag = " ✅" if lr.get("llm_correct") else " ❌"
        print(f"  {flag}{llm_flag} [{r['type']:15}] expect={r['expected']} "
              f"rule={r['rule_level']} | {r['query'][:50]}")


def main():
    print("=" * 60)
    print("LLM 路由 vs 规则路由 准确率实验")
    print("=" * 60)
    
    print("\n1. 加载轨迹数据库...")
    trajectories = load_trajectory_db()
    print(f"   共 {len(trajectories)} 条轨迹")
    
    print("\n2. 构建测试集...")
    test_cases = build_test_cases(trajectories)
    print(f"   共 {len(test_cases)} 个测试用例")
    for t in sorted(set(c["type"] for c in test_cases)):
        cnt = sum(1 for c in test_cases if c["type"] == t)
        print(f"     [{t}] {cnt} 条")
    
    print("\n3. 运行规则路由（0 Token）...")
    rule_results = run_rules_only(test_cases)
    
    # 尝试 LLM 路由
    llm_results = None
    print("\n4. 尝试 LLM 路由（需要网络）...")
    try:
        api = OpenAIChatAPI()
        llm_results = run_llm_router(test_cases, api)
    except Exception as e:
        print(f"   LLM 路由初始化失败: {e}")
    
    print_comparison(rule_results, llm_results)
    
    # 保存结果
    output = {
        "config": {"threshold": 0.85, "top_k": 3},
        "total": len(test_cases),
        "rule_accuracy": sum(1 for r in rule_results if r["rule_correct"]) / len(rule_results),
        "rule_results": rule_results,
        "llm_available": llm_results is not None,
    }
    if llm_results:
        output["llm_accuracy"] = sum(1 for r in llm_results if r.get("llm_correct")) / len(llm_results)
        output["llm_results"] = llm_results
        output["token_stats"] = api.get_token_stats()
    
    with open("experiments/01-llm-router-accuracy/results/latest.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 experiments/01-llm-router-accuracy/results/latest.json")


if __name__ == "__main__":
    main()
