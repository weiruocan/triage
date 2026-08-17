"""
跨领域验证实验

在 ToolBench 的 3 个领域（weather、search、jokes）上验证 TRIAGE 的 Token 节省效果。

实验设计：
1. 每个领域 10 条查询，共 30 条
2. 模拟 TRIAGE 的三链路分类：
   - 一级：完全相同查询（0 Token）
   - 二级：同领域相似查询（编辑复用，~250 Token）
   - 三级：跨领域查询（完整 ReAct，~800 Token）
3. 对比基线：标准 ReAct（每条查询 ~800 Token）
"""

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from collections import Counter


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_domain_data(domain):
    """加载领域数据"""
    path = os.path.join(DATA_DIR, f"{domain}.json")
    with open(path) as f:
        return json.load(f)


def compute_similarity(q1, q2):
    """计算语义相似度（基于关键词重叠）"""
    w1 = set(q1.lower().split())
    w2 = set(q2.lower().split())
    common = w1 & w2
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                 "to", "for", "of", "with", "and", "or", "but", "not", "i", "my",
                 "me", "we", "our", "can", "you", "do", "does", "could", "would",
                 "need", "want", "am", "be", "been", "it", "its", "that", "this"}
    meaningful = common - stopwords
    union = (w1 - stopwords) | (w2 - stopwords)
    if not union:
        return 0.0
    return len(meaningful) / len(union)


def classify_query(query, domain_items):
    """
    模拟 TRIAGE 路由分类
    
    规则：
    - 精确匹配 → Level 1
    - 同领域相似度 > 0.3 → Level 2
    - 否则 → Level 3
    """
    for item in domain_items:
        existing = item["query"]
        if existing == query:
            return "one", None
        
        sim = compute_similarity(query, existing)
        if sim > 0.3:
            return "two", {"matched_trajectory": existing[:50], "similarity": round(sim, 3)}
    
    return "three", None


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("跨领域验证实验（ToolBench）")
    print("=" * 60)
    
    domains = ["weather", "search", "jokes"]
    all_results = {}
    
    for domain in domains:
        items = load_domain_data(domain)
        queries = [item["query"] for item in items]
        trajectories = [item["trajectory"] for item in items]
        
        print(f"\n{'='*40}")
        print(f"领域: {domain} ({len(queries)} 条查询)")
        print(f"{'='*40}")
        
        # 模拟 TRIAGE 分类
        level_counts = Counter()
        total_triage_tokens = 0
        total_react_tokens = 0
        details = []
        
        # 先假设前 5 条是"已有轨迹"，后 5 条是"新查询"
        known_items = items[:5]
        new_queries = items[5:]
        
        for i, item in enumerate(new_queries):
            query = item["query"]
            traj_len = item["trajectory_length"]
            
            # 基线：完整 ReAct
            react_tokens = 800 + traj_len * 200  # 基础 800 + 每步 200
            total_react_tokens += react_tokens
            
            # TRIAGE 分类
            level, info = classify_query(query, known_items)
            
            if level == "one":
                triage_tokens = 0
            elif level == "two":
                triage_tokens = 250  # 编辑复用
            else:
                triage_tokens = react_tokens  # 全量 ReAct
            
            total_triage_tokens += triage_tokens
            level_counts[level] += 1
            
            details.append({
                "query": query[:60],
                "level": level,
                "react_tokens": react_tokens,
                "triage_tokens": triage_tokens,
                "saved": react_tokens - triage_tokens,
                "trajectory_length": traj_len,
                "info": info,
            })
        
        # 统计
        savings = total_react_tokens - total_triage_tokens
        ratio = savings / total_react_tokens * 100 if total_react_tokens > 0 else 0
        
        print(f"  Level 分布: {dict(level_counts)}")
        print(f"  基线 ReAct: {total_react_tokens} tokens")
        print(f"  TRIAGE:     {total_triage_tokens} tokens")
        print(f"  节省:        {savings} ({ratio:.1f}%)")
        
        for d in details:
            flag = "🟢" if d["level"] == "one" else ("🟡" if d["level"] == "two" else "🔴")
            print(f"  {flag} [{d['level']}] {d['query'][:50]}... "
                  f"ReAct={d['react_tokens']} → TRIAGE={d['triage_tokens']} "
                  f"(节省 {d['saved']})")
        
        all_results[domain] = {
            "level_counts": dict(level_counts),
            "total_react_tokens": total_react_tokens,
            "total_triage_tokens": total_triage_tokens,
            "savings": savings,
            "saving_ratio": round(ratio, 1),
            "details": details,
            "queries": len(new_queries),
        }
    
    # 汇总
    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    
    total_react = sum(r["total_react_tokens"] for r in all_results.values())
    total_triage = sum(r["total_triage_tokens"] for r in all_results.values())
    total_savings = total_react - total_triage
    total_ratio = total_savings / total_react * 100 if total_react > 0 else 0
    
    for domain, r in all_results.items():
        print(f"  {domain}: {r['saving_ratio']}% 节省 ({r['queries']} 条查询)")
    
    print(f"  ──")
    print(f"  总计: {total_react} → {total_triage} = {total_savings} ({total_ratio:.1f}%)")
    
    # 保存结果
    output = {
        "config": {"dataset": "ToolBench", "domains": domains, "experiment": "cross_domain_validation"},
        "results": all_results,
        "summary": {
            "total_react_tokens": total_react,
            "total_triage_tokens": total_triage,
            "total_savings": total_savings,
            "total_saving_ratio": round(total_ratio, 1),
            "domains": {d: r["saving_ratio"] for d, r in all_results.items()},
        },
    }
    
    with open(os.path.join(RESULTS_DIR, "latest.json"), "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 experiments/06-cross-domain-toolbench/results/latest.json")


if __name__ == "__main__":
    main()
