"""
跨领域验证实验 v2

用 sentence-transformers 计算语义相似度，更准确地模拟 TRIAGE 路由分类。
"""

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from collections import Counter
from triage.retrieval.trajectory_encoder import TrajectoryEncoder


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_domain_data(domain):
    path = os.path.join(DATA_DIR, f"{domain}.json")
    with open(path) as f:
        return json.load(f)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("跨领域验证实验 v2（ToolBench + sentence-transformers）")
    print("=" * 60)
    
    # 初始化编码器
    print("\n[1] 初始化语义编码器...")
    encoder = TrajectoryEncoder()
    _ = encoder.encode_query("test")  # 触发模型加载
    print(f"  编码器就绪 ({encoder.get_dimension()}d)")
    
    domains = ["weather", "search", "jokes"]
    all_results = {}
    
    for domain in domains:
        items = load_domain_data(domain)
        queries = [item["query"] for item in items]
        
        print(f"\n{'='*40}")
        print(f"领域: {domain} ({len(queries)} 条查询)")
        print(f"{'='*40}")
        
        # 编码所有查询
        embeddings = encoder.encode_queries(queries)
        
        # 模拟 TRIAGE：前 5 条为"已有轨迹"，后 5 条为"新查询"
        known_queries = queries[:5]
        known_embeddings = embeddings[:5]
        new_queries = queries[5:]
        new_embeddings = embeddings[5:]
        
        level_counts = Counter()
        total_triage_tokens = 0
        total_react_tokens = 0
        details = []
        
        for i, query in enumerate(new_queries):
            emb = new_embeddings[i]
            item = items[5 + i]
            traj_len = item["trajectory_length"]
            
            # 基线：完整 ReAct
            react_tokens = 800 + traj_len * 200
            total_react_tokens += react_tokens
            
            # 计算与已知轨迹的最大相似度
            max_sim = 0.0
            best_match = None
            for j, known_emb in enumerate(known_embeddings):
                sim = encoder.compute_similarity(emb, known_emb)
                if sim > max_sim:
                    max_sim = sim
                    best_match = known_queries[j]
            
            # 分类
            if max_sim > 0.95:
                level = "one"
                triage_tokens = 0
            elif max_sim > 0.6:
                level = "two"
                triage_tokens = 250
            else:
                level = "three"
                triage_tokens = react_tokens
            
            total_triage_tokens += triage_tokens
            level_counts[level] += 1
            
            details.append({
                "query": query[:60],
                "level": level,
                "react_tokens": react_tokens,
                "triage_tokens": triage_tokens,
                "saved": react_tokens - triage_tokens,
                "max_similarity": round(max_sim, 4),
                "best_match": best_match[:50] if best_match else None,
                "trajectory_length": traj_len,
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
            print(f"  {flag} [{d['level']}] sim={d['max_similarity']:.3f} "
                  f"ReAct={d['react_tokens']} → TRIAGE={d['triage_tokens']} "
                  f"(节省 {d['saved']})")
            print(f"        {d['query'][:60]}...")
        
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
    
    # 保存
    output = {
        "config": {"dataset": "ToolBench", "domains": domains, "experiment": "cross_domain_v2", "encoder": "all-MiniLM-L6-v2"},
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
    print(f"\n结果已保存")


if __name__ == "__main__":
    main()
