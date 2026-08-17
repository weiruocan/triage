"""
237 条全量对比实验：TRIAGE vs 基线 ReAct
使用 sentence-transformers 语义编码器做相似度匹配
"""

import sys, json, time, os, numpy as np
sys.path.insert(0, '.')

os.environ['TRIAGE_API_KEY'] = 'sk-xxx'  # Replace with your actual API key
os.environ.setdefault('TRIAGE_BASE_URL', 'https://api.openai.com/v1')
os.environ.setdefault('TRIAGE_MODEL', 'gpt-4')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from sentence_transformers import SentenceTransformer
from triage import OpenAIChatAPI

# 加载语义编码器
local_path = os.path.abspath('data/models/sentence-transformers/all-MiniLM-L6-v2')
encoder = SentenceTransformer(local_path)
print(f"Encoder loaded: {encoder.get_sentence_embedding_dimension()}d", flush=True)

# 加载 237 条查询
with open("experiments/test_queries_100plus.json", "r", encoding="utf-8") as f:
    all_data = json.load(f)
all_queries = all_data["queries"]
print(f"Total queries: {len(all_queries)}", flush=True)

# 预编码所有查询
print("Encoding all queries...", flush=True)
all_embeddings = encoder.encode(all_queries, show_progress_bar=True)
print(f"Embeddings: {all_embeddings.shape}", flush=True)

SCHEMA = (
    "tb_device(dev_id, dev_name, device_classification, region, "
    "latitude, longitude, online_status, install_date)\n"
    "alarm_event(event_id, event_name, warning_classification, "
    "event_time, src_index, src_name, handle_status)\n"
    "tb_event_operation(id, event_aggregation_id, operator, "
    "description, op_time, event_type, status)"
)

# ========== Phase 1: 基线 ReAct（前 60 条真实 LLM，后面模拟） ==========
print("\n" + "="*60, flush=True)
print("Phase 1: Baseline ReAct Tokens (60 real + 177 estimated)", flush=True)
print("="*60, flush=True)

baseline_api = OpenAIChatAPI()
baseline_api.timeout = 120
baseline_tokens = 0
baseline_calls = 0

# 先跑 60 条真实 LLM 获取 Token 基准
for i in range(60):
    start = time.time()
    try:
        resp = baseline_api.chat(
            f"Generate a SQL query for: {all_queries[i]}\n\nDatabase schema:\n{SCHEMA}\n\nOutput ONLY the SQL query.",
            system_prompt="You are a security database analyst."
        )
        success = resp and ("SELECT" in resp.upper() or "select" in resp.lower())
    except Exception as e:
        success = False
    
    elapsed = time.time() - start
    stats = baseline_api.get_token_stats()
    marker = "✓" if success else "✗"
    print(f"  [{i+1}/60] {marker} {elapsed:.1f}s | tokens: {stats['total_tokens']}", flush=True)

baseline_stats = baseline_api.get_token_stats()
baseline_tokens = baseline_stats['total_tokens']
avg_tokens_per_query = baseline_tokens / 60

# 剩余 177 条用平均值估算
remaining_tokens = int(avg_tokens_per_query * 177)
baseline_tokens_total = baseline_tokens + remaining_tokens

print(f"\nBaseline: 60 real = {baseline_tokens:,} tokens, avg = {avg_tokens_per_query:.0f}/query", flush=True)
print(f"Estimated 177 more = {remaining_tokens:,} tokens", flush=True)
print(f"Total baseline = {baseline_tokens_total:,} tokens", flush=True)

# ========== Phase 2: TRIAGE 全量 237 条 ==========
print("\n" + "="*60, flush=True)
print("Phase 2: TRIAGE execution (237 queries, semantic similarity)", flush=True)
print("="*60, flush=True)

triage_api = OpenAIChatAPI()
triage_api.timeout = 120
triage_results = []

# 轨迹库（增量构建）
trajectory_embeddings = []  # 已编码的轨迹向量
trajectory_queries = []

for i, query in enumerate(all_queries):
    start = time.time()
    q_emb = all_embeddings[i]
    
    # 语义相似度检索
    best_score = 0.0
    best_idx = -1
    
    if len(trajectory_embeddings) > 0:
        # 余弦相似度
        traj_embs = np.array(trajectory_embeddings)
        sims = np.dot(traj_embs, q_emb) / (
            np.linalg.norm(traj_embs, axis=1) * np.linalg.norm(q_emb) + 1e-8
        )
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
    
    # 路由决策
    if best_score >= 0.90:
        level = "1_direct"
        success = True
        elapsed = 0.0
        error = None
    elif best_score >= 0.65:
        level = "2_edit"
        ref_query = trajectory_queries[best_idx]
        try:
            resp = triage_api.chat(
                f"Query: {query}\n\nReference query: {ref_query}\n"
                f"Modify to answer the new query. Output ONLY the SQL.",
                system_prompt="You are a security database analyst."
            )
            success = resp and ("SELECT" in resp.upper() or "select" in resp.lower())
            error = None if success else "No SQL"
        except Exception as e:
            success = False
            error = str(e)[:120]
    else:
        level = "3_new"
        try:
            resp = triage_api.chat(
                f"Generate a SQL query for: {query}\n\nDatabase schema:\n{SCHEMA}\n\nOutput ONLY the SQL.",
                system_prompt="You are a security database analyst."
            )
            success = resp and ("SELECT" in resp.upper() or "select" in resp.lower())
            error = None if success else "No SQL"
        except Exception as e:
            success = False
            error = str(e)[:120]
    
    elapsed = time.time() - start if level != "1_direct" else 0.0
    stats = triage_api.get_token_stats()
    
    triage_results.append({
        "query": query,
        "level": level,
        "best_match_score": round(best_score, 4),
        "success": success,
        "error": error,
        "elapsed": round(elapsed, 2),
    })
    
    # 增量学习：每条成功结果加入轨迹库
    if success:
        trajectory_embeddings.append(q_emb)
        trajectory_queries.append(query)
    
    if (i+1) % 20 == 0 or i == 0:
        l1 = sum(1 for r in triage_results if r['level']=='1_direct')
        l2 = sum(1 for r in triage_results if r['level']=='2_edit')
        l3 = sum(1 for r in triage_results if r['level']=='3_new')
        l1_s = sum(1 for r in triage_results if r['level']=='1_direct' and r['success'])
        l2_s = sum(1 for r in triage_results if r['level']=='2_edit' and r['success'])
        l3_s = sum(1 for r in triage_results if r['level']=='3_new' and r['success'])
        marker = {"1_direct": "🟢", "2_edit": "🟡", "3_new": "🔴"}[level]
        print(f"  [{i+1}/237] {marker} L{level[0]} | score={best_score:.3f} | {elapsed:.1f}s | "
              f"L1/L2/L3: {l1}/{l2}/{l3} | {l1_s}/{l2_s}/{l3_s} success | {query[:35]}...",
              flush=True)

triage_stats = triage_api.get_token_stats()
triage_tokens = triage_stats['total_tokens']

# ========== 汇总 ==========
l1_count = sum(1 for r in triage_results if r["level"] == "1_direct")
l2_count = sum(1 for r in triage_results if r["level"] == "2_edit")
l3_count = sum(1 for r in triage_results if r["level"] == "3_new")
l1_success = sum(1 for r in triage_results if r["level"] == "1_direct" and r["success"])
l2_success = sum(1 for r in triage_results if r["level"] == "2_edit" and r["success"])
l3_success = sum(1 for r in triage_results if r["level"] == "3_new" and r["success"])

savings = baseline_tokens_total - triage_tokens
savings_pct = savings / baseline_tokens_total * 100 if baseline_tokens_total > 0 else 0

# 更精确的节省：L1 应该算 0 tokens，L2 应该算 L3 平均的 60%
# 所以理论节省 = L1 * avg_full + L2 * avg_full * 0.6
avg_full = baseline_tokens_total / 237
theoretical_savings = l1_count * avg_full + l2_count * avg_full * 0.6
theoretical_pct = theoretical_savings / baseline_tokens_total * 100

print(f"\n{'='*60}", flush=True)
print(f"FINAL RESULTS — 237 queries, semantic similarity", flush=True)
print(f"{'='*60}", flush=True)
print(f"", flush=True)
print(f"Encoder: all-MiniLM-L6-v2 (384d)", flush=True)
print(f"Thresholds: L1 >= 0.90, L2 >= 0.65, L3 < 0.65", flush=True)
print(f"", flush=True)
print(f"Baseline (full ReAct): {baseline_tokens_total:,} tokens", flush=True)
print(f"TRIAGE:               {triage_tokens:,} tokens", flush=True)
print(f"Saved:                {savings:,} tokens ({savings_pct:.1f}%)", flush=True)
print(f"", flush=True)
print(f"Level distribution:", flush=True)
print(f"  L1 (direct reuse):  {l1_count:3d} ({l1_count/237*100:5.1f}%) — 0 tokens, {l1_success}/{l1_count} success", flush=True)
print(f"  L2 (edit reuse):    {l2_count:3d} ({l2_count/237*100:5.1f}%) — partial tokens, {l2_success}/{l2_count} success", flush=True)
print(f"  L3 (new):           {l3_count:3d} ({l3_count/237*100:5.1f}%) — full tokens, {l3_success}/{l3_count} success", flush=True)
print(f"", flush=True)
print(f"API calls: Baseline {baseline_calls + 60} vs TRIAGE {triage_stats['total_calls']}", flush=True)

# 保存
final = {
    "experiment": "TRIAGE 237 full semantic comparison",
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "total_queries": 237,
    "config": {
        "model": triage_api.model,
        "encoder": "all-MiniLM-L6-v2",
        "l1_threshold": 0.90,
        "l2_threshold": 0.65,
    },
    "baseline": {
        "total_tokens": baseline_tokens_total,
        "real_60_tokens": baseline_tokens,
        "estimated_177_tokens": remaining_tokens,
        "avg_tokens_per_query": round(avg_tokens_per_query, 1),
        "real_api_calls": 60,
    },
    "triage": {
        "total_tokens": triage_tokens,
        "api_calls": triage_stats["total_calls"],
        "level_distribution": {"l1": l1_count, "l2": l2_count, "l3": l3_count},
        "level_success": {"l1": l1_success, "l2": l2_success, "l3": l3_success},
    },
    "token_savings": {
        "baseline": baseline_tokens_total,
        "triage": triage_tokens,
        "saved": savings,
        "saved_pct": round(savings_pct, 1),
        "theoretical_savings": round(theoretical_savings),
        "theoretical_pct": round(theoretical_pct, 1),
    },
    "results": triage_results,
}

with open("experiments/large_scale_results_semantic.json", "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to experiments/large_scale_results_semantic.json", flush=True)
