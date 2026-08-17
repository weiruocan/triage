"""
真实 ReAct 基线实验

用真实 deepseek-v4-flash 跑完整 ReAct 循环，获取准确的 Token 消耗数据。

对比目标：
- 基线：真实 LLM 完整 ReAct（本实验）
- TRIAGE：已有实验数据（03-real-llm-comparison）
"""

import sys
import json
import os
import time
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from triage import OpenAIChatAPI
from triage.skills.register_skills import register_all_skills


# 13 个测试查询（与 TRIAGE 实验一致）
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

# 系统提示词
SYSTEM_PROMPT = """You are a security monitor database assistant. You have access to the following tools:

1. validate_sql(sql: str) - Validate SQL query syntax
2. execute_sql(sql: str, max_rows: int) - Execute SQL query on the security monitor database
3. create_bar_chart(data: dict, title: str) - Create a bar chart

Database schema:
- tb_device: dev_id, dev_name, device_classification, region, online_status, install_date
- alarm_event: event_id, event_name, event_time, src_index, handle_status
- tb_event_operation: id, event_aggregation_id, operator, description, op_time, event_type, status

You must follow the ReAct loop:
1. Thought: reason about what to do
2. Action: call a tool with arguments
3. Observation: see the tool result
4. Repeat until done
5. Final Answer: provide the answer to the user

Return your response in JSON format:
{"thought": "...", "action": "tool_name", "action_input": {"param": "value"}}
Or for final answer:
{"thought": "...", "final_answer": "..."}
"""


def run_react_loop(query: str, api: OpenAIChatAPI, max_steps: int = 5) -> dict:
    """用真实 LLM 跑完整 ReAct 循环"""
    import sqlite3
    start_time = time.time()
    token_start = api.get_token_stats()
    history = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]
    steps = []
    success = False
    final_answer = ""

    for step in range(max_steps):
        step_start = api.get_token_stats()
        try:
            import requests
            payload = {"model": api.model, "messages": history, "temperature": 0.3, "max_tokens": 1024}
            response = requests.post(
                f"{api.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api.api_key}", "Content-Type": "application/json"},
                json=payload, timeout=api.timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            api.total_input_tokens += usage.get("prompt_tokens", 0)
            api.total_output_tokens += usage.get("completion_tokens", 0)
            api.total_calls += 1
        except Exception as e:
            return {"query": query, "success": False, "error": f"LLM failed at step {step}: {str(e)[:100]}",
                    "steps": steps, "total_tokens": api.get_token_stats()["total_tokens"] - token_start["total_tokens"],
                    "time_cost": round(time.time() - start_time, 2)}
        step_tokens = api.get_token_stats()["total_tokens"] - step_start["total_tokens"]
        parsed = {"thought": content, "action": None, "final_answer": None}
        try:
            parsed_json = json.loads(content)
            parsed.update(parsed_json)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    parsed_json = json.loads(match.group())
                    parsed.update(parsed_json)
                except: pass
        step_record = {"step": step, "thought": parsed.get("thought", content)[:200],
                       "action": parsed.get("action"), "action_input": parsed.get("action_input"),
                       "tokens": step_tokens}
        if parsed.get("final_answer"):
            final_answer = parsed["final_answer"]
            success = True
            steps.append(step_record)
            break
        action = parsed.get("action")
        action_input = parsed.get("action_input", {})
        if action in ("validate_sql", "execute_sql"):
            sql = action_input.get("sql", "") if isinstance(action_input, dict) else str(action_input)
            try:
                db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/security_monitor.db"))
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(sql)
                rows = cursor.fetchmany(20)
                observation = json.dumps([dict(row) for row in rows][:5], ensure_ascii=False) if rows else "No results"
                conn.close()
            except Exception as e:
                observation = f"SQL error: {str(e)[:100]}"
            step_record["observation"] = observation
            history.append({"role": "assistant", "content": content})
            history.append({"role": "user", "content": f"Observation: {observation}"})
        elif action == "create_bar_chart":
            step_record["observation"] = "Chart created."
            history.append({"role": "assistant", "content": content})
            history.append({"role": "user", "content": "Observation: Chart created."})
        else:
            history.append({"role": "assistant", "content": content})
        steps.append(step_record)
    total_tokens = api.get_token_stats()["total_tokens"] - token_start["total_tokens"]
    return {"query": query, "success": success, "final_answer": final_answer, "steps": steps,
            "total_tokens": total_tokens, "time_cost": round(time.time() - start_time, 2), "error": ""}


def main():
    import sqlite3
    print("=" * 60)
    print("真实 ReAct 基线实验")
    print("=" * 60)
    print("\n[1] 注册技能...")
    register_all_skills()
    print("\n[2] 初始化 LLM API...")
    api = OpenAIChatAPI()
    api.reset_stats()
    print(f"\n[3] 运行 {len(TEST_QUERIES)} 条查询的完整 ReAct...")
    results = []
    for i, query in enumerate(TEST_QUERIES):
        print(f"  [{i+1}/{len(TEST_QUERIES)}] {query[:50]}...", end=" ", flush=True)
        try:
            result = run_react_loop(query, api)
            results.append(result)
            flag = "✅" if result["success"] else "❌"
            print(f"{flag} {result['total_tokens']} tokens, {result['time_cost']}s, {len(result['steps'])} steps")
            if result["error"]:
                print(f"       error: {result['error'][:80]}")
        except Exception as e:
            print(f"❌ failed: {str(e)[:60]}")
            results.append({"query": query, "success": False, "error": str(e), "total_tokens": 0, "time_cost": 0, "steps": []})
    total_tokens = sum(r["total_tokens"] for r in results)
    success_count = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}\n实验结果\n{'='*60}")
    print(f"  查询总数: {len(results)}")
    print(f"  成功: {success_count}/{len(results)}")
    print(f"  总 Token: {total_tokens}")
    print(f"  平均 Token/条: {total_tokens / len(results):.0f}")
    print(f"  平均耗时: {sum(r['time_cost'] for r in results) / len(results):.1f}s")
    print(f"\n{'='*60}\n与 TRIAGE 对比\n{'='*60}")
    triage_path = os.path.join(os.path.dirname(__file__), "../03-real-llm-comparison/results/latest.json")
    if os.path.exists(triage_path):
        with open(triage_path) as f:
            triage_data = json.load(f)
        triage_total = triage_data["triage"]["total_tokens"]
        saving = total_tokens - triage_total
        ratio = saving / total_tokens * 100 if total_tokens > 0 else 0
        print(f"  基线 ReAct（真实）: {total_tokens} tokens")
        print(f"  TRIAGE（真实）:     {triage_total} tokens")
        print(f"  节省:               {saving} ({ratio:.1f}%)")
    output = {"config": {"queries": len(TEST_QUERIES), "llm_model": api.model, "experiment": "real_react_baseline"},
              "results": results, "summary": {"total_tokens": total_tokens, "success_count": success_count,
              "total_queries": len(results), "avg_tokens_per_query": round(total_tokens / len(results), 1) if results else 0,
              "avg_time_per_query": round(sum(r["time_cost"] for r in results) / len(results), 1) if results else 0},
              "token_stats": api.get_token_stats()}
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "latest.json"), "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 experiments/04-real-react-baseline/results/latest.json")

if __name__ == "__main__":
    main()
