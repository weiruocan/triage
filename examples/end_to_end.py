"""
TRIAGE 端到端示例（真实 LLM 调用）

展示从冷启动到自动复用的完整流程：
1. 第一次查询 → 冷启动（走完整 ReAct）
2. 第二次相同查询 → 一级复用（0 Token）
3. 第三次相似查询 → 二级编辑复用（~200 Token）

使用前请设置环境变量：
    export TRIAGE_API_KEY="sk-xxx"
    export TRIAGE_BASE_URL="https://api.openai.com/v1"
    export TRIAGE_MODEL="gpt-4"
"""

import os
from triage import triage_agent, OpenAIChatAPI


# 检查 API Key
if not os.getenv("TRIAGE_API_KEY"):
    print("=" * 60)
    print("请先设置环境变量 TRIAGE_API_KEY")
    print("  export TRIAGE_API_KEY=\"sk-xxx\"")
    print("=" * 60)
    exit(1)


# 创建 LLM 客户端
llm = OpenAIChatAPI()


@triage_agent(verbose=True)
def data_agent(query: str) -> str:
    """
    一个真实的数据查询智能体。
    调用 LLM 来分析查询并返回结果。
    """
    prompt = f"""
你是一个数据查询助手。用户查询：{query}

请分析用户的查询意图并返回一个简洁的答案。

要求：
1. 直接回答问题，不要添加额外说明
2. 如果查询某个指标，模拟返回一个合理的数值
3. 保持回答简洁（1-2句话）

示例：
- 查询："sensors online rate" → "传感器在线率为95.2%"
- 查询："cameras online rate" → "摄像头在线率为97.8%"
"""
    result = llm.chat(prompt)
    print(f"  [agent] LLM 调用完成")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("TRIAGE 端到端示例（真实 LLM 调用）")
    print("=" * 60)
    print()

    queries = [
        ("传感器在线率", "sensors online rate"),
        ("传感器在线率（相同查询）", "sensors online rate"),
        ("摄像头在线率（相似查询）", "cameras online rate"),
        ("今日报警次数（全新查询）", "alarm count today"),
    ]

    for desc, q in queries:
        print(f"[{queries.index((desc, q)) + 1}] {desc}：")
        print(f"    查询: {q}")
        result = data_agent(q)
        print(f"    结果: {result}")
        print()

    # 打印统计报告
    print("=" * 60)
    print(data_agent.get_report())
