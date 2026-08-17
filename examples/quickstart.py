"""
TRIAGE 快速入门示例

展示如何用一行装饰器接入 TRIAGE。
"""

# 模拟一个简单的 ReAct 智能体
from triage import triage_agent


@triage_agent(verbose=True)
def my_agent(query: str) -> str:
    """
    模拟一个 ReAct 智能体。
    实际使用中，这里是你已有的 LLM + 工具调用逻辑。
    """
    # 模拟 Token 消耗
    tokens = len(query) * 10
    
    print(f"  [ReAct] 处理查询: {query}")
    print(f"  [ReAct] 消耗 Token: {tokens}")
    
    return f"Result for: {query}"


if __name__ == "__main__":
    print("=" * 50)
    print("TRIAGE 快速入门示例")
    print("=" * 50)
    
    # 第一次查询：冷启动，走完整 ReAct
    print("\n[1] 第一次查询（冷启动，走完整 ReAct）：")
    r1 = my_agent("传感器的在线率是多少？")
    print(f"  结果: {r1}")
    
    # 第二次查询：相同查询，走一级链路（0 Token）
    print("\n[2] 第二次查询（相同，走一级链路，0 Token）：")
    r2 = my_agent("传感器的在线率是多少？")
    print(f"  结果: {r2}")
    
    # 第三次查询：相似查询，走二级链路（约 250 Token）
    print("\n[3] 第三次查询（相似，走二级链路，约 250 Token）：")
    r3 = my_agent("摄像头的在线率是多少？")
    print(f"  结果: {r3}")
    
    # 打印统计
    print("\n" + "=" * 50)
    print("Token 节省统计：")
    print(my_agent.get_report())
