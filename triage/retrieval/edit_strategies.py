"""
编辑策略

二级链路中，参考成功轨迹的经验，生成新内容。
核心原则：提供成功执行的 SQL 作为参考，让 LLM 在此基础上生成，而非从零生成。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Callable


@dataclass
class EditResult:
    """编辑结果"""
    success: bool
    edited_node: Optional[dict] = None
    token_cost: int = 0
    error: Optional[str] = None


def _clear_token_fields(node: dict) -> dict:
    """清除 Token 统计字段，避免污染新轨迹"""
    n = dict(node)
    n.pop("token_input", None)
    n.pop("token_output", None)
    return n


def reference_generate(
    node: dict,
    new_query: str,
    llm_func: Callable,
    original_query: str = "",
) -> EditResult:
    """
    参考成功经验生成新内容。
    
    核心逻辑：
    1. 提供旧轨迹的成功 SQL 作为参考
    2. 让 LLM 在此基础上生成新 SQL
    3. 不是从零生成，而是"参考 + 适配"
    
    Args:
        node: 旧轨迹节点（包含成功执行的 SQL）
        new_query: 用户的新查询
        llm_func: LLM 调用函数
        original_query: 旧轨迹的原始查询（用于上下文）
    """
    try:
        tool_input = node.get("tool_input", {})
        old_sql = tool_input.get("sql", "")
        action_name = node.get("action_name", "execute_sql")
        
        prompt = (
            f"参考下面这条成功执行过的 SQL，为新查询生成对应的 SQL：\n\n"
            f"旧查询：{original_query}\n"
            f"成功 SQL：\n```sql\n{old_sql}\n```\n\n"
            f"新查询：{new_query}\n\n"
            f"要求：\n"
            f"1. 保持与旧 SQL 相同的结构和风格\n"
            f'2. 只修改与新旧查询差异相关的部分\n'
            f'3. 返回 JSON：{{"tool_name": "{action_name}", "tool_input": {{"sql": "..."}}}}\n'
            f"4. 不要添加旧 SQL 中没有的额外功能"
        )
        
        result = llm_func(prompt)
        
        edited_node = _clear_token_fields(node)
        edited_node["tool_input"] = tool_input.copy()
        edited_node["tool_input"]["sql"] = result.strip()
        edited_node["llm_output"] = result
        edited_node["edited"] = True
        edited_node["edit_type"] = "reference_generate"
        
        return EditResult(success=True, edited_node=edited_node, token_cost=200)
    
    except Exception as e:
        return EditResult(success=False, error=str(e))


def no_change(node: dict) -> EditResult:
    """节点不变，直接复用"""
    edited_node = dict(node)
    edited_node["edited"] = False
    return EditResult(success=True, edited_node=edited_node, token_cost=0)


def execute_edit_strategy(
    node: dict,
    edit_type: str,
    new_query: str = "",
    llm_func: Optional[Callable] = None,
    original_query: str = "",
) -> EditResult:
    """执行编辑策略的入口"""
    
    strategies = {
        "reference_generate": lambda: reference_generate(node, new_query, llm_func, original_query),
        "no_change": lambda: no_change(node),
    }
    
    strategy = strategies.get(edit_type)
    if strategy is None:
        return EditResult(success=False, error=f"Unknown edit type: {edit_type}")
    return strategy()


__all__ = [
    "EditResult",
    "reference_generate",
    "no_change",
    "execute_edit_strategy",
]
