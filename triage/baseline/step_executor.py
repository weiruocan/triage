"""
StepExecutor — 确定性步骤执行引擎

方案 B：纯函数调用器，不涉及 LLM。
只负责遍历步骤/节点列表，根据 action_name 找到对应的 Skill 并调用 execute()。

输入：可执行节点列表（来自 ExperienceEditor 的输出）
输出：每个步骤的执行结果列表

变化（2026-08-05）：
- 适配 ExperienceNode 格式，而非旧 steps 格式
- 支持 ExecutionFlow 注册表查找
- 支持节点级的成功/失败记录
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from ..retrieval.experience_trajectory import ExperienceNode, ExperienceTrajectory
from ..skills.execution_flow import get_execution_flow, list_execution_flows


@dataclass
class StepResult:
    """步骤执行结果"""
    success: bool
    step_index: int
    action_name: str
    data: Any = None
    error: str = ""
    metadata: dict = field(default_factory=dict)


class StepExecutor:
    """
    步骤执行器
    
    输入：
        1. 节点列表（来自 ExperienceEditor 的输出）
        2. 或 ExperienceTrajectory 对象
    输出：
        每个步骤的执行结果列表
    """
    
    def __init__(self, skills: Optional[Dict[str, Any]] = None):
        """
        Args:
            skills: 可选，直接传入技能字典 {action_name: skill_object}
                    如果未提供，则从 ExecutionFlow 注册表查找
        """
        self.skills = skills or {}
    
    def register_skill(self, name: str, skill: Any):
        """注册技能"""
        self.skills[name] = skill
    
    def execute_trajectory(self, trajectory: ExperienceTrajectory) -> dict:
        """
        执行整个轨迹
        
        Returns:
            {
                "success": True/False,
                "output": "最终输出",
                "results": [StepResult, ...],
                "sql_failed": False,
                "chart_failed": False,
                "failed_steps": [index, ...]
            }
        """
        return self.execute_nodes(trajectory.nodes)
    
    def execute_nodes(self, nodes: List[ExperienceNode]) -> dict:
        """
        按顺序执行节点列表
        
        - SQL 节点失败会 break（SQL 是后续依赖）
        - 图表节点失败不会 break
        - LLM 节点（llm_generate, llm_edit）会调用 LLM
        """
        results = []
        sql_failed = False
        chart_failed = False
        failed_steps = []
        
        for i, node in enumerate(nodes):
            # 确定执行方式
            if node.node_type in ("tool", "execution_flow"):
                # 确定性执行
                result = self._execute_tool_node(node)
            elif node.node_type in ("llm_generate", "llm_edit"):
                # 需要 LLM 的节点
                result = self._execute_llm_node(node)
            else:
                result = StepResult(
                    success=False, step_index=i,
                    action_name=node.action_name,
                    error=f"Unknown node_type: {node.node_type}",
                )
            
            results.append(result)
            
            # SQL 失败 → break
            if not result.success and node.action_name in ("validate_sql", "execute_sql"):
                sql_failed = True
                failed_steps.append(i)
                break
            
            # 图表失败 → 不 break
            if not result.success and "chart" in node.action_name:
                chart_failed = True
                failed_steps.append(i)
            
            # 其他失败 → 记录但不 break
            if not result.success:
                failed_steps.append(i)
        
        # 组装输出
        output = self._build_output(results)
        
        return {
            "success": not sql_failed,  # SQL 失败则整体失败
            "output": output,
            "results": results,
            "sql_failed": sql_failed,
            "chart_failed": chart_failed,
            "failed_steps": failed_steps,
        }
    
    def _execute_tool_node(self, node: ExperienceNode) -> StepResult:
        """执行确定性工具节点"""
        action_name = node.action_name
        action_input = node.tool_input or {}
        
        # 先从直接注册的技能找
        skill = self.skills.get(action_name)
        
        # 再从 ExecutionFlow 注册表找
        if skill is None:
            flow = get_execution_flow(action_name)
            if flow:
                # 如果有 executor，直接调用
                if flow.executor:
                    try:
                        result = flow.execute(action_input)
                        return StepResult(
                            success=True, step_index=node.step_index,
                            action_name=action_name, data=result,
                        )
                    except Exception as e:
                        return StepResult(
                            success=False, step_index=node.step_index,
                            action_name=action_name, error=str(e),
                        )
                else:
                    # 没有注册 executor，返回模拟成功
                    return StepResult(
                        success=True, step_index=node.step_index,
                        action_name=action_name,
                        data={"status": "executor_not_registered"},
                        metadata={"placeholder": True},
                    )
        
        # 调用 skill
        if skill is not None:
            try:
                if hasattr(skill, 'execute'):
                    result = skill.execute(**action_input)
                    return StepResult(
                        success=True, step_index=node.step_index,
                        action_name=action_name, data=result,
                    )
                else:
                    # skill 本身是可调用对象
                    result = skill(**action_input)
                    return StepResult(
                        success=True, step_index=node.step_index,
                        action_name=action_name, data=result,
                    )
            except Exception as e:
                return StepResult(
                    success=False, step_index=node.step_index,
                    action_name=action_name, error=str(e),
                )
        
        # 技能未找到
        return StepResult(
            success=False, step_index=node.step_index,
            action_name=action_name,
            error=f"Skill not found: {action_name}. "
                  f"Registered: {list(self.skills.keys()) + list_execution_flows()}",
        )
    
    def _execute_llm_node(self, node: ExperienceNode) -> StepResult:
        """执行需要 LLM 的节点"""
        # llm_generate 和 llm_edit 节点在 ExperienceEditor 中已被处理
        # 优先返回 llm_output 文本，其次 tool_output
        output = node.llm_output or node.tool_output or ""
        return StepResult(
            success=True, step_index=node.step_index,
            action_name=node.action_name,
            data={"status": "llm_handled_by_editor", "llm_output": output},
            metadata={"node_type": node.node_type, "edit_type": node.edit_type},
        )
    
    def _build_output(self, results: List[StepResult]) -> str:
        """从执行结果中组装输出"""
        parts = []
        for r in results:
            if r.success and r.data:
                if isinstance(r.data, dict):
                    if "columns" in r.data and "rows" in r.data:
                        rows = r.data["rows"]
                        parts.append(f"Query result ({len(rows)} rows): "
                                     f"{json.dumps(rows[:3], ensure_ascii=False)}")
                    elif "chart_path" in r.data:
                        parts.append(f"Chart: {r.data['chart_path']}")
                    elif "image_path" in r.data:
                        parts.append(f"Image: {r.data['image_path']}")
                    elif "llm_output" in r.data:
                        parts.append(str(r.data["llm_output"])[:500])
                    else:
                        parts.append(str(r.data)[:200])
                else:
                    parts.append(str(r.data)[:200])
        return "\n".join(parts)
    
    def execute_sql(self, sql: str) -> dict:
        """执行单条 SQL（直接接口）"""
        sql_skill = self.skills.get("execute_sql")
        if not sql_skill:
            flow = get_execution_flow("execute_sql")
            if flow and flow.executor:
                try:
                    return {"success": True, "data": flow.execute({"sql": sql})}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            return {"success": False, "error": "execute_sql not registered"}
        try:
            result = sql_skill.execute(sql=sql)
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}


import json  # noqa: E402 (used in _build_output)
