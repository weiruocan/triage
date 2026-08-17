"""
ExperienceEditor — 轨迹编辑器

二级链路的核心：参考成功轨迹的经验，编辑适配新查询。
编辑时优先查 Skill 注册表，有匹配的细粒度 Skill 则直接调用（0 Token，确定），
否则回退到 LLM reference_generate（~200 Token，概率）。
"""

from __future__ import annotations
from typing import Optional, Callable

from .edit_strategies import execute_edit_strategy, EditResult
from .experience_trajectory import ExperienceNode, ExperienceTrajectory
from ..skills.execution_flow import get_execution_flow


class ExperienceEditor:
    """
    轨迹编辑器
    
    用法:
        editor = ExperienceEditor(llm_func=my_llm_call)
        edited_trajectory = editor.edit_trajectory(
            trajectory=old_traj,
            edit_nodes=[{"step_index": 0, "edit_type": "reference_generate"}],
            new_query="查前天的摄像头在线率",
        )
    """
    
    def __init__(self, llm_func: Optional[Callable] = None):
        self.llm_func = llm_func
    
    def set_llm_func(self, llm_func: Callable):
        self.llm_func = llm_func
    
    def edit_trajectory(
        self,
        trajectory: ExperienceTrajectory,
        edit_nodes: list[dict],
        new_query: str = "",
    ) -> ExperienceTrajectory:
        """
        编辑轨迹，适配新查询。
        
        编辑优先级：
        1. 节点关联了细粒度 Skill → 直接调用 Skill（0 Token，确定）
        2. 节点可编辑且有 LLM → reference_generate（~200 Token，概率）
        3. 编辑失败 → 保留原节点（兜底）
        """
        new_traj = ExperienceTrajectory(
            query=new_query or trajectory.query,
            trajectory_id=trajectory.trajectory_id,
            category=trajectory.category,
            success=trajectory.success,
        )
        
        edit_map = {e["step_index"]: e for e in edit_nodes}
        total_tokens = 0
        
        for i, node in enumerate(trajectory.nodes):
            nd = node.to_dict()
            
            if i in edit_map:
                edit_info = edit_map[i]
                edit_type = edit_info.get("edit_type", "no_change")
                
                # 优先尝试 Skill 调用（确定性执行）
                edited = self._try_skill_execution(node, nd, new_query)
                if edited:
                    total_tokens += 0  # Skill 调用 0 Token
                    new_traj.add_node(edited)
                    continue
                
                # 回退到 LLM 生成
                result = execute_edit_strategy(
                    node=nd,
                    edit_type=edit_type,
                    new_query=new_query,
                    llm_func=self.llm_func,
                    original_query=trajectory.query,
                )
                
                if result.success and result.edited_node:
                    en = ExperienceNode.from_dict(result.edited_node)
                    en.edit_type = edit_type
                    total_tokens += result.token_cost
                    new_traj.add_node(en)
                else:
                    # 编辑失败，保留原节点
                    new_traj.add_node(node)
            else:
                new_traj.add_node(node)
        
        new_traj.total_tokens = total_tokens
        return new_traj
    
    def _try_skill_execution(
        self,
        node: ExperienceNode,
        node_dict: dict,
        new_query: str,
    ) -> Optional[ExperienceNode]:
        """尝试用 Skill 调用替代 LLM 编辑"""
        skill_name = node.skill_name or node_dict.get("skill_name")
        if not skill_name:
            return None
        
        flow = get_execution_flow(skill_name)
        if not flow:
            return None
        
        try:
            # 从新查询中提取参数（简化版，实际可用 LLM 提取）
            result = flow.execute({"query": new_query})
            edited = ExperienceNode(
                step_index=node.step_index,
                action_name=node.action_name,
                description=f"Skill: {skill_name}",
                node_type="tool",
                tool_name=skill_name,
                tool_input={"query": new_query},
                tool_output=result,
                editable=False,
                token_input=0,
                token_output=0,
            )
            return edited
        except Exception:
            return None
    
    def get_edit_token_savings(self, original, edited) -> dict:
        """计算编辑前后 Token 节省"""
        o = original.total_token_cost if hasattr(original, 'total_token_cost') else 0
        e = edited.total_token_cost if hasattr(edited, 'total_token_cost') else 0
        saved = o - e
        return {
            "original_tokens": o,
            "edit_tokens": e,
            "saved_tokens": saved,
            "saving_ratio": round(saved / o * 100, 2) if o else 0,
        }


__all__ = ["ExperienceEditor"]
