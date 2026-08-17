"""ExperienceNode & ExperienceTrajectory

轨迹数据模型。一条轨迹 = 一个查询的完整执行记录 = 按序排列的节点列表。
"""

from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, Any


@dataclass
class ExperienceNode:
    """轨迹节点 = 执行中的一步"""
    step_index: int
    action_name: str
    description: str
    node_type: str  # "tool" | "llm_generate" | "llm_edit" | "execution_flow"
    
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[Any] = None
    execution_flow: Optional[dict] = None
    llm_prompt: Optional[str] = None
    llm_output: Optional[str] = None
    editable: bool = False
    edit_type: Optional[str] = None  # "reference_generate" | "no_change" | None
    token_input: int = 0
    token_output: int = 0
    skill_name: Optional[str] = None  # 关联的细粒度 Skill 名
    cached_result: Optional[Any] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @property
    def token_cost(self):
        return self.token_input + self.token_output
    
    @property
    def is_zero_token(self):
        return self.node_type in ("tool", "execution_flow") or (
            self.node_type == "llm_edit" and self.edit_type == "no_change"
        )


@dataclass
class ExperienceTrajectory:
    """一条完整的执行轨迹"""
    query: str
    nodes: list = field(default_factory=list)
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    category: str = ""
    total_tokens: int = 0
    success: bool = True
    skill_name: Optional[str] = None  # 关联的细粒度 Skill 名
    cached_result: Optional[Any] = None
    created_at: Optional[str] = None
    
    def add_node(self, node):
        self.nodes.append(node)
        self.total_tokens += node.token_cost
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "trajectory_id": self.trajectory_id,
            "category": self.category,
            "total_tokens": self.total_tokens,
            "success": self.success,
            "cached_result": self.cached_result,
            "created_at": self.created_at,
            "nodes": [n.to_dict() for n in self.nodes],
        }
    
    @classmethod
    def from_dict(cls, d):
        nodes = [ExperienceNode.from_dict(n) for n in d.get("nodes", [])]
        return cls(
            query=d["query"],
            nodes=nodes,
            trajectory_id=d.get("trajectory_id", str(uuid.uuid4())[:8]),
            category=d.get("category", ""),
            total_tokens=d.get("total_tokens", sum(n.token_cost for n in nodes)),
            success=d.get("success", True),
            cached_result=d.get("cached_result"),
            created_at=d.get("created_at"),
        )
    
    @property
    def total_token_cost(self):
        return sum(n.token_cost for n in self.nodes)
    
    @property
    def zero_token_ratio(self):
        if not self.nodes:
            return 0.0
        return sum(1 for n in self.nodes if n.is_zero_token) / len(self.nodes)
    
    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def migrate_from_old_trajectory(old_data: dict) -> ExperienceTrajectory:
    """从旧格式迁移到新格式"""
    query = old_data.get("query", "").replace("\nBegin!", "").strip()
    old_steps = old_data.get("steps", [])
    nodes = []
    
    for i, s in enumerate(old_steps):
        action_name = s.get("action_name", "unknown")
        action_input = s.get("action_input", {})
        observation = s.get("observation", "")
        token_input = s.get("token_input", 0)
        token_output = s.get("token_output", 0)
        thought = s.get("thought", "")
        
        if action_name in (
            "execute_sql", "create_bar_chart", "create_pie_chart",
            "create_line_chart", "save_to_file", "write_report"
        ):
            node_type, editable = "tool", False
        elif action_name == "validate_sql":
            node_type, editable = "llm_edit", True
        elif token_input > 0 and token_output > 0:
            node_type, editable = "llm_generate", True
        else:
            node_type, editable = "tool", False
        
        node = ExperienceNode(
            step_index=i,
            action_name=action_name,
            description=thought[:100] if thought else action_name,
            node_type=node_type,
            tool_name=action_name if node_type == "tool" else None,
            tool_input=action_input if node_type == "tool" else None,
            tool_output=observation,
            token_input=token_input,
            token_output=token_output,
            editable=editable,
            edit_type="reference_generate" if editable else None,
        )
        nodes.append(node)
    
    return ExperienceTrajectory(
        query=query,
        nodes=nodes,
        total_tokens=sum(n.token_cost for n in nodes),
        success=True,
    )


__all__ = ["ExperienceNode", "ExperienceTrajectory", "migrate_from_old_trajectory"]
