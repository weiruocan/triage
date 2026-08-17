"""ExecutionFlow"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional

@dataclass
class ExecutionFlow:
    skill_name: str; description: str; input_schema: dict; output_schema: dict
    executor: Optional[Callable] = None; estimated_token_cost: int = 0
    def execute(self, params: dict) -> Any:
        if self.executor is None: raise ValueError(f"ExecutionFlow '{self.skill_name}' not registered")
        return self.executor(**params)
    def to_dict(self) -> dict:
        return {"skill_name": self.skill_name, "description": self.description,
                "input_schema": self.input_schema, "output_schema": self.output_schema,
                "estimated_token_cost": self.estimated_token_cost}
    @classmethod
    def from_skill(cls, skill) -> "ExecutionFlow":
        return cls(skill_name=skill.name, description=skill.description,
                   input_schema=skill.input_schema, output_schema=skill.output_schema, executor=skill.execute)

_registry: dict[str, ExecutionFlow] = {}
def register_execution_flow(flow: ExecutionFlow): _registry[flow.skill_name] = flow
def get_execution_flow(name: str) -> Optional[ExecutionFlow]: return _registry.get(name)
def list_execution_flows() -> list[str]: return list(_registry.keys())
def create_sql_execution_flow() -> ExecutionFlow:
    return ExecutionFlow(skill_name="execute_sql", description="Execute SQL query",
        input_schema={"type":"object","properties":{"sql":{"type":"string"}},"required":["sql"]},
        output_schema={"type":"object","properties":{"columns":{"type":"array"},"rows":{"type":"array"},"total":{"type":"integer"}}},
        estimated_token_cost=0)
def create_chart_execution_flow(chart_type: str = "bar") -> ExecutionFlow:
    return ExecutionFlow(skill_name=f"create_{chart_type}_chart", description=f"Create {chart_type} chart",
        input_schema={"type":"object","properties":{"data":{"type":"object"},"title":{"type":"string"},"x_label":{"type":"string"},"y_label":{"type":"string"}},"required":["data","title"]},
        output_schema={"type":"object","properties":{"chart_path":{"type":"string"}}},
        estimated_token_cost=0)
