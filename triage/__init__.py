"""
TRIAGE: Three-level Routing and Intelligent Agent Guidance for Efficient Execution

减少 LLM 智能体 Token 消耗的三级路由框架。
核心概念：TaaS（Trajectory-as-a-Skill）——将历史执行轨迹抽象为可复用的技能。
"""

from .model_api import OpenAIChatAPI
from .agent import TriageAgentWrapper
from .decorator import triage_agent
from .cli import main as cli_main
from .retrieval.trajectory_encoder import TrajectoryEncoder
from .retrieval.trajectory_db import TrajectoryDatabase
from .retrieval.experience_editor import ExperienceEditor
from .retrieval.edit_strategies import reference_generate, no_change
from .retrieval.experience_trajectory import ExperienceNode, ExperienceTrajectory
from .router.llm_router import LLMRouter
from .baseline.orchestrator import TriageOrchestrator
from .baseline.step_executor import StepExecutor
from .skills.execution_flow import ExecutionFlow, register_execution_flow, get_execution_flow
from .skills.skill_extractor import SkillExtractor

__version__ = "0.2.0"
__all__ = [
    # 核心装饰器
    "triage_agent",
    # 智能体
    "TriageAgentWrapper",
    "TriageOrchestrator",
    "StepExecutor",
    # LLM
    "OpenAIChatAPI",
    # 检索
    "TrajectoryEncoder",
    "TrajectoryDatabase",
    "ExperienceEditor",
    "ExperienceNode",
    "ExperienceTrajectory",
    "reference_generate",
    "no_change",
    # 路由
    "LLMRouter",
    # 技能
    "ExecutionFlow",
    "register_execution_flow",
    "get_execution_flow",
    "SkillExtractor",
    # CLI
    "cli_main",
]
