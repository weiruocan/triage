"""
@triage_agent 装饰器 — 一行代码接入 TRIAGE 三链路编排

使用方式：

    from triage.agent import triage_agent

    @triage_agent()
    def my_agent(query: str) -> str:
        # 用户已有的智能体逻辑（作为三级链路回退）
        return f"Processed: {query}"

    # 现在 my_agent 自动获得了 TRIAGE 能力
    result = my_agent("What is the online rate of sensors?")
    print(result.level, result.total_tokens, result.token_saved)

行为：
    - 一级链路：完全相同查询 → 直接复用缓存（0 Token）
    - 二级链路：相似查询 → 编辑轨迹后复用（仅编辑节点消耗 Token）
    - 三级链路：全新查询 → 走用户原始函数，自动入库

设计原则：
    - 最小侵入：不改动用户原有函数签名
    - 渐进增强：先用默认设置跑，再逐步自定义
    - 自动冷启动：轨迹库为空时退化到纯用户函数
"""

from __future__ import annotations
import json
import time
import inspect
import functools
from typing import Optional, Callable, Any, Dict

from . import OpenAIChatAPI
from .router.llm_router import LLMRouter, RouterDecision
from .retrieval.trajectory_db import TrajectoryDatabase
from .retrieval.trajectory_encoder import TrajectoryEncoder
from .retrieval.experience_trajectory import ExperienceNode, ExperienceTrajectory
from .retrieval.experience_editor import ExperienceEditor
from .baseline.step_executor import StepExecutor
from .baseline.orchestrator import TriageOrchestrator, LinkResult
from .skills.register_skills import register_all_skills


class TriageAgentWrapper:
    """
    TRIAGE 智能体包装器

    在用户函数外层包装三链路编排逻辑。
    """

    def __init__(
        self,
        func: Callable,
        model_api: Optional[OpenAIChatAPI] = None,
        trajectory_db: Optional[TrajectoryDatabase] = None,
        encoder: Optional[TrajectoryEncoder] = None,
        step_executor: Optional[StepExecutor] = None,
        category: str = "default",
        auto_register_skills: bool = True,
        verbose: bool = False,
    ):
        self.func = func
        self.category = category
        self.verbose = verbose
        self.stats = {
            "level1_reuse": 0,
            "level2_edit": 0,
            "level3_new": 0,
            "total_tokens_saved": 0,
            "total_token_cost": 0,
            "total_calls": 0,
        }

        # 自动注册技能
        if auto_register_skills:
            register_all_skills()

        # 初始化
        self.model_api = model_api or OpenAIChatAPI()
        self.encoder = encoder or TrajectoryEncoder()
        self.db = trajectory_db or TrajectoryDatabase(encoder=self.encoder)
        self.router = LLMRouter(self.model_api)
        self.executor = step_executor or StepExecutor()

        # 编排器（不传 react_engine，冷启动自动退化）
        self.orchestrator = TriageOrchestrator(
            model_api=self.model_api,
            retriever=self.db,
            step_executor=self.executor,
            trajectory_db=self.db,
            encoder=self.encoder,
        )

    def _log(self, msg: str, level: str = "info"):
        if level == "error":
            print(f"  [triage] ❌ {msg}")
        elif self.verbose or level == "warning":
            print(f"  [triage] {level.upper()} {msg}")

    def run(self, query: str, **kwargs) -> LinkResult:
        """
        运行 TRIAGE 三链路编排

        Args:
            query: 用户查询
            **kwargs: 传递给用户函数的额外参数

        Returns:
            LinkResult: 包含链路级别、Token 消耗、输出等
        """
        self.stats["total_calls"] += 1

        # 1. 检索相似轨迹
        retrieved = self.db.search(query, top_k=5)
        self._log(f"检索到 {len(retrieved)} 条相似轨迹")

        # 2. 路由决策
        decision = self.router.route(query, retrieved)
        self._log(f"路由决策: {decision.level} (confidence={decision.confidence:.2f})")

        # 3. 分发
        if decision.level == "one":
            result = self._level1(query, decision, retrieved)
        elif decision.level == "two":
            result = self._level2(query, decision, retrieved)
        else:
            result = self._level3(query, **kwargs)

        # 更新统计
        self.stats["total_tokens_saved"] += result.token_saved
        self.stats["total_token_cost"] += result.total_tokens

        if result.level == "one":
            self.stats["level1_reuse"] += 1
        elif result.level == "two":
            self.stats["level2_edit"] += 1
        else:
            self.stats["level3_new"] += 1

        return result

    def _level1(self, query: str, decision: RouterDecision,
                retrieved: list) -> LinkResult:
        """一级链路：直接复用（不依赖任何特定技能，对所有节点类型通用）"""
        result = LinkResult(query=query, level="one",
                            matched_trajectory_id=decision.matched_trajectory_id)

        try:
            for t in retrieved:
                traj = t.get("trajectory", {})
                if isinstance(traj, str):
                    traj = json.loads(traj)
                
                # 1. 优先返回缓存结果
                cached = traj.get("cached_result") or traj.get("final_answer", "")
                if cached and len(str(cached)) > 50:
                    result.output = str(cached)
                    result.total_tokens = 0
                    result.token_saved = traj.get("total_tokens", 0)
                    return result
                
                # 2. 返回最后一个节点的输出（无论什么技能）
                nodes = traj.get("nodes", [])
                if nodes:
                    last = nodes[-1]
                    # 优先取 tool_output，其次 llm_output，最后 observation
                    output = (last.get("tool_output") or 
                             last.get("llm_output") or 
                             last.get("observation") or 
                             last.get("description", ""))
                    if output and len(str(output)) > 10:
                        result.output = str(output)[:500]
                        result.total_tokens = 0
                        result.token_saved = traj.get("total_tokens", 0)
                        return result
                
                # 3. 返回轨迹摘要
                result.output = f"[Reuse] {traj.get('query', '')[:100]}"
                result.total_tokens = 0
                result.token_saved = traj.get("total_tokens", 0)
                return result

            result.error = "No cache or SQL found"
            result.success = False

        except Exception as e:
            result.success = False
            result.error = str(e)
            self._log(f"一级链路失败: {e}")

        return result

    def _level2(self, query: str, decision: RouterDecision,
                retrieved: list) -> LinkResult:
        """二级链路：编辑复用"""
        result = LinkResult(query=query, level="two",
                            matched_trajectory_id=decision.matched_trajectory_id,
                            edit_nodes=decision.edit_nodes)

        try:
            # 找到匹配轨迹
            matched = None
            for t in retrieved:
                traj = t.get("trajectory", {})
                if isinstance(traj, str):
                    traj = json.loads(traj)
                tid = traj.get("trajectory_id", t.get("id", ""))
                if str(tid) == str(decision.matched_trajectory_id):
                    matched = traj
                    break

            if not matched:
                raise ValueError(f"未找到匹配轨迹: {decision.matched_trajectory_id}")

            # 编辑轨迹
            editor = ExperienceEditor(llm_func=self.model_api.chat)
            old_traj = ExperienceTrajectory.from_dict(matched)
            new_traj = editor.edit_trajectory(
                trajectory=old_traj,
                edit_nodes=decision.edit_nodes,
                new_query=query,
            )

            # 执行编辑后的轨迹
            exec_result = self.executor.execute_trajectory(new_traj)
            if exec_result.get("success"):
                result.output = exec_result.get("output", "")
                result.total_tokens = new_traj.total_tokens
                result.token_saved = old_traj.total_tokens - new_traj.total_tokens
                return result

            raise ValueError("StepExecutor 执行失败")

        except Exception as e:
            result.success = False
            result.error = str(e)
            self._log(f"二级链路失败: {e}")

        return result

    def _level3(self, query: str, **kwargs) -> LinkResult:
        """三级链路：走用户原始函数"""
        result = LinkResult(query=query, level="three")

        try:
            # 调用用户原始函数
            output = self.func(query, **kwargs)
            result.output = str(output) if output else ""
            result.success = True
            result.total_tokens = self.model_api.get_token_stats()["total_tokens"]

            # 自动入库
            self._store_trajectory(query, result.output, result.total_tokens)

        except Exception as e:
            result.success = False
            result.error = str(e)

        return result

    def _store_trajectory(self, query: str, output: str, tokens: int):
        """将三级链路结果入库（通用版本，不依赖具体技能名）"""
        try:
            node = ExperienceNode(
                step_index=0,
                action_name="llm_response",
                description=f"Agent response for: {query}",
                node_type="llm_generate",
                llm_output=output[:500],
                llm_prompt=f"Query: {query}",
                token_input=tokens, token_output=0,
                editable=True,
                edit_type="reference_generate",
            )
            traj = ExperienceTrajectory(
                query=query, nodes=[node],
                category=self.category, success=True,
                total_tokens=tokens,
                cached_result=output[:1000],
            )
            embedding = self.encoder.encode_query(query)
            self.db.add_trajectory(
                query=query, query_embedding=embedding,
                trajectory_json=traj.to_json(),
                total_tokens=traj.total_tokens,
                zero_token_ratio=traj.zero_token_ratio,
                success=1, category=self.category,
            )
            self._log(f"轨迹入库成功: {query[:50]}...")
            # 自动 Skill 提炼分析（在原有 try 块内）
            try:
                from .skills.skill_extractor import analyze_and_extract, flush_skills
                # 获取刚插入的 ID
                traj_id = self.db.count()
                analyze_and_extract(traj.to_json(), traj_id)
                if traj_id > 0 and traj_id % 10 == 0:
                    registered = flush_skills()
                    if registered > 0:
                        self._log(f"自动提炼 {registered} 个 Skill")
            except Exception:
                pass
        except Exception as e:
            self._log(f"轨迹入库失败: {e}", level="warning")

    def get_stats(self) -> dict:
        return dict(self.stats)

    def get_report(self) -> str:
        """生成可读的 Token 节省报告"""
        s = self.stats
        total = s["total_tokens_saved"] + s["total_token_cost"]
        ratio = s["total_tokens_saved"] / total * 100 if total > 0 else 0
        lines = [
            "=" * 50,
            "TRIAGE Token 节省报告",
            "=" * 50,
            f"  总调用: {s['total_calls']} 次",
            f"  一级复用: {s['level1_reuse']} 次 (0 Token)",
            f"  二级编辑: {s['level2_edit']} 次 (~250 Token/条)",
            f"  三级 ReAct: {s['level3_new']} 次 (全量 Token)",
            f"  Token 节省: {s['total_tokens_saved']} ({ratio:.1f}%)",
            f"  Token 消耗: {s['total_token_cost']}",
            "=" * 50,
        ]
        return "\n".join(lines)


def triage_agent(
    model_api: Optional[OpenAIChatAPI] = None,
    trajectory_db: Optional[TrajectoryDatabase] = None,
    encoder: Optional[TrajectoryEncoder] = None,
    step_executor: Optional[StepExecutor] = None,
    category: str = "default",
    auto_register_skills: bool = True,
    verbose: bool = False,
):
    """
    @triage_agent 装饰器

    装饰一个接受 query 字符串并返回字符串的函数。

    Args:
        model_api: LLM API 实例（默认从 config.yaml 读取）
        trajectory_db: 轨迹数据库（默认自动初始化）
        encoder: 语义编码器（默认 sentence-transformers）
        step_executor: 步骤执行器（默认自动初始化）
        category: 轨迹分类标签
        auto_register_skills: 是否自动注册内置技能
        verbose: 是否打印详细日志

    Usage:
        @triage_agent(verbose=True)
        def my_agent(query: str) -> str:
            return f"Result for: {query}"

        result = my_agent("What is the online rate of sensors?")
        print(result.level, result.token_saved)

        # 查看统计
        print(my_agent.triage_stats())
    """

    def decorator(func: Callable) -> Callable:
        wrapper = TriageAgentWrapper(
            func=func,
            model_api=model_api,
            trajectory_db=trajectory_db,
            encoder=encoder,
            step_executor=step_executor,
            category=category,
            auto_register_skills=auto_register_skills,
            verbose=verbose,
        )

        @functools.wraps(func)
        def wrapped(query: str, **kwargs) -> LinkResult:
            return wrapper.run(query, **kwargs)

        # 在 wrapped 函数上挂载辅助方法
        wrapped.triage_stats = wrapper.get_stats
        wrapped.triage_report = wrapper.get_report
        wrapped.triage_wrapper = wrapper

        return wrapped

    return decorator
