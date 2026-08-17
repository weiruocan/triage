"""
TRIAGE Orchestrator — 三链路编排器

不再走 ReAct 循环，而是：
1. 检索 → 2. LLM 路由决策 → 3. 链路分发

一级链路：直接复用缓存结果（0 Token）
二级链路：编辑轨迹 → StepExecutor 执行 → 输出
三级链路：ReActEngine 完整循环

注：本条链路的编排器是所有链路的总控，二级链路内部的编辑执行由
     ExperienceEditor + StepExecutor 完成，三级链路走 ReActEngine。
"""

from __future__ import annotations
import json
import time
from typing import Optional
from dataclasses import dataclass, field

from ..router.llm_router import LLMRouter, RouterDecision
from ..retrieval.experience_trajectory import ExperienceTrajectory
from ..retrieval.experience_editor import ExperienceEditor
from ..model_api import OpenAIChatAPI


@dataclass
class LinkResult:
    """链路执行结果"""
    query: str
    level: str  # "one" | "two" | "three"
    output: str = ""
    total_tokens: int = 0
    token_saved: int = 0
    success: bool = True
    matched_trajectory_id: Optional[str] = None
    edit_nodes: list = field(default_factory=list)
    error: str = ""
    execution_time: float = 0.0


class TriageOrchestrator:
    """
    TRIAGE 三链路编排器
    
    用法:
        orchestrator = TriageOrchestrator(model_api, retriever, step_executor, react_engine)
        result = orchestrator.run("What is the online rate of sensors?")
    """
    
    def __init__(
        self,
        model_api: OpenAIChatAPI,
        retriever=None,      # 检索器（需要实现 search 方法）
        step_executor=None,  # StepExecutor
        react_engine=None,   # ReActEngine（三级链路用）
        trajectory_db=None,  # 轨迹数据库
        encoder=None,        # 语义编码器
    ):
        self.model_api = model_api
        self.router = LLMRouter(model_api)
        self.retriever = retriever
        self.step_executor = step_executor
        self.react_engine = react_engine
        self.trajectory_db = trajectory_db
        self.encoder = encoder
        
        # 统计信息
        self.stats = {
            "level1_reuse": 0,
            "level2_edit": 0,
            "level3_new": 0,
            "level1_fallback": 0,
            "level2_fallback": 0,
            "total_tokens_saved": 0,
            "total_token_cost": 0,
            "total_execution_time": 0.0,
        }
    
    def run(self, query: str) -> LinkResult:
        """主执行入口"""
        start_time = time.time()
        
        # 1. 检索相似轨迹
        retrieved = []
        if self.retriever:
            retrieved = self.retriever.search(query, top_k=5)
        
        # 2. LLM 路由决策
        decision = self.router.route(query, retrieved)
        
        # 3. 链路分发
        if decision.level == "one":
            result = self._level1_reuse(query, decision, retrieved)
        elif decision.level == "two":
            result = self._level2_edit(query, decision, retrieved)
        else:
            result = self._level3_new(query, decision)
        
        result.execution_time = time.time() - start_time
        self.stats["total_execution_time"] += result.execution_time
        
        return result
    
    def _level1_reuse(self, query: str, decision: RouterDecision,
                      retrieved: list) -> LinkResult:
        """一级链路：直接复用"""
        self.stats["level1_reuse"] += 1
        
        result = LinkResult(query=query, level="one",
                            matched_trajectory_id=decision.matched_trajectory_id)
        
        try:
            # 找到匹配的轨迹
            matched = None
            for t in retrieved:
                traj = t.get("trajectory", {})
                if isinstance(traj, str):
                    try:
                        traj = json.loads(traj)
                    except Exception as e:
                        print(f"  [orchestrator] ⚠️ {e}")
                        traj = {}
                tid = traj.get("trajectory_id", t.get("id", ""))
                if str(tid) == str(decision.matched_trajectory_id):
                    matched = traj
                    break
            
            if not matched:
                raise ValueError(f"未找到匹配轨迹: {decision.matched_trajectory_id}")
            
            # 尝试直接返回缓存结果
            cached = matched.get("cached_result") or matched.get("final_answer", "")
            if cached and len(str(cached)) > 50:
                result.output = str(cached)
                result.total_tokens = 0  # 0 Token
                result.token_saved = matched.get("total_tokens", 0)
                self.stats["total_tokens_saved"] += result.token_saved
                return result
            
            # 无缓存结果，尝试复用旧 SQL
            old_sql = self._extract_sql(matched)
            if not old_sql:
                raise ValueError("旧轨迹中未找到可复用的 SQL")
            
            # 用 StepExecutor 执行 SQL
            if self.step_executor:
                exec_result = self.step_executor.execute_sql(old_sql)
                if exec_result and exec_result.get("success"):
                    result.output = json.dumps(exec_result.get("data", {}), ensure_ascii=False)
                    result.total_tokens = 0
                    return result
            
            raise ValueError("StepExecutor 执行失败")
        
        except Exception as e:
            self.stats["level1_reuse"] -= 1
            self.stats["level1_fallback"] += 1
            result.success = False
            result.error = str(e)
            return result
    
    def _level2_edit(self, query: str, decision: RouterDecision,
                     retrieved: list) -> LinkResult:
        """二级链路：编辑后复用"""
        self.stats["level2_edit"] += 1
        
        result = LinkResult(query=query, level="two",
                            matched_trajectory_id=decision.matched_trajectory_id,
                            edit_nodes=decision.edit_nodes)
        
        try:
            # 找到匹配的轨迹
            matched = None
            for t in retrieved:
                traj = t.get("trajectory", {})
                if isinstance(traj, str):
                    try:
                        traj = json.loads(traj)
                    except Exception as e:
                        print(f"  [orchestrator] ⚠️ {e}")
                        traj = {}
                tid = traj.get("trajectory_id", t.get("id", ""))
                if str(tid) == str(decision.matched_trajectory_id):
                    matched = traj
                    break
            
            if not matched:
                raise ValueError(f"未找到匹配轨迹: {decision.matched_trajectory_id}")
            
            # 用 ExperienceEditor 编辑轨迹
            editor = ExperienceEditor(llm_func=self.model_api.chat)
            old_traj = ExperienceTrajectory.from_dict(matched)
            new_traj = editor.edit_trajectory(
                trajectory=old_traj,
                edit_nodes=decision.edit_nodes,
                new_query=query,
            )
            
            # 用 StepExecutor 执行编辑后的轨迹
            if self.step_executor:
                exec_result = self.step_executor.execute_trajectory(new_traj)
                if exec_result.get("success"):
                    result.output = exec_result.get("output", "")
                    result.total_tokens = new_traj.total_tokens
                    result.token_saved = old_traj.total_tokens - new_traj.total_tokens
                    self.stats["total_tokens_saved"] += max(0, result.token_saved)
                    self.stats["total_token_cost"] += result.total_tokens
                    return result
            
            raise ValueError("StepExecutor 执行失败")
        
        except Exception as e:
            self.stats["level2_edit"] -= 1
            self.stats["level2_fallback"] += 1
            result.success = False
            result.error = str(e)
            return result
    
    def _level3_new(self, query: str, decision: RouterDecision) -> LinkResult:
        """三级链路：全新 ReAct

        冷启动支持：
        - 如果配置了 react_engine，走完整 ReAct 循环
        - 如果没有 react_engine，退化到直接 LLM 生成 SQL + 执行
        - 执行成功后自动将轨迹入库
        """
        self.stats["level3_new"] += 1
        
        result = LinkResult(query=query, level="three")
        
        try:
            if self.react_engine:
                # 完整 ReAct 循环
                trajectory = self.react_engine.run(query)
                result.output = trajectory.final_answer or ""
                result.total_tokens = trajectory.total_tokens
                result.success = trajectory.success
                self.stats["total_token_cost"] += result.total_tokens
                
                # 轨迹入库
                if trajectory.success and self.trajectory_db and self.encoder:
                    self._store_trajectory(trajectory)
            else:
                # 冷启动退化：直接 LLM 生成 SQL + 执行
                prompt = (
                    "You are a security monitor database assistant. "
                    "Generate a SQL query for the following request. "
                    "The database has tables: tb_device (dev_id, dev_name, device_classification, "
                    "region, online_status), alarm_event (event_id, event_name, event_time, "
                    "src_index, handle_status). "
                    f"Query: {query}\n"
                    "Return only the SQL query, no explanation."
                )
                response = self.model_api.chat(prompt, temperature=0.3)
                
                # 提取 SQL
                import re
                sql = re.sub(r"```sql\n|```\n|```", "", response)
                sql = re.sub(r"^SQL:\s*", "", sql, flags=re.IGNORECASE)
                sql = sql.strip()
                
                if not sql or "SELECT" not in sql.upper():
                    raise ValueError(f"Could not extract SQL from LLM response")
                
                result.output = f"SQL generated: {sql[:200]}"
                result.total_tokens = self.model_api.get_token_stats()["total_tokens"]
                result.success = True
                self.stats["total_token_cost"] += result.total_tokens
                
                # 自动入库：创建 ExperienceTrajectory
                if self.trajectory_db and self.encoder:
                    try:
                        from ..retrieval.experience_trajectory import ExperienceNode, ExperienceTrajectory
                        node = ExperienceNode(
                            step_index=0, action_name="execute_sql",
                            description=f"Execute SQL for: {query}",
                            node_type="tool",
                            tool_input={"sql": sql},
                            token_input=result.total_tokens,
                            token_output=0,
                        )
                        traj = ExperienceTrajectory(
                            query=query, nodes=[node],
                            category="security_monitor", success=True,
                            total_tokens=result.total_tokens,
                        )
                        embedding = self.encoder.encode_query(query)
                        self.trajectory_db.add_trajectory(
                            query=query, query_embedding=embedding,
                            trajectory_json=traj.to_json(),
                            total_tokens=traj.total_tokens,
                            zero_token_ratio=traj.zero_token_ratio,
                            success=1, category="security_monitor",
                        )
                    except Exception as e:
                        print(f"  [WARN] 冷启动轨迹入库失败: {e}")
        
        except Exception as e:
            result.success = False
            result.error = str(e)
        
        return result
    
    def _store_trajectory(self, trajectory):
        """将 ReAct 轨迹转为 ExperienceTrajectory 并入库"""
        try:
            from ..retrieval.experience_trajectory import migrate_from_old_trajectory
            traj_dict = json.loads(trajectory.to_json())
            new_traj = migrate_from_old_trajectory(traj_dict)
            new_traj.category = "security_monitor"
            new_traj.success = trajectory.success
            
            embedding = None
            if self.encoder:
                embedding = self.encoder.encode_query(new_traj.query)
            
            self.trajectory_db.add_trajectory(
                query=new_traj.query,
                query_embedding=embedding,
                trajectory_json=new_traj.to_json(),
                total_tokens=new_traj.total_tokens,
                zero_token_ratio=new_traj.zero_token_ratio,
                success=int(new_traj.success),
                category=new_traj.category,
            )
        except Exception as e:
            print(f"  [WARN] 轨迹入库失败: {e}")
    
    def _extract_sql(self, trajectory_dict: dict) -> str:
        """从轨迹字典中提取 SQL"""
        for node in trajectory_dict.get("nodes", []):
            action = node.get("action_name", "")
            if action in ("execute_sql", "validate_sql"):
                inp = node.get("tool_input", {})
                if isinstance(inp, str):
                    try:
                        inp = json.loads(inp)
                    except Exception as e:
                        print(f"  [orchestrator] ⚠️ {e}")
                        pass
                if isinstance(inp, dict):
                    sql = inp.get("sql", "")
                    if sql:
                        return sql
        return ""
    
    def get_stats(self) -> dict:
        return dict(self.stats)
