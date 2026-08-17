"""
SkillExtractor — 自动 Skill 提炼

从高频轨迹模式中自动提炼细粒度 Skill。
一条轨迹模式出现 N 次后，自动分析并注册为永久 Skill。
"""

from __future__ import annotations
import json
import re
from typing import Optional
from dataclasses import dataclass, field

from .execution_flow import ExecutionFlow, register_execution_flow, get_execution_flow


@dataclass
class ExtractedSkill:
    """自动提炼的 Skill 信息"""
    skill_name: str
    description: str
    query_pattern: str          # 匹配的查询模式，如 "查{device_type}在线率"
    sql_template: str           # SQL 模板，如 "SELECT ... WHERE device_type='{device_type}'"
    parameters: list[str]       # 可变参数列表，如 ["device_type", "date"]
    source_trajectory_ids: list = field(default_factory=list)  # 来源轨迹 ID
    frequency: int = 1          # 出现次数


class SkillExtractor:
    """
    从轨迹中自动提炼细粒度 Skill。
    
    用法:
        extractor = SkillExtractor()
        skill = extractor.extract_from_trajectories(trajectories)
        if skill:
            extractor.register_skill(skill)
    """
    
    def __init__(self, min_frequency: int = 3):
        self.min_frequency = min_frequency
        self._patterns: dict[str, list] = {}  # sql_template -> list of (query, traj_id)
    
    def analyze_trajectory(self, trajectory_json: str, traj_id: int):
        """分析一条轨迹，提取 SQL 模式"""
        try:
            traj = json.loads(trajectory_json) if isinstance(trajectory_json, str) else trajectory_json
            query = traj.get("query", "")
            nodes = traj.get("nodes", [])
            
            for node in nodes:
                if node.get("action_name") != "execute_sql":
                    continue
                tool_input = node.get("tool_input", {})
                sql = tool_input.get("sql", "") if tool_input else ""
                if not sql:
                    continue
                
                # 用 SQL 模板作为 key（去掉具体值，保留结构）
                template = self._normalize_sql(sql)
                if template not in self._patterns:
                    self._patterns[template] = []
                self._patterns[template].append({
                    "query": query,
                    "traj_id": traj_id,
                    "sql": sql,
                    "node": node,
                })
        except (json.JSONDecodeError, KeyError):
            pass
    
    def extract_skills(self) -> list[ExtractedSkill]:
        """从分析结果中提炼 Skill"""
        skills = []
        
        for template, instances in self._patterns.items():
            if len(instances) < self.min_frequency:
                continue
            
            # 取第一条实例作为参考
            first = instances[0]
            sql = first["sql"]
            query = first["query"]
            traj_ids = [inst["traj_id"] for inst in instances]
            
            # 提取可变参数（对比多个实例的 SQL）
            params = self._extract_parameters(instances)
            
            # 生成 Skill 名
            skill_name = self._generate_skill_name(query, params)
            
            # 生成 SQL 模板
            sql_template = self._generate_template(sql, params, instances)
            
            skill = ExtractedSkill(
                skill_name=skill_name,
                description=f"自动提炼：{query}",
                query_pattern=query,
                sql_template=sql_template,
                parameters=params,
                source_trajectory_ids=traj_ids,
                frequency=len(instances),
            )
            skills.append(skill)
        
        return skills
    
    def register_skill(self, skill: ExtractedSkill) -> bool:
        """将提炼的 Skill 注册到 ExecutionFlow 注册表"""
        if get_execution_flow(skill.skill_name):
            return False  # 已存在
        
        def executor(**kwargs):
            """自动生成的 Skill 执行器"""
            sql = skill.sql_template
            for param in skill.parameters:
                if param in kwargs:
                    # 替换 {param} 为实际值，处理引号
                    val = kwargs[param]
                    sql = sql.replace(f"'{'{'+param+'}'}'", f"'{val}'")
                    sql = sql.replace(f"'{'{'+param+'}'}", f"'{val}'")
                    sql = sql.replace(f"{{{param}}}", str(val))
            # 调用底层 execute_sql
            exec_flow = get_execution_flow("execute_sql")
            if exec_flow:
                return exec_flow.execute(sql=sql)
            return {"success": False, "error": "execute_sql not found"}
        
        # 构建输入 schema
        properties = {p: {"type": "string"} for p in skill.parameters}
        required = list(skill.parameters)
        
        flow = ExecutionFlow(
            skill_name=skill.skill_name,
            description=skill.description,
            input_schema={"type": "object", "properties": properties, "required": required},
            output_schema={"type": "object", "properties": {"success": {"type": "boolean"}}},
            executor=executor,
            estimated_token_cost=0,
        )
        register_execution_flow(flow)
        return True
    
    def _normalize_sql(self, sql: str) -> str:
        """将 SQL 泛化为模板"""
        # 替换字符串值为占位符
        normalized = re.sub(r"'\w+'", "'{val}'", sql)
        # 替换数字为占位符
        normalized = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '{date}', normalized)
        normalized = re.sub(r'\b\d+\b', '{num}', normalized)
        return normalized
    
    def _extract_parameters(self, instances: list) -> list[str]:
        """从多个实例中提取可变参数"""
        if len(instances) < 2:
            return []
        
        # 取前两个实例，比较 SQL 差异
        sql1 = instances[0]["sql"]
        sql2 = instances[1]["sql"]
        
        params = []
        # 查找不同的字符串值
        vals1 = re.findall(r"'(\w+(?:\s+\w+)*)'", sql1)
        vals2 = re.findall(r"'(\w+(?:\s+\w+)*)'", sql2)
        
        for v1, v2 in zip(vals1, vals2):
            if v1.lower() != v2.lower():
                # 从查询中推断参数名
                query1 = instances[0]["query"]
                params.append(self._infer_param_name(v1, query1))
        
        return list(set(params))
    
    def _infer_param_name(self, value: str, query: str) -> str:
        """从查询中推断参数名"""
        # 常见映射
        value_lower = value.lower()
        if value_lower in ("sensors", "cameras", "vibration fiber", "electronic pulse"):
            return "device_type"
        if re.match(r'\d{4}-\d{2}-\d{2}', value_lower):
            return "date"
        if value_lower in ("area a", "area b", "area c"):
            return "region"
        if value_lower in ("online", "offline"):
            return "status"
        return "param"
    
    def _generate_template(self, sql: str, params: list[str], instances: list) -> str:
        """生成带参数的 SQL 模板"""
        template = sql
        if len(instances) >= 2:
            # 用第一个实例的值替换为参数名
            first = instances[0]
            first_sql = first["sql"]
            vals = re.findall(r"'(\w+(?:\s+\w+)*)'", first_sql)
            for val, param in zip(vals, params):
                template = template.replace(f"'{val}'", f"'{'{'+param+'}'}'", 1)
        return template
    
    def _generate_skill_name(self, query: str, params: list[str]) -> str:
        """从查询和参数生成 Skill 名"""
        # 提取关键词
        keywords = re.findall(r'[\u4e00-\u9fff\w]+', query)
        # 取前 2-3 个关键词
        name_parts = [k for k in keywords if len(k) > 1][:3]
        if params:
            name_parts.extend(params)
        name = "_".join(name_parts)
        # 确保是合法标识符
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        return f"auto_{name}"[:50]


# 全局实例
_extractor = SkillExtractor()


def analyze_and_extract(trajectory_json: str, traj_id: int):
    """分析轨迹并尝试提炼 Skill（供入库时调用）"""
    _extractor.analyze_trajectory(trajectory_json, traj_id)


def flush_skills():
    """将积累的分析结果提炼为 Skill"""
    skills = _extractor.extract_skills()
    registered = 0
    for skill in skills:
        if _extractor.register_skill(skill):
            registered += 1
            print(f"  [skill_extractor] 自动注册 Skill: {skill.skill_name} (频率: {skill.frequency})")
    return registered


__all__ = ["SkillExtractor", "analyze_and_extract", "flush_skills"]
