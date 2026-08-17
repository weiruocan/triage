"""
注册旧项目的 Skill 到 ExecutionFlow 注册表

使用直接依赖注入的方式，避免旧项目的相对导入问题。
"""

import os
import sqlite3
import re

from .execution_flow import register_execution_flow, ExecutionFlow


# 数据库路径
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/security_monitor.db"))


def _sql_validate(sql: str) -> dict:
    """验证 SQL 语法"""
    try:
        sql = re.sub(r'```sql\n|```\n|```', '', sql)
        sql = re.sub(r'^SQL:\s*', '', sql, flags=re.IGNORECASE)
        sql = sql.strip()
        if not sql:
            return {"success": False, "error": "SQL is empty"}
        sql_upper = sql.upper()
        if 'SELECT' not in sql_upper or 'FROM' not in sql_upper:
            return {"success": False, "error": "SQL must contain SELECT and FROM"}
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(f"EXPLAIN {sql}")
            return {"success": True, "data": sql}
        except sqlite3.Error as e:
            return {"success": False, "error": f"SQL syntax error: {str(e)}"}
        finally:
            conn.close()
    except Exception as e:
        return {"success": False, "error": str(e)}


def _sql_execute(sql: str, max_rows: int = 100) -> dict:
    """执行 SQL 查询"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchmany(max_rows)
        if not rows:
            return {"success": True, "data": {"columns": [], "rows": [], "total": 0}}
        columns = [desc[0] for desc in cursor.description]
        return {
            "success": True,
            "data": {"columns": columns, "rows": [dict(row) for row in rows], "total": len(rows)},
        }
    except sqlite3.Error as e:
        return {"success": False, "error": f"SQL error: {str(e)}"}
    finally:
        if conn:
            conn.close()


def _report_generate(content: str, output_path: str = None) -> dict:
    """生成报告"""
    try:
        if output_path is None:
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output"))
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "report.md")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"success": True, "data": {"output_path": output_path, "length": len(content)}}
    except Exception as e:
        return {"success": False, "error": str(e)}


def register_all_skills():
    """注册所有技能"""
    from .execution_flow import list_execution_flows
    count_before = len(list_execution_flows())
    
    print("  [register_skills] 注册 ExecutionFlow 执行器...")
    
    # validate_sql
    register_execution_flow(ExecutionFlow(
        skill_name="validate_sql",
        description="验证 SQL 查询语句的语法正确性",
        input_schema={"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
        output_schema={"type": "object", "properties": {"success": {"type": "boolean"}, "data": {"type": "string"}}},
        executor=_sql_validate,
    ))
    
    # execute_sql
    register_execution_flow(ExecutionFlow(
        skill_name="execute_sql",
        description="在安保监控数据库中执行 SQL 查询",
        input_schema={"type": "object", "properties": {
            "sql": {"type": "string"}, "max_rows": {"type": "integer"},
        }, "required": ["sql"]},
        output_schema={"type": "object", "properties": {
            "columns": {"type": "array"}, "rows": {"type": "array"}, "total": {"type": "integer"}
        }},
        executor=_sql_execute,
    ))
    
    # generate_report
    register_execution_flow(ExecutionFlow(
        skill_name="generate_report",
        description="将分析结果和图表整合为最终报告",
        input_schema={"type": "object", "properties": {
            "content": {"type": "string"}, "output_path": {"type": "string"},
        }, "required": ["content"]},
        output_schema={"type": "object", "properties": {"output_path": {"type": "string"}}},
        executor=_report_generate,
    ))
    
    # create_bar_chart
    register_execution_flow(ExecutionFlow(
        skill_name="create_bar_chart",
        description="生成柱状图",
        input_schema={"type": "object", "properties": {
            "data": {"type": "object"}, "title": {"type": "string"},
        }, "required": ["data", "title"]},
        output_schema={"type": "object", "properties": {"image_path": {"type": "string"}}},
        executor=lambda **kwargs: {"success": True, "data": {"image_path": "chart_placeholder.png"}},
    ))
    
    flows = list_execution_flows()
    new_count = len(flows) - count_before
    print(f"  [register_skills] 注册完成: 共 {len(flows)} 个（新增 {new_count} 个）")
    for f in flows:
        print(f"    - {f}")
    return flows


if __name__ == "__main__":
    register_all_skills()
