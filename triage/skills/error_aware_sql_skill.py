"""
ErrorAwareSQLSkill — 错误感知的 SQL 生成技能

核心能力：
1. 维护错误指纹数据库（记录失败 SQL + 正确 SQL + 上下文）
2. 生成 SQL 时检索历史错误经验，注入 prompt
3. 自动验证（EXPLAIN）+ 重试（最多 3 次）
4. 成功时标记错误经验为"已修复"

Token 消耗：~200/次（单次 LLM 调用，最小上下文）
"""

from __future__ import annotations
import json
import os
import re
import sqlite3
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from ..model_api import OpenAIChatAPI


# 安保监控数据库 schema（最小上下文）
DB_SCHEMA = """
## Database Tables

### 1. tb_device (device table)
- dev_id: device code (PK)
- dev_name: device name
- device_classification: device type (camera, vibration_fiber, electronic_pulse)
- region: region name (e.g., Area A-East Gate, Perimeter-East)
- latitude: latitude
- longitude: longitude
- online_status: online status (0-offline, 1-online)
- install_date: installation date

### 2. alarm_event (event table)
- event_id: event unique ID (PK)
- event_name: event type name
- warning_classification: warning classification
- event_time: event time (YYYY-MM-DD HH:MM:SS)
- src_index: source device code (FK to tb_device.dev_id)
- src_name: source device name
- handle_status: handling status (0-unhandled, 1-handled)

### 3. tb_event_operation (event handling record)
- id: PK
- event_aggregation_id: event aggregation ID (FK to alarm_event.event_id)
- operator: handler name
- description: handling notes
- op_time: handling time
- event_type: 0-false alarm, 1-real alarm
- status: 0-received, 1-in progress, 2-completed
"""

# 错误指纹数据库路径
ERROR_DB_PATH = os.path.join(
    os.path.dirname(__file__), "../../data/error_fingerprints.db"
)


@dataclass
class SQLResult:
    """SQL 生成结果"""
    success: bool
    sql: str = ""
    raw_response: str = ""
    retry_count: int = 0
    token_cost: int = 0
    error: str = ""


class ErrorAwareSQLSkill:
    """
    错误感知的 SQL 生成技能
    
    用法:
        skill = ErrorAwareSQLSkill(model_api)
        result = skill.generate("What is the online rate of sensors?")
    """
    
    def __init__(self, model_api: Optional[OpenAIChatAPI] = None):
        self.model_api = model_api
        self.max_retries = 3
        self._init_error_db()
    
    def set_model_api(self, model_api: OpenAIChatAPI):
        self.model_api = model_api
    
    def _init_error_db(self):
        """初始化错误指纹数据库"""
        os.makedirs(os.path.dirname(ERROR_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(ERROR_DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS error_fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_type TEXT NOT NULL,
                error_sql TEXT,
                correct_sql TEXT,
                user_query TEXT,
                db_schema_hash TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                fixed_count INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()
    
    def generate(self, user_query: str, context: Optional[dict] = None) -> SQLResult:
        """
        生成 SQL
        
        Args:
            user_query: 用户查询（自然语言）
            context: 额外上下文（如表名、时间范围等）
        
        Returns:
            SQLResult
        """
        if not self.model_api:
            return SQLResult(success=False, error="model_api not configured")
        
        last_error = None
        
        for attempt in range(self.max_retries):
            # 1. 检索相关错误经验
            similar_errors = self._search_similar_errors(user_query)
            
            # 2. 构造最小上下文 prompt
            prompt = self._build_prompt(user_query, similar_errors, attempt, context)
            
            # 3. 单次 LLM 调用
            try:
                response = self.model_api.chat(prompt, temperature=0.3)
                sql = self._extract_sql(response)
                token_cost = self.model_api.get_token_stats().get("total_tokens", 0)
            except Exception as e:
                last_error = f"LLM call failed: {str(e)}"
                continue
            
            if not sql:
                last_error = "Could not extract SQL from response"
                continue
            
            # 4. 自动验证
            is_valid, error_msg = self._validate_sql(sql)
            if is_valid:
                # 成功：标记错误经验为已修复
                if attempt > 0 and similar_errors:
                    for err in similar_errors:
                        self._increment_fixed_count(err["id"])
                return SQLResult(
                    success=True, sql=sql, raw_response=response,
                    retry_count=attempt, token_cost=token_cost,
                )
            
            # 5. 失败：记录错误指纹
            last_error = error_msg
            self._add_fingerprint("sql_validation_error", sql, user_query)
        
        return SQLResult(
            success=False, error=f"SQL generation failed after {self.max_retries} retries: {last_error}",
        )
    
    def _build_prompt(self, user_query: str, similar_errors: list,
                      attempt: int, context: Optional[dict] = None) -> str:
        """构造最小上下文 prompt"""
        parts = [
            "You are a SQL expert. Generate a SQLite query based on the database schema and user query.",
            "",
            "## Database Schema",
            DB_SCHEMA,
            "",
            "## SQLite Notes",
            "- event_time is TEXT, format 'YYYY-MM-DD HH:MM:SS'",
            "- Use strftime for date: strftime('%Y-%m-%d', event_time)",
            "- Last week = past 7 days",
            "- Use SQLite syntax, not PostgreSQL",
        ]
        
        if context:
            parts.append("")
            parts.append("## Additional Context")
            for k, v in context.items():
                parts.append(f"- {k}: {v}")
        
        if similar_errors:
            parts.append("")
            parts.append("## Historical Error Experience (AVOID These)")
            for i, err in enumerate(similar_errors, 1):
                parts.append(f"{i}. Error: {err['error_type']}")
                parts.append(f"   Wrong SQL: {err['error_sql']}")
                if err.get('correct_sql'):
                    parts.append(f"   Correct SQL: {err['correct_sql']}")
                parts.append("")
        
        parts.append("")
        parts.append("## User Query")
        parts.append(user_query)
        parts.append("")
        parts.append("## Requirements")
        parts.append("1. Output ONLY the SQL statement, no explanation")
        parts.append("2. Start with SELECT, avoid DDL or DML")
        parts.append("3. Use explicit WHERE conditions for time ranges")
        parts.append("4. Use GROUP BY for aggregations")
        
        return "\n".join(parts)
    
    def _extract_sql(self, text: str) -> str:
        """从 LLM 响应中提取纯 SQL"""
        text = re.sub(r'```sql\n|```\n|```', '', text)
        text = re.sub(r'^SQL:\s*', '', text, flags=re.IGNORECASE)
        text = text.strip()
        
        idx = text.upper().find('SELECT')
        if idx >= 0:
            text = text[idx:]
        
        lines = text.split('\n')
        sql_lines = []
        for line in lines:
            if line.strip() and not line.strip().upper().startswith(('--', '/*')):
                sql_lines.append(line)
            elif line.strip().startswith('--'):
                continue
            else:
                break
        return '\n'.join(sql_lines).strip() if sql_lines else text
    
    def _validate_sql(self, sql: str) -> Tuple[bool, str]:
        """用 EXPLAIN 验证 SQL 语法"""
        if not sql or not sql.strip():
            return False, "SQL is empty"
        sql_upper = sql.upper()
        if 'SELECT' not in sql_upper or 'FROM' not in sql_upper:
            return False, "SQL must contain SELECT and FROM"
        
        db_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "../../data/security_monitor.db"
        ))
        
        if not os.path.exists(db_path):
            return True, ""  # 数据库不存在时跳过验证
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(f"EXPLAIN {sql}")
            conn.close()
            return True, ""
        except sqlite3.Error as e:
            return False, f"SQL syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    # ===== 错误指纹数据库操作 =====
    
    def _search_similar_errors(self, query: str, top_k: int = 3) -> list:
        """检索相关错误经验"""
        conn = sqlite3.connect(ERROR_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, error_type, error_sql, correct_sql, user_query,
                   created_at, fixed_count
            FROM error_fingerprints
            ORDER BY fixed_count DESC, created_at DESC
            LIMIT ?
        """, (top_k * 3,))
        
        results = []
        query_keywords = set(re.findall(r'[a-zA-Z_]+', query.lower()))
        
        for row in cursor.fetchall():
            user_q = (row[4] or "").lower()
            q_keywords = set(re.findall(r'[a-zA-Z_]+', user_q))
            overlap = len(query_keywords & q_keywords)
            if overlap > 0 or len(results) < top_k:
                results.append({
                    "id": row[0], "error_type": row[1],
                    "error_sql": row[2], "correct_sql": row[3],
                    "user_query": row[4], "created_at": row[5],
                    "fixed_count": row[6],
                })
        
        conn.close()
        return results[:top_k]
    
    def _add_fingerprint(self, error_type: str, error_sql: str,
                         user_query: str, correct_sql: str = None):
        conn = sqlite3.connect(ERROR_DB_PATH)
        conn.execute(
            "INSERT INTO error_fingerprints (error_type, error_sql, correct_sql, user_query) VALUES (?, ?, ?, ?)",
            (error_type, error_sql, correct_sql, user_query),
        )
        conn.commit()
        conn.close()
    
    def _increment_fixed_count(self, fingerprint_id: int):
        conn = sqlite3.connect(ERROR_DB_PATH)
        conn.execute("UPDATE error_fingerprints SET fixed_count = fixed_count + 1 WHERE id = ?",
                     (fingerprint_id,))
        conn.commit()
        conn.close()
    
    def get_error_statistics(self) -> dict:
        """获取错误经验统计"""
        conn = sqlite3.connect(ERROR_DB_PATH)
        total = conn.execute("SELECT COUNT(*) FROM error_fingerprints").fetchone()[0]
        type_dist = dict(conn.execute(
            "SELECT error_type, COUNT(*) FROM error_fingerprints GROUP BY error_type"
        ).fetchall())
        conn.close()
        return {"total": total, "type_distribution": type_dist}


if __name__ == "__main__":
    # 测试
    from ..model_api import OpenAIChatAPI
    
    api = OpenAIChatAPI()
    skill = ErrorAwareSQLSkill(api)
    
    result = skill.generate("What is the online rate of sensors?")
    print(f"SQL: {result.sql}")
    print(f"Retry: {result.retry_count}")
    print(f"Tokens: {result.token_cost}")
    print(f"Success: {result.success}")
