"""
trajectory_db — 轨迹数据库

SQLite + FAISS 存储和检索历史执行轨迹。

默认路径：data/trajectories.db（用户新建项目时从头开始）
可通过环境变量 TRIAGE_DB_PATH 或构造函数参数覆盖。

用法：
    db = TrajectoryDatabase()
    db.add_trajectory(query="...", trajectory_json="...")
    results = db.search("What is the online rate?", top_k=5)
    stats = db.get_statistics()
"""

from __future__ import annotations
import os
import json
import sqlite3
from typing import List, Optional

from .trajectory_encoder import TrajectoryEncoder
from .faiss_index import FaissIndex

# 默认数据库路径：环境变量 > 项目 data 目录 > 内存
DB_PATH = os.environ.get(
    "TRIAGE_DB_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/trajectories.db")),
)

try:
    import faiss as _faiss_module
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class TrajectoryDatabase:
    """
    轨迹数据库

    支持环境变量覆盖：
        export TRIAGE_DB_PATH="./my_trajectories.db"

    用法:
        db = TrajectoryDatabase()
        db.add_trajectory(query="...", trajectory_json="...")
        results = db.search("What is the online rate?", top_k=5)
        stats = db.get_statistics()
    """

    def __init__(self, db_path: str = None, encoder: TrajectoryEncoder = None,
                 use_faiss: bool = True):
        self.db_path = db_path or DB_PATH
        self.encoder = encoder or TrajectoryEncoder()
        self.use_faiss = use_faiss and _HAS_FAISS
        self._faiss_index = None
        if self.use_faiss:
            self._init_faiss()
        self._ensure_db()

    def _init_faiss(self):
        """初始化 FAISS 索引"""
        try:
            self._faiss_index = FaissIndex()
        except Exception:
            self._faiss_index = None
            self.use_faiss = False

    def rebuild_faiss(self):
        """从 SQLite 重建 FAISS 索引"""
        if not _HAS_FAISS:
            return
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT id, query_embedding FROM experience_trajectories WHERE query_embedding IS NOT NULL"
        ).fetchall()
        conn.close()
        if rows:
            ids = [row[0] for row in rows]
            embeddings = [json.loads(row[1]) for row in rows if row[1]]
            self._faiss_index = FaissIndex()
            self._faiss_index.build(embeddings, ids)

    def _ensure_db(self):
        """确保数据库存在（自动创建目录和表）"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experience_trajectories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                query_embedding TEXT,
                trajectory_json TEXT NOT NULL,
                total_tokens INTEGER DEFAULT 0,
                zero_token_ratio REAL DEFAULT 0.0,
                success INTEGER DEFAULT 0,
                category TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        conn.close()

    def add_trajectory(self, query: str, trajectory_json: str,
                       query_embedding: Optional[list] = None,
                       total_tokens: int = 0, zero_token_ratio: float = 0.0,
                       success: int = 1, category: str = "") -> int:
        """添加轨迹"""
        if query_embedding is None:
            query_embedding = self.encoder.encode_query(query)

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO experience_trajectories "
            "(query, query_embedding, trajectory_json, total_tokens, zero_token_ratio, success, category) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (query, json.dumps(query_embedding), trajectory_json,
             total_tokens, zero_token_ratio, success, category),
        )
        conn.commit()
        traj_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
        return traj_id

    def search(self, query: str, top_k: int = 5, category: str = None) -> List[dict]:
        """
        检索相似轨迹

        优先使用 FAISS 索引，否则回退到线性扫描。
        """
        query_embedding = self.encoder.encode_query(query)

        # FAISS 检索
        if self._faiss_index is not None and self._faiss_index.is_loaded and category is None:
            faiss_results = self._faiss_index.search(query_embedding, top_k=top_k)
            results = []
            conn = sqlite3.connect(self.db_path)
            for r in faiss_results:
                row = conn.execute(
                    "SELECT id, query, trajectory_json, total_tokens, zero_token_ratio, "
                    "success, category, created_at FROM experience_trajectories WHERE id = ?",
                    (r["id"],)
                ).fetchone()
                if row:
                    results.append({
                        "id": row[0], "query": row[1], "trajectory": row[2],
                        "total_tokens": row[3], "zero_token_ratio": row[4],
                        "success": bool(row[5]), "category": row[6],
                        "created_at": row[7], "similarity": r["similarity"],
                    })
            conn.close()
            return results

        # 线性扫描（回退）
        conn = sqlite3.connect(self.db_path)
        sql = ("SELECT id, query, query_embedding, trajectory_json, total_tokens, "
               "zero_token_ratio, success, category, created_at FROM experience_trajectories")
        params = []
        if category:
            sql += " WHERE category = ?"
            params.append(category)
        sql += " ORDER BY id DESC"
        rows = conn.execute(sql, params).fetchall()
        conn.close()

        results = []
        for row in rows:
            try:
                stored_embedding = json.loads(row[2]) if row[2] else None
            except (json.JSONDecodeError, TypeError):
                stored_embedding = None
            if stored_embedding:
                similarity = self.encoder.compute_similarity(query_embedding, stored_embedding)
            else:
                similarity = 0.0
            results.append({
                "id": row[0], "query": row[1], "trajectory": row[3],
                "total_tokens": row[4], "zero_token_ratio": row[5],
                "success": bool(row[6]), "category": row[7],
                "created_at": row[8], "similarity": similarity,
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def get_by_id(self, traj_id: int) -> Optional[dict]:
        """根据 ID 获取轨迹"""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT id, query, trajectory_json, total_tokens, zero_token_ratio, "
            "success, category, created_at FROM experience_trajectories WHERE id = ?",
            (traj_id,)
        ).fetchone()
        conn.close()
        if row:
            return {
                "id": row[0], "query": row[1], "trajectory": row[2],
                "total_tokens": row[3], "zero_token_ratio": row[4],
                "success": bool(row[5]), "category": row[6],
                "created_at": row[7],
            }
        return None

    def get_by_query(self, query: str) -> Optional[dict]:
        """根据精确查询获取轨迹"""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT id, query, trajectory_json, total_tokens, zero_token_ratio, "
            "success, category, created_at FROM experience_trajectories WHERE query = ?",
            (query,)
        ).fetchone()
        conn.close()
        if row:
            return {
                "id": row[0], "query": row[1], "trajectory": row[2],
                "total_tokens": row[3], "zero_token_ratio": row[4],
                "success": bool(row[5]), "category": row[6],
                "created_at": row[7],
            }
        return None

    def update_trajectory(self, traj_id: int, **kwargs):
        """更新轨迹字段"""
        allowed = {"total_tokens", "zero_token_ratio", "success", "category"}
        updates = [(k, v) for k, v in kwargs.items() if k in allowed]
        if not updates:
            return
        conn = sqlite3.connect(self.db_path)
        set_clause = ", ".join(f"{k} = ?" for k, _ in updates)
        values = [v for _, v in updates]
        conn.execute(
            f"UPDATE experience_trajectories SET {set_clause} WHERE id = ?",
            values + [traj_id],
        )
        conn.commit()
        conn.close()

    def delete_trajectory(self, traj_id: int):
        """删除轨迹"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM experience_trajectories WHERE id = ?", (traj_id,))
        conn.commit()
        conn.close()

    def count(self, category: str = None) -> int:
        """轨迹数量"""
        conn = sqlite3.connect(self.db_path)
        if category:
            row = conn.execute(
                "SELECT COUNT(*) FROM experience_trajectories WHERE category = ?",
                (category,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM experience_trajectories"
            ).fetchone()
        conn.close()
        return row[0] if row else 0

    def get_statistics(self) -> dict:
        """获取数据库统计信息"""
        conn = sqlite3.connect(self.db_path)
        total = conn.execute(
            "SELECT COUNT(*), SUM(total_tokens), AVG(total_tokens), AVG(zero_token_ratio) "
            "FROM experience_trajectories"
        ).fetchone()
        conn.close()
        return {
            "total_trajectories": total[0] or 0,
            "total_tokens": total[1] or 0,
            "avg_tokens": round(total[2], 1) if total[2] else 0,
            "avg_zero_token_ratio": round(total[3], 3) if total[3] else 0,
        }

    def get_all_trajectories(self, category: str = None,
                              limit: int = 100, offset: int = 0) -> List[dict]:
        """获取轨迹列表"""
        conn = sqlite3.connect(self.db_path)
        sql = ("SELECT id, query, trajectory_json, total_tokens, zero_token_ratio, "
               "success, category, created_at FROM experience_trajectories")
        params = []
        if category:
            sql += " WHERE category = ?"
            params.append(category)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [{
            "id": row[0], "query": row[1], "trajectory": row[2],
            "total_tokens": row[3], "zero_token_ratio": row[4],
            "success": bool(row[5]), "category": row[6],
            "created_at": row[7],
        } for row in rows]


__all__ = ["TrajectoryDatabase"]
