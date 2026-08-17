"""
FAISS 索引 — 轨迹向量检索加速

当轨迹库规模 > 5000 条时，FAISS 索引可以显著加速检索。
当前使用 IVF flat 索引（倒排 + 精确搜索），平衡速度与精度。

构建流程：
1. 从 SQLite 读取所有轨迹的 query_embedding
2. 用 FAISS 建 IVF flat 索引
3. 检索时：query_embedding → FAISS search → 返回 top-k 轨迹 ID

用法:
    from triage.retrieval.faiss_index import FaissIndex
    index = FaissIndex()
    index.build(trajectory_db)       # 构建索引
    results = index.search(query_embedding, top_k=5)  # 检索
    index.add(embedding, traj_id)    # 增量添加
"""

from __future__ import annotations
import json
import os
import numpy as np
from typing import Optional, List


INDEX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/trajectories/faiss_index.bin"))
MAPPING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/trajectories/faiss_mapping.json"))


class FaissIndex:
    """FAISS 向量索引封装"""

    def __init__(self, dimension: int = 384, index_path: str = None):
        self.dimension = dimension
        self.index_path = index_path or INDEX_PATH
        self.mapping_path = os.path.join(os.path.dirname(self.index_path), "faiss_mapping.json")
        self._index = None
        self._id_mapping = {}  # FAISS index → trajectory_id

    def _ensure_faiss(self):
        """确保 FAISS 可用"""
        try:
            import faiss
            return faiss
        except ImportError:
            raise ImportError(
                "FAISS 未安装。运行: pip install faiss-cpu\n"
                "如果安装失败，系统会回退到线性扫描（当前模式，足够支持 1704 条轨迹）"
            )

    def build(self, embeddings: List[list], traj_ids: List[int]):
        """
        构建 FAISS 索引

        Args:
            embeddings: 向量列表，每个元素是 list[float]
            traj_ids: 对应的轨迹 ID 列表
        """
        faiss = self._ensure_faiss()
        
        if not embeddings:
            print("  [faiss] 没有向量数据，跳过索引构建")
            return

        embeddings_np = np.array(embeddings, dtype=np.float32)
        self.dimension = embeddings_np.shape[1]

        # IVF flat 索引
        nlist = min(int(np.sqrt(len(embeddings))), 100)  # 聚类中心数
        quantizer = faiss.IndexFlatL2(self.dimension)
        self._index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)

        # 训练并添加
        self._index.train(embeddings_np)
        self._index.add(embeddings_np)

        # ID 映射
        self._id_mapping = {i: traj_ids[i] for i in range(len(traj_ids))}

        # 保存到磁盘
        self.save()

        print(f"  [faiss] 索引构建完成: {len(embeddings)} 条向量, dim={self.dimension}, nlist={nlist}")

    def search(self, query_embedding: list, top_k: int = 5) -> List[dict]:
        """
        检索最近邻

        Args:
            query_embedding: 查询向量
            top_k: 返回数量

        Returns:
            [{"id": traj_id, "distance": float, "similarity": float}, ...]
        """
        if self._index is None:
            self.load()

        if self._index is None:
            return []

        faiss = self._ensure_faiss()
        query_np = np.array([query_embedding], dtype=np.float32)

        # IVF 需要先设置 nprobe
        self._index.nprobe = min(10, self._index.ntotal)

        distances, indices = self._index.search(query_np, min(top_k, self._index.ntotal))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            traj_id = self._id_mapping.get(int(idx))
            if traj_id is not None:
                # L2 距离转余弦相似度（近似）
                l2_dist = float(distances[0][i])
                similarity = max(0.0, 1.0 - l2_dist / 4.0)  # 粗略转换
                results.append({
                    "id": traj_id,
                    "distance": round(l2_dist, 4),
                    "similarity": round(similarity, 4),
                })

        return results

    def add(self, embedding: list, traj_id: int):
        """增量添加一条向量"""
        if self._index is None:
            self.load()

        faiss = self._ensure_faiss()
        embedding_np = np.array([embedding], dtype=np.float32)
        self._index.add(embedding_np)

        new_idx = self._index.ntotal - 1
        self._id_mapping[new_idx] = traj_id

    def save(self):
        """保存索引到磁盘"""
        if self._index is None:
            return

        faiss = self._ensure_faiss()
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self._index, self.index_path)
        with open(self.mapping_path, 'w') as f:
            json.dump(self._id_mapping, f)

    def load(self):
        """从磁盘加载索引"""
        if not os.path.exists(self.index_path):
            return

        faiss = self._ensure_faiss()
        self._index = faiss.read_index(self.index_path)
        self.dimension = self._index.d

        if os.path.exists(self.mapping_path):
            with open(self.mapping_path) as f:
                self._id_mapping = json.load(f)
                # 确保 key 是 int
                self._id_mapping = {int(k): v for k, v in self._id_mapping.items()}

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0


def build_faiss_index_from_db(trajectory_db=None, encoder=None):
    """
    从轨迹数据库构建 FAISS 索引

    Args:
        trajectory_db: TrajectoryDatabase 实例
        encoder: TrajectoryEncoder 实例

    Returns:
        FaissIndex 实例
    """
    if trajectory_db is None:
        from .trajectory_db import TrajectoryDatabase
        trajectory_db = TrajectoryDatabase()

    if encoder is None:
        from .trajectory_encoder import TrajectoryEncoder
        encoder = TrajectoryEncoder()

    # 获取所有轨迹
    all_trajs = trajectory_db.get_all_trajectories(limit=100000)

    if not all_trajs:
        print("  [faiss] 轨迹库为空，跳过索引构建")
        return None

    # 编码所有查询
    queries = [t["query"] for t in all_trajs]
    traj_ids = [t["id"] for t in all_trajs]

    print(f"  [faiss] 编码 {len(queries)} 条查询...")
    embeddings = encoder.encode_queries(queries)

    # 构建索引
    index = FaissIndex(dimension=encoder.get_dimension())
    index.build(embeddings, traj_ids)

    return index


if __name__ == "__main__":
    print("构建 FAISS 索引...")
    index = build_faiss_index_from_db()
    if index:
        print(f"  索引大小: {index.size} 条")
        print(f"  索引路径: {index.index_path}")
    else:
        print("  FAISS 索引构建失败")
