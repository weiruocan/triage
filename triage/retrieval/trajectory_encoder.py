"""
TrajectoryEncoder — 语义编码器

封装 sentence-transformers，用于轨迹查询的语义相似度检索。

支持两种模式：
1. sentence-transformers（推荐，需要下载模型）
2. 回退编码器（纯 Python，零依赖，适合离线环境）
"""

from __future__ import annotations
import json
import hashlib
import re
from typing import Optional, List


class TrajectoryEncoder:
    """
    语义编码器
    
    用法:
        encoder = TrajectoryEncoder(model_name="all-MiniLM-L6-v2")
        embedding = encoder.encode_query("What is the online rate of sensors?")
        similarity = encoder.compute_similarity(embedding1, embedding2)
    """
    
    def __init__(self, model_name: str = "", model_path: Optional[str] = None):
        self.model_name = model_name
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__),
            "../../data/models/sentence-transformers/all-MiniLM-L6-v2"
        )
        self._model = None
        self._dimension = 384  # all-MiniLM-L6-v2 默认维度
    
    def _load_model(self):
        """加载 sentence-transformers 模型"""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # 优先使用本地模型路径
            local_path = os.path.abspath(self.model_path)
            if os.path.exists(local_path):
                self._model = SentenceTransformer(local_path)
            elif self.model_name:
                self._model = SentenceTransformer(self.model_name)
            else:
                raise RuntimeError(f"Model not found at {local_path} and no model_name specified")
            
            self._dimension = self._model.get_sentence_embedding_dimension()
            print(f"  [encoder] 加载模型: {os.path.abspath(self.model_path)} (dim={self._dimension})")
            
        except ImportError:
            print(f"  [encoder] sentence-transformers 未安装，使用回退编码器")
            self._model = None
            self._dimension = 64
    
    def encode_query(self, query: str) -> list:
        """
        将查询编码为向量
        
        Returns:
            list of float: 向量
        """
        if self._model is None:
            self._load_model()
        
        if self._model is not None:
            # 使用 sentence-transformers
            embedding = self._model.encode(query, normalize_embeddings=True)
            return embedding.tolist()
        else:
            # 回退：基于关键词的哈希编码
            return self._fallback_encode(query)
    
    def encode_queries(self, queries: List[str]) -> List[list]:
        """批量编码"""
        if self._model is None:
            self._load_model()
        
        if self._model is not None:
            embeddings = self._model.encode(queries, normalize_embeddings=True)
            return [e.tolist() for e in embeddings]
        else:
            return [self._fallback_encode(q) for q in queries]
    
    def _fallback_encode(self, text: str) -> list:
        """
        回退编码器：基于关键词的哈希编码
        
        零依赖，但语义区分度较低。
        适用于离线环境或无 GPU 的场景。
        """
        # 提取关键词
        words = set(re.findall(r'[a-zA-Z_]+', text.lower()))
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 
                        'in', 'on', 'at', 'to', 'for', 'of', 'with',
                        'and', 'or', 'but', 'not', 'what', 'which',
                        'how', 'show', 'list', 'get', 'find', 'do',
                        'i', 'my', 'me', 'we', 'our', 'can', 'you'}
        keywords = words - common_words
        
        # 哈希编码
        embedding = []
        for i in range(self._dimension):
            seed = f"{text}_{i}".encode()
            h = hashlib.md5(seed).hexdigest()
            val = (int(h[:8], 16) % 1000) / 1000.0  # 0.0 ~ 1.0
            embedding.append(val)
        
        # 关键词加权
        for keyword in keywords:
            for i in range(min(len(keyword), self._dimension)):
                seed = f"{keyword}_{i}".encode()
                h = hashlib.md5(seed).hexdigest()
                val = (int(h[:8], 16) % 200 - 100) / 1000.0  # -0.1 ~ 0.1
                embedding[i] += val
        
        # 归一化
        norm = sum(v * v for v in embedding) ** 0.5
        if norm > 0:
            embedding = [v / norm for v in embedding]
        
        return embedding
    
    @staticmethod
    def compute_similarity(embedding1: list, embedding2: list) -> float:
        """计算余弦相似度"""
        if not embedding1 or not embedding2:
            return 0.0
        dot = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    
    def get_dimension(self) -> int:
        """获取编码维度"""
        return self._dimension


import os  # noqa: E402

if __name__ == "__main__":
    encoder = TrajectoryEncoder()
    
    # 测试编码
    q1 = "What is the online rate of sensors?"
    q2 = "What is the online rate of cameras?"
    q3 = "Show the alarm distribution by type"
    
    e1 = encoder.encode_query(q1)
    e2 = encoder.encode_query(q2)
    e3 = encoder.encode_query(q3)
    
    print(f"维度: {encoder.get_dimension()}")
    print(f"相似度 (q1 vs q2): {encoder.compute_similarity(e1, e2):.4f}")
    print(f"相似度 (q1 vs q3): {encoder.compute_similarity(e1, e3):.4f}")
