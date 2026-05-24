#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/9/3 15:34
# @Author  : 希澈
# @File    : agent_memory.py
# @Software: PyCharm
import asyncio
import logging
import math
import os
import re
from datetime import datetime
from typing import Optional, List

import numpy as np
from pydantic import BaseModel

from oasis.social_agent.agent_cognition import AgentCognition

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings

log = logging.getLogger("social.agent")

# ── 去重阈值：insight 余弦相似度超过此值视为重复 ──────────────────────
INSIGHT_DEDUP_THRESHOLD = 0.88
# ── 动态阈值基准与增长系数 ──────────────────────────────────────────────
REFLECT_BASE = 100.0
REFLECT_K = 10.0  # 每 e^1 条记录额外增加的阈值
SHARED_MEMORY_COLLECTION_NAME = "agent_memory"
RECAGENT_SHORT_TERM_SIMILARITY_THRESHOLD = 0.78
RECAGENT_SHORT_TERM_PROMOTE_COUNT = 3
RECAGENT_FORGETTING_DECAY_RATE = 0.01


class _EmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def __call__(self, input: List[str]) -> List[List[float]]:
        return [
            self.model.encode(text, normalize_embeddings=False).tolist()
            for text in input
        ]


class Record(BaseModel):
    id: Optional[str] = None
    type: str #observation | insight | action
    post_id: Optional[int] = None
    author_id: Optional[int] = None
    author_name: Optional[str] = None
    content: str
    create_time: int
    access_time: int
    importance_score: Optional[int] = 1
    pointer_ids: Optional[str] = None#用于存储反思记忆中指向的相关记忆的id字符
    memory_layer: Optional[str] = None
    reinforcement_count: Optional[int] = 1


class ActionRecord(BaseModel):
    post_id: Optional[int] = None
    author_id: Optional[int] = None
    author_name: Optional[str] = None
    action_type: str
    action_content: Optional[str] = None
    action_reason: str


class Memory:
    _shared_collections = {}

    def __init__(self, agent_id:int, name:str, client: chromadb.Client,llm_model, embedding_model_config):
        self.agent_id = agent_id
        self.name = name

        self.accumulated_score = 0.0
        self._reflect_threshold_override: Optional[float] = None
        self.reflect_records_cache: list[Record] = []
        self.llm_model = llm_model
        self.embedding_model_config=embedding_model_config

        collection_key = (
            id(client),
            embedding_model_config["model_name_or_path"],
            embedding_model_config["device"],
        )
        if collection_key not in self._shared_collections:
            self._shared_collections[collection_key] = client.get_or_create_collection(
                name=SHARED_MEMORY_COLLECTION_NAME,
                embedding_function=SentenceTransformerEmbeddingFunction(
                    model_name=embedding_model_config["model_name_or_path"],
                    device=embedding_model_config["device"]
                ),
                metadata={
                    "description": "shared_agent_long_term_memory",
                }
            )
        self.collection = self._shared_collections[collection_key]
        self.record_id = self._get_next_record_id()
        self.lock = asyncio.Lock()

    def _get_next_record_id(self) -> int:
        try:
            results = self.collection.get(where={"agent_id": {"$eq": self.agent_id}})
        except Exception as e:
            log.warning(
                f"[Memory] Agent {self.agent_id} failed to restore record_id: {e}"
            )
            return 1

        max_record_id = 0
        for record_id in results.get("ids", []):
            match = re.fullmatch(rf"{self.agent_id}-id(\d+)", str(record_id))
            if match:
                max_record_id = max(max_record_id, int(match.group(1)))
        return max_record_id + 1

    def _agent_where(self, where: Optional[dict] = None) -> dict:
        agent_filter = {"agent_id": {"$eq": self.agent_id}}
        if where is None:
            return agent_filter
        if "$and" in where and isinstance(where["$and"], list):
            return {"$and": [agent_filter, *where["$and"]]}
        return {"$and": [agent_filter, where]}

    @staticmethod
    def _is_observation_type(record_type: Optional[str]) -> bool:
        return bool(record_type) and (
            record_type == "observation" or record_type.startswith("observation_")
        )

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        # Chroma's default distance for normalized sentence-transformer embeddings
        # is often close to L2. This bounded transform is stable across backends.
        return 1.0 / (1.0 + max(0.0, float(distance)))

    @staticmethod
    def _record_from_chroma_item(
            rec_id: str,
            doc: str,
            meta: dict,
    ) -> Record:
        return Record(
            id=rec_id,
            type=meta.get("type"),
            post_id=meta.get("post_id", None),
            author_id=meta.get("author_id", None),
            author_name=meta.get("author_name", None),
            content=doc if isinstance(doc, str) else str(doc),
            create_time=meta.get("create_time"),
            access_time=meta.get("access_time"),
            importance_score=meta.get("importance_score", 1),
            pointer_ids=meta.get("pointer_ids", None),
            memory_layer=meta.get("memory_layer", "long_term"),
            reinforcement_count=meta.get("reinforcement_count", 1),
        )

    # ── 动态反思阈值 ────────────────────────────────────────────
    @property
    def reflect_threshold(self) -> float:
        """
        随记忆条数自适应升高，防止后期频繁触发反思。
        公式：base + K * ln(1 + record_id)
        record_id=1   → 100 + 10*0.69 ≈ 107
        record_id=50  → 100 + 10*3.93 ≈ 139
        record_id=500 → 100 + 10*6.22 ≈ 162
        """
        if self._reflect_threshold_override is not None:
            return self._reflect_threshold_override
        return REFLECT_BASE + REFLECT_K * math.log1p(self.record_id)

    @reflect_threshold.setter
    def reflect_threshold(self, value: float):
        """兼容外部直接赋值的旧代码（设为手动覆盖值）。"""
        self._reflect_threshold_override = value

    async def add_record(self, record: Record):
        """
        整个方法加锁，消除并发竞态。
        insight 类型记录不进 reflect_records_cache。
        将记录写入内存并维护反思缓存与累计分数。
        若达到 reflect_threshold，则重置累计分数并返回用于反思的记录列表。
        否则将记录加入 reflect_records_cache 并返回 None。
        """
        async with self.lock:
            await self._add_to_memory_unsafe(record)
            self.accumulated_score += record.importance_score

            if self.accumulated_score > self.reflect_threshold:
                # 达到触发阈值：重置并返回待反思的缓存
                self.accumulated_score = 0.0
                reflect_records = self.reflect_records_cache.copy()
                self.reflect_records_cache = []
                return reflect_records  # 返回列表，供 Agent 处理反思逻辑
            else:
                if record.type == "action" or self._is_observation_type(record.type):
                    self.reflect_records_cache.append(record)
                return None

    async def add_record_no_reflection(self, record: Record):
        """
        公共写入接口（供外部直接调用，如写 insight）。
        加锁以保证线程安全。
        """
        async with self.lock:
            await self._add_to_memory_unsafe(record)

    async def _add_to_memory_unsafe(self, record: Record):
        """
        实际写入 ChromaDB，调用方需自行持有 self.lock。
        """
        ids = [f"{self.agent_id}-id{self.record_id}"]
        metadatas = [
            {
                "agent_id": self.agent_id,
                "type": record.type,
                **({"post_id": record.post_id} if record.post_id is not None else {}),
                **({"author_id": record.author_id} if record.author_id is not None else {}),
                **({"author_name": record.author_name} if record.author_name is not None else {}),
                "create_time": record.create_time,
                "access_time": record.access_time,
                "importance_score": record.importance_score,
                "memory_layer": record.memory_layer or "long_term",
                "reinforcement_count": record.reinforcement_count or 1,
                **({"pointer_ids": record.pointer_ids} if record.pointer_ids is not None else {}),
            }
        ]
        self.collection.add(
            ids=ids,
            documents=[record.content],
            metadatas=metadatas,
        )
        record.id = ids[0]
        self.record_id += 1

    # ── Insight 去重写入 ────────────────────────────────────────
    async def add_insight_if_novel(self, record: Record) -> bool:
        """
        写入 insight 前进行语义去重。
        若库中已存在余弦相似度 ≥ INSIGHT_DEDUP_THRESHOLD 的 insight，则跳过。
        返回 True 表示成功写入，False 表示因重复被跳过。

        使用方式（在 agent._handle_reflection 中替换 add_to_memory）：
            written = await self.memory.add_insight_if_novel(insight_record)
        """
        # 先查询最近邻 insight
        try:
            existing = self.collection.query(
                query_texts=[record.content],
                n_results=5,
                where=self._agent_where({"type": {"$eq": "insight"}}),
            )
            distances = existing.get("distances", [[]])[0]
            if distances:
                # ChromaDB 默认 L2 距离；转为余弦相似度近似：sim ≈ 1/(1+d)
                # 若嵌入已归一化则 d = 2*(1-cos)，cos = 1 - d/2 更精确
                min_dist = min(distances)
                cos_sim = 1.0 - min_dist / 2.0  # 适用于归一化向量
                if cos_sim >= INSIGHT_DEDUP_THRESHOLD:
                    log.debug(
                        f"[Memory] Agent {self.agent_id} skipped duplicate insight "
                        f"(cos_sim={cos_sim:.3f}): {record.content[:60]}..."
                    )
                    return False
        except Exception as e:
            # 去重失败时降级为直接写入，不阻断主流程
            log.warning(f"[Memory] Agent {self.agent_id} insight dedup query failed: {e}")

        await self.add_record_no_reflection(record)
        return True

    # ── 批量观察写入（配合 agent.observe 批量打分） ──────────────
    async def batch_add_observations(
            self,
            records: List[Record],
    ) -> list[tuple[list[Record], Record]]:
        """
        批量写入 observation 记录，配合 agent 侧批量 importance 评估使用。
        每条记录仍独立触发反思逻辑（通过 add_record）。

        agent.observe() 侧改造示例：
            # 1. 批量生成摘要和 importance
            items = await self.cognition_model.batch_summarize_items_with_importance(
                contents, agent_name, item_type, description)
            # 2. 批量构造 Record 并写入
            records = [Record(type="observation",
                              content=item["summary"],
                              importance_score=item["importance"], ...)
                       for item in items]
            await self.memory.batch_add_observations(records)
        """
        reflection_batches: list[tuple[list[Record], Record]] = []
        for record in records:
            reflect_records = await self.add_record(record)
            if reflect_records:
                reflection_batches.append((reflect_records, record))
        return reflection_batches

    def convert_results_to_records(
            self, results, current_time, token_budget: Optional[int] = None
    ) -> List[Record]:
        """
        将 ChromaDB 返回结果转换为 Record 列表，并按重要性/新近性/相似度排序。
        新增 token_budget 参数，超出预算即停止追加。
        """
        ids = results.get("ids")
        docs = results.get("documents")
        metas = results.get("metadatas")
        distances = results.get("distances", None)

        if ids and isinstance(ids[0], list):        ids = ids[0]
        if docs and isinstance(docs[0], list):      docs = docs[0]
        if metas and isinstance(metas[0], list):    metas = metas[0]
        if distances and isinstance(distances[0], list): distances = distances[0]

        if not docs or not metas:
            return []

        if distances is None:
            distances = [0.0] * len(docs)

        # ── 三维评分矩阵 ────────────────────────────────────────────────
        rating_matrix = [[], [], []]
        for doc, meta, distance in zip(docs, metas, distances):
            rating_matrix[0].append(meta.get("importance_score", 1))
            access_time = meta.get("access_time", 0)
            decay = (
                math.exp(-0.995 * (current_time - access_time))
                if current_time >= access_time
                else 0.0
            )
            rating_matrix[1].append(decay)
            sim = 1.0 / (1.0 + max(0.0, float(distance)))
            rating_matrix[2].append(sim)

        rating_matrix = np.array(rating_matrix)
        min_vals = rating_matrix.min(axis=1, keepdims=True)
        max_vals = rating_matrix.max(axis=1, keepdims=True)
        norm_matrix = (rating_matrix - min_vals) / (max_vals - min_vals + 1e-8)
        final_scores = norm_matrix[0] + norm_matrix[1] + norm_matrix[2]
        sorted_indices = np.argsort(-final_scores)

        # ── Token 预算控制 ───────────────────────────────────────────────
        effective_budget = token_budget if token_budget is not None else self.llm_model.max_memory_tokens
        current_tokens = 0
        records: List[Record] = []

        for idx in sorted_indices:
            doc = docs[idx]
            meta = metas[idx]
            rec_id = ids[idx]
            content = doc if isinstance(doc, str) else str(doc)

            content_tokens = self.llm_model.get_tokens_num(content)
            if current_tokens + content_tokens > effective_budget:
                break
            current_tokens += content_tokens

            records.append(self._record_from_chroma_item(rec_id, content, meta))
        return records

    def retrieve_recent_memory(
            self, current_time: int, since_time: Optional[int] = None, interval: int = 10
    ) -> List[Record]:
        """
        统一的近期记忆检索接口。
        - 若提供 since_time，检索 [since_time, current_time] 区间。
        - 否则使用固定 interval 窗口（用于 perform_action_by_llm 的 memory_content）。
        """
        lower = since_time if since_time is not None else max(0, current_time - interval)
        if lower > current_time:
            lower = 0

        where_clause = {
            "$and": [
                {"access_time": {"$gte": lower}},
                {"access_time": {"$lte": current_time}},
                {"type": {"$ne": "observation_post"}},
            ]
        }
        results = self.collection.get(where=self._agent_where(where_clause))
        return self.convert_results_to_records(results, current_time)

    def retrieve_recent_memory_by_time(self, login_times: List[int]) -> List[Record]:
        """
        按登录时间区间检索，内部委托给统一接口。
        保持对外签名不变，兼容 agent.py 中的调用。
        """
        current_time = login_times[-1]
        previous_time = login_times[-2] if len(login_times) >= 2 else 0
        return self.retrieve_recent_memory(current_time, since_time=previous_time)

    def retrieve_memory_by_content(self, query: str, current_time) -> List[Record]:
        post_related_results = self.collection.query(
            query_texts=[query],
            where=self._agent_where(),
        )
        ids = post_related_results.get('ids', [])[0]
        metas = post_related_results.get('metadatas', [])[0]

        post_related_records = self.convert_results_to_records(post_related_results, current_time)
        if len(ids)!=0:
            self.collection.update(
                ids=ids,
                metadatas=[{**meta, "access_time": current_time} for meta in metas],
            )
        return post_related_records

    def _touch_record_ids(self, ids: list[str], current_time: int) -> None:
        if not ids:
            return
        try:
            results = self.collection.get(ids=ids)
            metas = results.get("metadatas", [])
            if metas:
                self.collection.update(
                    ids=ids,
                    metadatas=[{**m, "access_time": current_time} for m in metas],
                )
        except Exception as e:
            log.warning(
                f"[Memory] Agent {self.agent_id} failed to update access_time: {e}"
            )

    async def write_recagent_observation(
            self,
            record: Record,
            *,
            similarity_threshold: float = RECAGENT_SHORT_TERM_SIMILARITY_THRESHOLD,
            promote_threshold: int = RECAGENT_SHORT_TERM_PROMOTE_COUNT,
            keep_promoted_short_term: bool = False,
    ) -> dict:
        """
        RecAgent 风格的记忆写入：sensory memory -> short-term memory -> long-term memory。

        这个方法只负责“记忆层级转换”本身，不负责原始感知内容的预处理。调用方需要提前完成：
        1. 将原始帖子/通知等观察压缩成 `record.content`；
        2. 根据内容、用户画像或上下文给出 `record.importance_score`；
        3. 填好 `create_time`、`access_time` 等时间字段。

        写入逻辑：
        - 新观察先被视为 sensory memory 输入；
        - 如果它与已有 short-term memory 足够相似，就不新增重复记录，而是强化已有短期记忆；
        - 强化次数达到 `promote_threshold` 后，该短期记忆会晋升为 long-term memory；
        - 如果没有相似短期记忆，则把新观察存入 short-term memory。

        返回值中的 `status` 用于告诉调用方本次写入发生了什么：
        - `stored_short_term`：新增了一条短期记忆；
        - `reinforced_short_term`：强化了已有短期记忆；
        - `promoted_to_long_term`：已有短期记忆被提升为长期记忆。
        """
        current_time = record.access_time

        # 只在当前 agent 自己的 short-term memory 中寻找最相似的一条记录。
        # RecAgent 论文里的短期记忆会被反复激活和强化，因此这里先不查长期记忆，
        # 避免把已经稳定沉淀的 long-term memory 又当作可强化的短期片段。
        query_results = self.collection.query(
            query_texts=[record.content],
            n_results=1,
            where=self._agent_where({"memory_layer": {"$eq": "short_term"}}),
        )

        ids = query_results.get("ids", [[]])[0]
        docs = query_results.get("documents", [[]])[0]
        metas = query_results.get("metadatas", [[]])[0]
        distances = query_results.get("distances", [[]])[0]

        if ids and docs and metas and distances:
            # Chroma 返回的是距离，业务上更容易按“相似度”理解，所以统一转换成 similarity。
            # 只有超过阈值的记录才视作同一类短期经验，否则保留为新的短期记忆。
            similarity = self._distance_to_similarity(distances[0])
            if similarity >= similarity_threshold:
                short_id = ids[0]
                short_doc = docs[0]
                short_meta = metas[0]

                # reinforcement_count 表示这类短期经验被重复遇到/激活的次数。
                # importance_score 取较大值，避免后来的高重要性观察被旧记录的低分覆盖。
                reinforcement_count = int(
                    short_meta.get("reinforcement_count", 1)
                ) + 1
                updated_meta = {
                    **short_meta,
                    "access_time": current_time,
                    "importance_score": max(
                        int(short_meta.get("importance_score", 1)),
                        int(record.importance_score or 1),
                    ),
                    "reinforcement_count": reinforcement_count,
                }

                if reinforcement_count >= promote_threshold:
                    # 短期记忆被反复强化后，认为它已经具有稳定行为参考价值，
                    # 因此转换为长期记忆。这里沿用原短期记忆的 document 和 metadata，
                    # 只修改 memory_layer、reinforcement_count 和 access_time。
                    promoted_record = self._record_from_chroma_item(
                        short_id, short_doc, updated_meta
                    )
                    promoted_record.memory_layer = "long_term"
                    promoted_record.reinforcement_count = reinforcement_count
                    promoted_record.access_time = current_time

                    # 默认删除原 short-term 记录，避免同一内容同时出现在短期和长期记忆中。
                    # 如果实验需要保留短期痕迹，可通过 keep_promoted_short_term=True 保留。
                    if keep_promoted_short_term:
                        self.collection.update(
                            ids=[short_id],
                            metadatas=[updated_meta],
                        )
                    else:
                        self.collection.delete(ids=[short_id])

                    # add_record 会按普通长期记忆写入流程保存，并可能触发反思/洞察生成。
                    reflect_records = await self.add_record(promoted_record)
                    return {
                        "status": "promoted_to_long_term",
                        "record_id": promoted_record.id,
                        "similarity": similarity,
                        "reinforcement_count": reinforcement_count,
                        "reflect_records": reflect_records,
                    }

                # 尚未达到晋升阈值，只刷新短期记忆的访问时间、重要性和强化次数。
                self.collection.update(ids=[short_id], metadatas=[updated_meta])
                return {
                    "status": "reinforced_short_term",
                    "record_id": short_id,
                    "similarity": similarity,
                    "reinforcement_count": reinforcement_count,
                    "reflect_records": None,
                }

        # 没有足够相似的短期记忆：把这次 sensory 输入转成新的 short-term memory。
        # 使用 add_record_no_reflection，避免每条短期观察都触发长期反思，控制噪声和成本。
        record.memory_layer = "short_term"
        record.reinforcement_count = 1
        await self.add_record_no_reflection(record)
        return {
            "status": "stored_short_term",
            "record_id": record.id,
            "similarity": None,
            "reinforcement_count": 1,
            "reflect_records": None,
        }

    def retrieve_recagent_memory(
            self,
            query: str,
            current_time: int,
            *,
            long_term_k: int = 8,
            short_term_token_budget: Optional[int] = None,
            long_term_token_budget: Optional[int] = None,
            include_insights: bool = True,
    ) -> dict[str, List[Record]]:
        """
        RecAgent 风格的记忆读取。

        读取结果分为两层：
        1. `short_term`：当前仍处于短期层的近期/新鲜经验。
           这些记录不做语义检索排序，而是整体取出后交给 `convert_results_to_records`
           按时间和 token budget 处理，因为短期记忆强调“最近经历”和“当前上下文残留”。
        2. `long_term`：与当前 query 语义相关的长期经验和可选 insight。
           这些记录通过向量检索获得，用来提供稳定偏好、历史行为和反思结论。

        参数说明：
        - `query`：通常由当前可见帖子、通知内容拼接而成；
        - `current_time`：用于刷新被读取记忆的 access_time，并参与 token/时间相关处理；
        - `long_term_k`：最终返回的长期记忆条数上限；
        - `short_term_token_budget` / `long_term_token_budget`：分别控制两层记忆文本预算；
        - `include_insights`：是否允许把 reflection/insight 类型的长期记忆返回给 agent。

        注意：这里仍然使用当前 Chroma 向量库作为统一后端，只是通过 metadata 中的
        `memory_layer` 字段模拟 RecAgent 的分层记忆机制。
        """
        # 短期记忆直接按 layer 全量取出。它们代表最近看到、尚未沉淀的内容，
        # 即使和本轮 query 的语义不完全相似，也可能影响用户此刻的行动。
        short_results = self.collection.get(
            where=self._agent_where({"memory_layer": {"$eq": "short_term"}}),
        )
        short_records = self.convert_results_to_records(
            short_results,
            current_time,
            token_budget=short_term_token_budget,
        )

        # 长期记忆需要语义相关性检索。这里多取一些候选，再在后面过滤掉短期记忆、
        # insight 开关不允许的记录，并截断到 long_term_k。
        # 多取候选是为了抵消过滤造成的数量损失。
        n_results = max(long_term_k * 5, long_term_k + len(short_records), 20)
        long_results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=self._agent_where(),
        )
        long_records = self.convert_results_to_records(
            long_results,
            current_time,
            token_budget=long_term_token_budget,
        )
        # collection.query 的 where 只限制 agent_id，没有排除 short-term。
        # 因此这里再按 memory_layer 过滤，保证 short_term 和 long_term 两个返回桶不重复。
        long_records = [
            r for r in long_records if (r.memory_layer or "long_term") != "short_term"
        ]
        # insight 是由反思流程产生的稳定总结。某些消融实验只想读原始经验时，
        # 可以通过 include_insights=False 关闭这类记录。
        if not include_insights:
            long_records = [r for r in long_records if r.type != "insight"]
        long_records = long_records[:long_term_k]

        # 被读出的记忆视为“本轮被访问”，刷新 access_time。
        # 后续遗忘机制会根据 access_time 计算 age，因此这里会影响长期记忆保留概率。
        touched_ids = [r.id for r in short_records + long_records if r.id]
        self._touch_record_ids(touched_ids, current_time)
        return {
            "short_term": short_records,
            "long_term": long_records,
        }

    def forget_low_value_long_term(
            self,
            current_time: int,
            *,
            retention_threshold: float = 0.1,
            decay_rate: float = RECAGENT_FORGETTING_DECAY_RATE,
            max_records: Optional[int] = None,
            dry_run: bool = False,
    ) -> list[str]:
        """
        RecAgent 风格的长期记忆遗忘机制。

        该方法只处理 long-term memory，不删除 short-term memory。短期记忆是否晋升或保留，
        由 `write_recagent_observation` 中的强化/晋升逻辑负责。

        保留分数计算公式：
            retention = importance * exp(-decay_rate * age)

        含义：
        - `importance`：记忆本身的重要性分数，越重要越不容易忘；
        - `age`：距离上次访问 `access_time` 过去了多少时间步，越久未访问越容易忘；
        - `decay_rate`：遗忘衰减速度，越大表示随时间遗忘得越快；
        - `retention_threshold`：保留分数低于该阈值时，记忆会成为删除候选。

        参数说明：
        - `max_records`：本轮最多删除多少条，避免一次性删除过多长期记忆；
        - `dry_run`：只返回将会删除的 record id，不真正删除，便于调试和消融实验。

        返回值是本轮被删除或 dry-run 下将被删除的 record id 列表。
        """
        # 取出当前 agent 的全部记忆。这里不直接在 where 里筛 long-term，
        # 是为了兼容旧数据：旧数据可能没有 memory_layer 字段，读取时应当视作长期记忆。
        results = self.collection.get(where=self._agent_where())
        ids = results.get("ids", [])
        metas = results.get("metadatas", [])
        candidates: list[tuple[float, str]] = []

        for rec_id, meta in zip(ids, metas):
            # RecAgent 的短期记忆表示仍在工作区中的新鲜经验，
            # 不参与长期遗忘；它们只会被强化、晋升，或继续留在短期层。
            if meta.get("memory_layer") == "short_term":
                continue

            # age 基于 access_time 而不是 create_time：
            # 一条老记忆只要经常被读取，就说明仍有参考价值，不应因为创建早而被忘掉。
            age = max(0, current_time - int(meta.get("access_time", current_time)))
            importance = int(meta.get("importance_score", 1))
            retention = importance * math.exp(-decay_rate * age)

            # 保留分数低于阈值的长期记忆进入候选池。
            # 候选池中暂时不删除，后面会按 retention 从低到高排序再统一处理。
            if retention < retention_threshold:
                candidates.append((retention, rec_id))

        # 优先删除 retention 最低、最不值得保留的记忆。
        candidates.sort(key=lambda item: item[0])
        delete_ids = [rec_id for _, rec_id in candidates]
        if max_records is not None:
            # 控制单轮遗忘规模，避免一次运行时记忆库变化过大。
            delete_ids = delete_ids[:max_records]
        if delete_ids and not dry_run:
            # dry_run=True 时只返回候选 id，不执行真正删除。
            self.collection.delete(ids=delete_ids)
        return delete_ids

    # ──  更新 observation：同步刷新内容 + 时间 ───────────────────
    def update_observation_record(
            self, post_id: int, current_time: int, new_content: Optional[str] = None
    ) -> bool:
        """
        根据 post_id 查找并更新已存储的 observation 记忆。

        这个方法用于普通观察记忆的“去重刷新”：
        - 如果某个帖子之前已经被观察并写入记忆，再次看到它时不应重复新增一条；
        - 而是刷新它的 `access_time`，表示这条观察在当前时间步又被访问；
        - 如果调用方提供了新的摘要 `new_content`，还要同步刷新 Chroma 里的 document 内容。

        两种更新模式：
        1. `new_content is None`
           只更新 metadata 中的 `access_time`。这是旧版调用路径，保留向后兼容。
        2. `new_content is not None`
           同时更新 document 和 metadata。由于 ChromaDB 的 `update()` 在当前使用方式下
           不能可靠修改 document，这里采用 delete -> add，并保持原 record id 不变。

        返回值：
        - `True`：找到至少一条对应 observation，并完成刷新；
        - `False`：没有找到对应 observation，调用方可以继续新增记录。

        agent.observe() 侧调用示例：
            already_seen = self.memory.update_observation_record(
                post_id, self.env.current_time, new_content=new_summary
            )
        """
        # 先按 post_id 找到当前 agent 记忆库中与该帖子相关的所有记录。
        # 同一个 post_id 可能对应 observation、action 或其他类型记录，
        # 因此下一步还需要按 observation 类型再筛一次。
        results = self.collection.get(
            where=self._agent_where({"post_id": {"$eq": post_id}})
        )
        all_ids = results.get("ids", [])
        all_metas = results.get("metadatas", [])

        # 只更新 observation 类型记录，避免误改同一 post_id 上的 action/insight 等记忆。
        selected = [
            (rec_id, meta)
            for rec_id, meta in zip(all_ids, all_metas)
            if self._is_observation_type(meta.get("type"))
        ]
        ids = [rec_id for rec_id, _ in selected]
        metas = [meta for _, meta in selected]

        if not ids:
            # 没有观察记录说明这是第一次看到该帖子，交给调用方新增。
            return False

        if new_content is None:
            # 仅刷新 access_time（向后兼容旧调用）
            self.collection.update(
                ids=ids,
                metadatas=[{**m, "access_time": current_time} for m in metas],
            )
        else:
            # 内容有变化：delete → re-add（保持同一 id）
            # 这里保留原 metadata，只更新 access_time；record id 也保持不变，
            # 这样外部通过 pointer_ids 等字段引用该记录时不会失效。
            updated_metas = [{**m, "access_time": current_time} for m in metas]
            self.collection.delete(ids=ids)
            self.collection.add(
                ids=ids,
                documents=[new_content] * len(ids),
                metadatas=updated_metas,
            )
            log.debug(
                f"[Memory] Agent {self.agent_id} refreshed observation "
                f"post_id={post_id} content."
            )
        return True

    def get_observation_summary_by_post_id(self, post_id: int) -> Optional[str]:
        """
        Return the stored observation document for a post id.

        The post id remains metadata for lookup, while callers can use the
        returned summary to build human-readable action memories.
        """
        results = self.collection.get(
            where=self._agent_where({"post_id": {"$eq": post_id}})
        )
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        candidates = [
            doc
            for doc, meta in zip(docs, metas)
            if self._is_observation_type(meta.get("type"))
            and isinstance(doc, str)
            and doc.strip()
        ]
        if not candidates:
            return None
        return candidates[0]
