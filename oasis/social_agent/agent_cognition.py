from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from string import Template
from typing import Any

from oasis.localization import is_zh_locale
try:
    from oasis.social_agent.llm_model import LLMAbortError
except ModuleNotFoundError:
    class LLMAbortError(RuntimeError):
        pass
from oasis.social_agent.social_context_builder import SocialContextBuilder


log_dir = "/data/lijiantong/Data/oasis/log/"
now = datetime.now().strftime("%Y-%m-%d")
AGENT_COGNITION_LOG_PATH = os.getenv(
    "OASIS_AGENT_COGNITION_RECORD_PATH",
    os.path.join(log_dir, f"agent_cognition_records_{now}.jsonl"),
)
os.makedirs(os.path.dirname(AGENT_COGNITION_LOG_PATH), exist_ok=True)

BATCH_OBSERVATION_EVAL_CHUNK_SIZE = 5
BATCH_IMPORTANCE_EVAL_CHUNK_SIZE = 5
DEFAULT_IMPORTANCE_RATING = 5
FALLBACK_SUMMARY_MAX_CHARS = 500

logger = logging.getLogger("social.agent_cognition")


def _iter_chunks(items: list[Any], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


class AgentCognition:
    """Small observation helper used by the profile-only simulator."""

    batch_observation_eval_prompt = Template(
        """
You are helping $name process multiple social-media $item_type_plural.
The user's profile and description are:
$description

For each item, write a 1-2 sentence factual summary: who is involved,
what happened, or what claim is expressed. Do not infer platform metadata.
Also rate each item's memory importance for this user from 1 to 10.

Return ONLY a valid JSON object in this exact format:
{
  "items": [
    {"summary": "summary for item 0", "importance": 5},
    {"summary": "summary for item 1", "importance": 5}
  ]
}
The list must contain exactly $num_items objects in the same order as input.

Items:
$items_content
        """
    )

    batch_importance_eval_prompt = Template(
        """
$description

Rate the significance of each memory item for the person described above
on a scale of 1 to 10:
- 1 = mundane
- 10 = highly meaningful

Return ONLY a valid JSON object in this exact format:
{
  "ratings": [<int>, <int>, ...]
}
The list must have exactly $num_items integers in the same order as input.

Memory items:
$memory_items
        """
    )

    zh_prompts = {
        "batch_observation_eval_prompt": Template(
            """
你正在帮助 $name 处理多条社交媒体$item_type_zh。
该用户的画像和描述如下：
$description

请为每条内容写 1 到 2 句事实摘要：涉及谁、发生了什么或表达了什么主张。
不要推断平台元数据。同时请结合该用户画像，为每条内容给出记忆重要性分数，范围为 1 到 10。

请只返回一个合法 JSON 对象，格式严格如下，不要添加额外文本：
{
  "items": [
    {"summary": "第 0 条内容的摘要", "importance": 5},
    {"summary": "第 1 条内容的摘要", "importance": 5}
  ]
}
列表必须恰好包含 $num_items 个对象，并与输入内容顺序一致。

内容：
$items_content
            """
        ),
        "batch_importance_eval_prompt": Template(
            """
$description

请为上面描述的人评估多条记忆的重要程度。
对每一项记忆按 1 到 10 分打分：
- 1 = 非常日常
- 10 = 非常重要

请只返回一个合法 JSON 对象，格式严格如下，不要添加额外文本：
{
  "ratings": [<int>, <int>]
}
列表必须恰好包含 $num_items 个整数，并与输入记忆顺序一致。

记忆项：
$memory_items
            """
        ),
    }

    def __init__(self, agent_id: int, model, user_info):
        self.agent_id = agent_id
        self.model = model
        self.current_time = 0
        self.locale = getattr(user_info, "locale", "en")
        if is_zh_locale(self.locale):
            for attr_name, prompt in self.zh_prompts.items():
                setattr(self, attr_name, prompt)
        self.cognition_records: list[dict[str, Any]] = []

    def convert_to_record(
        self,
        tag: str,
        is_success: bool,
        user_prompt: str,
        response: Any,
    ) -> None:
        self.cognition_records.append({
            "tag": tag,
            "is_success": is_success,
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ],
        })

    async def save_cognition_records(self) -> None:
        if not self.cognition_records:
            return
        record = {
            "agent_id": self.agent_id,
            "timestamp": self.current_time,
            "messages": self.cognition_records,
        }
        try:
            with open(AGENT_COGNITION_LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.cognition_records = []
        except LLMAbortError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Failed to save agent cognition records. Error: {exc}"
            ) from exc

    async def batch_importance_evaluation(
        self,
        description: str,
        memory_contents: list[str],
    ) -> list[int]:
        if not memory_contents:
            return []

        results: list[int] = []
        for chunk in _iter_chunks(memory_contents, BATCH_IMPORTANCE_EVAL_CHUNK_SIZE):
            results.extend(await self._batch_importance_evaluation_chunk(
                description,
                chunk,
            ))
        return results

    async def _batch_importance_evaluation_chunk(
        self,
        description: str,
        memory_contents: list[str],
    ) -> list[int]:
        numbered = "\n".join(
            f"{index}. {content}"
            for index, content in enumerate(memory_contents)
        )
        user_prompt = self.batch_importance_eval_prompt.substitute(
            description=description,
            num_items=len(memory_contents),
            memory_items=numbered,
        )
        try:
            response = await self.model.get_json_response(user_prompt)
            ratings = response.get("ratings", [])
            if not isinstance(ratings, list):
                ratings = []
            result = [
                self._normalize_importance(rating)
                for rating in ratings[:len(memory_contents)]
            ]
            result.extend(
                [DEFAULT_IMPORTANCE_RATING] * (len(memory_contents) - len(result))
            )
            self.convert_to_record(
                "batch_importance_evaluation",
                True,
                user_prompt,
                json.dumps({"ratings": result}, ensure_ascii=False),
            )
            return result
        except LLMAbortError:
            raise
        except Exception as exc:
            self.convert_to_record(
                "batch_importance_evaluation",
                False,
                user_prompt,
                str(exc),
            )
            logger.warning(
                "Agent %s importance evaluation failed; using defaults: %s",
                self.agent_id,
                exc,
            )
            return [DEFAULT_IMPORTANCE_RATING] * len(memory_contents)

    async def batch_summarize_items_with_importance(
        self,
        item_contents: list[str],
        agent_name: str,
        item_type: str = "item",
        description: str = "",
    ) -> list[dict[str, Any]]:
        if not item_contents:
            return []

        results: list[dict[str, Any]] = []
        for chunk in _iter_chunks(
            item_contents,
            BATCH_OBSERVATION_EVAL_CHUNK_SIZE,
        ):
            results.extend(
                await self._batch_summarize_items_with_importance_chunk(
                    chunk,
                    agent_name,
                    item_type,
                    description,
                )
            )
        return results

    async def _batch_summarize_items_with_importance_chunk(
        self,
        item_contents: list[str],
        agent_name: str,
        item_type: str = "item",
        description: str = "",
    ) -> list[dict[str, Any]]:
        item_type_en, item_type_plural, item_type_zh, numbered = (
            SocialContextBuilder.format_batch_summary_items(
                locale=self.locale,
                item_contents=item_contents,
                item_type=item_type,
            )
        )
        user_prompt = self.batch_observation_eval_prompt.substitute(
            name=agent_name,
            description=description or "No profile or description provided.",
            item_type_plural=item_type_plural,
            item_type_zh=item_type_zh,
            num_items=len(item_contents),
            items_content=numbered,
        )
        try:
            response = await self.model.get_json_response(user_prompt)
            raw_items = response.get("items", [])
            if not isinstance(raw_items, list):
                raw_items = []
            result = [
                self._observation_result(raw_items, index, item_content)
                for index, item_content in enumerate(item_contents)
            ]
            self.convert_to_record(
                "batch_summarize_items_with_importance",
                True,
                user_prompt,
                json.dumps({"items": result}, ensure_ascii=False),
            )
            return result
        except LLMAbortError:
            raise
        except Exception as exc:
            self.convert_to_record(
                "batch_summarize_items_with_importance",
                False,
                user_prompt,
                str(exc),
            )
            logger.warning(
                "Agent %s observation summary failed; using raw text: %s",
                self.agent_id,
                exc,
            )
            return [
                {
                    "summary": str(item_content).strip()[:FALLBACK_SUMMARY_MAX_CHARS],
                    "importance": DEFAULT_IMPORTANCE_RATING,
                }
                for item_content in item_contents
            ]

    @staticmethod
    def _normalize_importance(value: Any) -> int:
        try:
            numeric = int(float(value))
        except (TypeError, ValueError):
            numeric = DEFAULT_IMPORTANCE_RATING
        return max(1, min(10, numeric))

    @classmethod
    def _observation_result(
        cls,
        raw_items: list[Any],
        index: int,
        fallback_content: str,
    ) -> dict[str, Any]:
        raw_item = raw_items[index] if index < len(raw_items) else {}
        if isinstance(raw_item, dict):
            summary = str(raw_item.get("summary") or "").strip()
            importance = cls._normalize_importance(raw_item.get("importance"))
        else:
            summary = ""
            importance = DEFAULT_IMPORTANCE_RATING
        if not summary:
            summary = str(fallback_content).strip()[:FALLBACK_SUMMARY_MAX_CHARS]
        return {"summary": summary, "importance": importance}
