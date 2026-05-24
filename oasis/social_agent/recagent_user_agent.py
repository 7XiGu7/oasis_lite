from __future__ import annotations

import json
from typing import Any

import chromadb

from oasis.localization import is_zh_locale
from oasis.social_agent.action_importance import heuristic_action_importance
from oasis.social_agent.agent_memory import Record
from oasis.social_agent.baseline_agent_utils import (
    format_available_actions,
    format_notifications,
    format_observation_context,
    format_records,
    initialize_baseline_agent,
    other_info,
    profile_value,
)
from oasis.social_agent.social_context_builder import SocialContextBuilder
from oasis.social_platform import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.typing import ActionType


class RecAgentUserAgent:
    """Profile-memory-action user simulation baseline with layered memory."""

    BEHAVIORAL_FEATURES = ("Watcher", "Explorer", "Critic", "Chatter", "Poster")
    BEHAVIORAL_FEATURE_ALIASES = {
        feature.lower(): feature for feature in BEHAVIORAL_FEATURES
    }
    BEHAVIORAL_FEATURE_CONFLICTS = {
        "Watcher": {"Chatter", "Poster"},
        "Chatter": {"Watcher"},
        "Poster": {"Watcher"},
    }
    BEHAVIORAL_FEATURE_DESCRIPTIONS = {
        "zh": {
            "Watcher": "Watcher：偏向先阅读和观察，对可见帖子用点赞、评论、转发、引用或关注等轻量反馈表达看法，较少主动发起全新内容。",
            "Explorer": "Explorer：会主动浏览不同推荐内容、寻找新作者或新信息，愿意沿着推荐流和讨论线索探索更多内容。",
            "Critic": "Critic：对内容质量、事实依据和观点逻辑标准较高，容易指出问题、质疑不严谨表达或给出批判性评论。",
            "Chatter": "Chatter：重视社交互动，倾向回复通知、参与评论区对话，并容易受到熟人或互动对象的观点影响。",
            "Poster": "Poster：表达和分享欲较强，倾向主动发布、转发或引用帖子，把自己的看法公开给更多人。",
        },
        "en": {
            "Watcher": "Watcher: tends to read and observe first, then gives lightweight feedback such as likes, comments, reposts, quotes, or follows instead of starting new posts.",
            "Explorer": "Explorer: actively browses recommendations, seeks new authors or information, and follows recommendation or discussion trails for more content.",
            "Critic": "Critic: holds high standards for content quality, evidence, and logic, and is likely to question weak claims or write critical comments.",
            "Chatter": "Chatter: values social interaction, tends to reply to notifications and join comment threads, and is influenced by familiar or interactive accounts.",
            "Poster": "Poster: has a strong drive to express and share, tending to create, repost, or quote posts publicly.",
        },
    }

    def __init__(
        self,
        agent_id: int,
        user_info: UserInfo,
        model: Any,
        embedding_model_config: dict,
        client: chromadb.Client,
        channel: Channel | None = None,
        available_actions: list[ActionType] | None = None,
        max_content_length: int = 280,
        enable_thinking: bool | None = False,
        forgetting_interval: int = 24,
    ):
        initialize_baseline_agent(
            self,
            agent_id=agent_id,
            user_info=user_info,
            model=model,
            embedding_model_config=embedding_model_config,
            client=client,
            channel=channel,
            available_actions=available_actions,
            max_content_length=max_content_length,
            enable_thinking=enable_thinking,
        )
        self.forgetting_interval = forgetting_interval

    def _behavioral_features(self) -> list[str]:
        info = other_info(self)
        source_features = self._source_behavioral_features(info.get("behavioral_features"))
        if source_features:
            return self._merge_compatible_features(source_features)

        traits = info.get("traits") if isinstance(info.get("traits"), dict) else {}
        description = " ".join(
            str(value)
            for value in (info.get("traits_info"), self.description)
            if value
        ).lower()
        openness = self._trait_score(traits, "openness")
        conscientiousness = self._trait_score(traits, "conscientiousness")
        extroversion = self._trait_score(traits, "extroversion")
        agreeableness = self._trait_score(traits, "agreeableness")
        neuroticism = self._trait_score(traits, "neuroticism")

        features = []

        if openness is not None and openness >= 0.6:
            features.append("Explorer")
        if (
            (agreeableness is not None and agreeableness <= 0.4)
            or (conscientiousness is not None and conscientiousness >= 0.65)
            or (neuroticism is not None and neuroticism >= 0.65)
        ):
            features.append("Critic")
        if (
            extroversion is not None
            and extroversion >= 0.6
            and (agreeableness is None or agreeableness >= 0.4)
        ):
            features.append("Chatter")
        if (
            extroversion is not None
            and extroversion >= 0.5
            and (openness is None or openness >= 0.45)
        ):
            features.append("Poster")
        if extroversion is not None and extroversion <= 0.4:
            features.append("Watcher")

        if not features:
            features.append("Watcher")

        text_feature_keywords = [
            ("Explorer", ("curious", "novel", "explore", "creative", "好奇", "探索", "求新", "创新")),
            ("Critic", ("critical", "skeptical", "strict", "批判", "质疑", "谨慎", "高标准")),
            ("Chatter", ("talkative", "social", "互动", "健谈", "社交")),
            ("Poster", ("expressive", "active", "share", "表达", "主动", "分享", "发布")),
            ("Watcher", ("observe", "quiet", "reserved", "观察", "浏览", "内向", "沉稳")),
        ]
        for feature, keywords in text_feature_keywords:
            if feature not in features and any(keyword in description for keyword in keywords):
                features.append(feature)
        return self._merge_compatible_features(features or ["Watcher"])

    @classmethod
    def _source_behavioral_features(cls, value: Any) -> list[str]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            return [cls._normalize_behavioral_feature(value)]
        if isinstance(value, (list, tuple, set)):
            return [
                cls._normalize_behavioral_feature(str(item))
                for item in value
                if item not in (None, "")
            ]
        return [cls._normalize_behavioral_feature(str(value))]

    @classmethod
    def _normalize_behavioral_feature(cls, value: str) -> str:
        stripped = value.strip()
        return cls.BEHAVIORAL_FEATURE_ALIASES.get(stripped.lower(), stripped)

    @classmethod
    def _merge_compatible_features(cls, features: list[str]) -> list[str]:
        merged = []
        for feature in features:
            if feature in merged:
                continue
            if any(cls._features_conflict(feature, existing) for existing in merged):
                continue
            merged.append(feature)
        return merged or ["Watcher"]

    @classmethod
    def _features_conflict(cls, first: str, second: str) -> bool:
        return (
            second in cls.BEHAVIORAL_FEATURE_CONFLICTS.get(first, set())
            or first in cls.BEHAVIORAL_FEATURE_CONFLICTS.get(second, set())
        )

    def _behavioral_feature_descriptions(self) -> list[str]:
        locale_key = "zh" if is_zh_locale(self.locale) else "en"
        descriptions = self.BEHAVIORAL_FEATURE_DESCRIPTIONS[locale_key]
        return [
            descriptions.get(feature, feature)
            for feature in self._behavioral_features()
        ]

    @staticmethod
    def _trait_score(traits: dict[str, Any], key: str) -> float | None:
        value = traits.get(key)
        if value is None:
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, score))

    def _profile_information(self) -> str:
        values = {
            "name": profile_value(self, "name"),
            "gender": profile_value(self, "gender"),
            "profile": other_info(self).get("user_profile") or self.description,
            "traits": other_info(self).get("traits_info") or other_info(self).get("traits"),
            "behavioral_features": self._behavioral_feature_descriptions(),
        }
        if is_zh_locale(self.locale):
            labels = {
                "name": "姓名",
                "gender": "性别",
                "profile": "用户画像",
                "traits": "人格特征",
                "behavioral_features": "行为特征",
            }
            return "\n".join(
                f"{labels[key]}：{json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}"
                for key, value in values.items()
                if value not in (None, "")
            )
        labels = {
            "name": "Name",
            "gender": "Gender",
            "profile": "User profile",
            "traits": "Personality traits",
            "behavioral_features": "Behavioral features",
        }
        return "\n".join(
            f"{labels[key]}: {json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}"
            for key, value in values.items()
            if value not in (None, "")
        )

    def _build_messages(self) -> list[dict[str, str]]:
        query = "\n".join(
            [str(post.get("content", "")) for post in (self.env.posts or [])]
            + [str(item.get("content", "")) for item in (self.env.notifications or [])]
        )
        memory = self.memory.retrieve_recagent_memory(query, self.env.current_time)
        short_empty = "没有短期记忆。" if is_zh_locale(self.locale) else "No short-term memory."
        long_empty = "没有相关长期记忆。" if is_zh_locale(self.locale) else "No relevant long-term memory."
        short_memory = format_records(memory["short_term"], empty_text=short_empty)
        long_memory = format_records(memory["long_term"], empty_text=long_empty)
        context = format_observation_context(self)
        notifications = format_notifications(self)

        if is_zh_locale(self.locale):
            system_prompt = (
                "你正在使用一个带推荐流和社交互动的社交媒体平台。请以这个用户的身份，"
                "根据自己的画像、短期记忆、长期记忆和当前上下文，选择下一步平台行动。"
                "最终必须只调用一个可用工具，content 参数必须使用中文。"
            )
            user_prompt = f"""
[用户资料]
{self._profile_information()}

[记忆信息]
短期记忆：
{short_memory}

相关长期记忆：
{long_memory}

[本轮任务]
你现在打开了平台。请浏览当前推荐和社交通知，并选择一个符合自己画像、行为特征和记忆状态的下一步行动。

[当前上下文]
当前帖子：
{context}

通知：
{notifications}

[可用行动]
{format_available_actions(self)}

请调用且只调用一个可用工具。工具参数只能来自当前上下文中的可见对象；如需生成自然语言内容，应综合自己的画像、行为特征、记忆状态、当前帖子和通知来表达。
"""
        else:
            system_prompt = (
                "You are using a social media platform with recommendations and "
                "social interactions as this user. Choose your next platform action "
                "from your profile, short-term memory, long-term memory, and current "
                "context, then call exactly one available tool."
            )
            user_prompt = f"""
[Profile Information]
{self._profile_information()}

[Memory Information]
Short-term memory:
{short_memory}

Relevant long-term memory:
{long_memory}

[Instruction]
You have opened the platform now. Browse the current recommendations and
notifications, then choose one next action consistent with your profile,
behavioral features, and memory state.

[Context]
Current posts:
{context}

Notifications:
{notifications}

[Available Actions]
{format_available_actions(self)}

Call exactly one available tool. Tool arguments must come from visible context objects
or, when natural-language content is needed, be generated from your profile,
behavioral features, memory state, current posts, and notifications.
"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def perform_action_by_llm(self):
        await self.runtime.prepare_environment()
        await self.runtime.observe_recagent()
        if self.forgetting_interval > 0 and self.env.current_time % self.forgetting_interval == 0:
            self.memory.forget_low_value_long_term(
                self.env.current_time,
                max_records=10,
            )
        messages = self._build_messages()
        assistant_output, reasoning = await self.runtime.request_tool_response(messages)
        messages = await self.runtime.run_tool_feedback_loop(
            messages,
            assistant_output,
            variant="recagent_user",
            reasoning_content=reasoning,
        )
        await self.runtime.finalize(messages, variant="recagent_user")

    async def perform_action_by_data(self, func_name, *args, **kwargs) -> Any:
        return await self.runtime.perform_action_by_data(func_name, *args, **kwargs)

    async def store_action_memory(
        self,
        *,
        action_name: str,
        arguments: dict,
        result: dict,
        target_post: dict | None,
        variant: str,
        action_reason: str | None = None,
    ) -> None:
        record_content = SocialContextBuilder.format_action_history_content(
            locale=self.locale,
            agent_name=self.user_info.name,
            action_name=action_name,
            arguments=arguments,
            target_post=target_post,
            target_context=self.runtime._target_post_context(
                arguments.get("post_id"), target_post
            ),
            action_reason=action_reason,
        )
        importance = heuristic_action_importance(action_name)
        prefix = "[行动经历]" if is_zh_locale(self.locale) else "[Action Experience]"
        record = Record(
            type="action",
            post_id=target_post.get("post_id") if target_post else None,
            author_id=target_post.get("user_id") if target_post else None,
            author_name=target_post.get("user_name") if target_post else None,
            content=prefix + f"[{variant}] " + record_content,
            create_time=self.env.current_time,
            access_time=self.env.current_time,
            importance_score=importance,
            memory_layer="short_term",
        )
        write_result = await self.memory.write_recagent_observation(record)
        reflect_records = write_result.get("reflect_records")
        if reflect_records:
            await self.runtime.handle_reflection(reflect_records, record)
