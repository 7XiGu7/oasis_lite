from __future__ import annotations

import json
from typing import Any

from oasis.localization import is_zh_locale, zh_action_label
from oasis.social_platform.typing import ActionType


_SUMMARY_MARKERS = ("Summary:", "摘要：")


class SocialContextBuilder:
    """Formatting and context-building helpers shared by social-agent classes."""

    @classmethod
    def extract_observation_summary(cls, memory_content: str | None) -> str:
        text = "" if memory_content is None else str(memory_content).strip()
        for marker in _SUMMARY_MARKERS:
            if marker in text:
                summary = text.split(marker, 1)[1]
                return summary.split("|", 1)[0].strip()
        return text

    @classmethod
    def post_context_from_target(
        cls,
        *,
        locale: str,
        target_post: dict[str, Any] | None,
        observation_memory: str | None,
    ) -> str:
        content_text = cls.extract_observation_summary(observation_memory)
        if not content_text and target_post:
            content_text = str(target_post.get("content") or "").strip()
        return content_text

    # Agent context builders
    # These methods build the user/profile/memory/post text blocks that the
    # agent inserts into action-selection prompts.

    @staticmethod
    def format_profile_value(value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @classmethod
    def build_profile_attr(cls, *, user_info: Any, locale: str) -> str:
        other_info = (user_info.profile or {}).get("other_info", {})
        traits = other_info.get("traits_info") or other_info.get("traits")

        def profile_value(key: str, default: Any = None) -> Any:
            return getattr(user_info, key, None) or other_info.get(key, default)

        if is_zh_locale(locale):
            profile_fields = [
                ("name", profile_value("name")),
                ("gender", profile_value("gender")),
                ("traits", traits),
            ]
            labels = {
                "name": "姓名",
                "gender": "性别",
                "traits": "人格特征",
            }
            parts = [
                f"{labels[key]}：{cls.format_profile_value(value)}"
                for key, value in profile_fields
                if value not in (None, "")
            ]
            return " | ".join(parts)

        profile_fields = [
            ("name", profile_value("name")),
            ("traits", traits),
        ]
        labels = {
            "name": "Name",
            "traits": "Personality traits",
        }
        parts = [
            f"{labels[key]}: {cls.format_profile_value(value)}"
            for key, value in profile_fields
            if value not in (None, "")
        ]
        return " | ".join(parts)

    @staticmethod
    def format_records(records: list[Any], *, empty_text: str = "") -> str:
        if not records:
            return empty_text
        return "\n".join(
            f"{idx}. {record.content}" for idx, record in enumerate(records, 1)
        )

    @classmethod
    def format_recent_memory(cls, memory_records: list[Any]) -> str:
        return cls.format_records(memory_records)

    @classmethod
    def build_combed_memory(
        cls,
        *,
        locale: str,
        posts: list[dict[str, Any]],
    ) -> str:
        combed_contents = []
        for info in posts:
            content = info["content"]
            user_id = int(info["user_id"])
            name = info["user_name"]
            post_id = info["post_id"]
            is_following = info["is_following"]
            if is_zh_locale(locale):
                comb_content = (
                    f"用户 ID：{user_id} | 用户名：{name}\n"
                    f"帖子 ID：{post_id} | 帖子内容：“{content}”\n"
                    f"点赞数：“{info['num_likes']}” | "
                    f"评论数：“{len(info['comments'])}” | "
                    f"转发数：“{info['num_shares']}”。\n"
                    f"来源：{'已关注用户' if is_following else '推荐'}。"
                )
            else:
                comb_content = (
                    f"User id: {user_id} | User name: {name}  \n"
                    f"Post id: {post_id} | Post content: '{content}'  \n"
                    f"Num of Likes:'{info['num_likes']}' | "
                    f"Num of Comments:'{len(info['comments'])}' | "
                    f"Num of Shares:'{info['num_shares']}'. \n"
                    f"Source: {'Following' if is_following else 'Recommendation'}."
                )
            combed_contents.append(comb_content)
        if is_zh_locale(locale):
            return "\n" + "\n".join(
                f"第 {idx} 条：{content}\n"
                for idx, content in enumerate(combed_contents, 1)
            )
        return "\n" + "\n".join(
            f"No.{idx}. {content}\n"
            for idx, content in enumerate(combed_contents, 1)
        )

    # Action-memory builders

    @staticmethod
    def append_reason_trace(
        trace: list[dict[str, Any]],
        *,
        action_name: str,
        arguments: dict[str, Any] | None,
        reasoning_content: str | None,
        status: str,
        include_in_reason: bool,
        feedback: str | None = None,
    ) -> None:
        trace.append({
            "action_name": action_name,
            "arguments": arguments or {},
            "reasoning_content": reasoning_content,
            "status": status,
            "include_in_reason": include_in_reason,
            "feedback": feedback,
        })



    @classmethod
    def format_reason_trace_for_prompt(cls, trace: list[dict[str, Any]], locale: str) -> str:
        lines = []
        for idx, item in enumerate(trace, 1):
            if not item.get("include_in_reason"):
                continue
            if is_zh_locale(locale):
                reasoning = str(item.get("reasoning_content") or "").strip()
                feedback = str(item.get("feedback") or "").strip()
                if not reasoning and not feedback:
                    continue
                arguments = item.get("arguments") or {}
                content = str(arguments.get("content") or "").strip()
                post_id = arguments.get("post_id")
                parts = [
                    f"{idx}. 状态={item.get('status')}",
                    f"行动={item.get('action_name')}",
                ]
                if post_id is not None:
                    parts.append(f"帖子 ID={post_id}")
                if content:
                    parts.append(f"生成内容=\"{content}\"")
                if reasoning:
                    parts.append(f"行动理由=\"{reasoning}\"")
                if feedback:
                    parts.append(f"反馈=\"{feedback}\"")
                lines.append(" | ".join(parts))
            else:
                reasoning = str(item.get("reasoning_content") or "").strip()
                feedback = str(item.get("feedback") or "").strip()
                if not reasoning and not feedback:
                    continue
                arguments = item.get("arguments") or {}
                content = str(arguments.get("content") or "").strip()
                post_id = arguments.get("post_id")
                parts = [
                    f"{idx}. status={item.get('status')}",
                    f"action={item.get('action_name')}",
                ]
                if post_id is not None:
                    parts.append(f"post_id={post_id}")
                if content:
                    parts.append(f"content={content}")
                if reasoning:
                    parts.append(f"reasoning={reasoning}")
                if feedback:
                    parts.append(f"feedback={feedback}")
                lines.append(" | ".join(parts))
        return "\n".join(lines)

    @staticmethod
    def _en_action_phrase(action_name: str) -> str:
        return {
            ActionType.LIKE_POST.value: "showed approval of",
            ActionType.UNLIKE_POST.value: "withdrew approval from",
            ActionType.DISLIKE_POST.value: "reacted negatively to",
            ActionType.UNDO_DISLIKE_POST.value: "withdrew a negative reaction to",
            ActionType.REPOST.value: "shared",
            ActionType.QUOTE_POST.value: "quoted",
            ActionType.CREATE_COMMENT.value: "commented on",
            ActionType.FOLLOW.value: "followed the author after reading",
            ActionType.DO_NOTHING.value: "read without interacting with",
            ActionType.REPORT_POST.value: "reported",
        }.get(action_name, "responded to")

    @staticmethod
    def _zh_action_phrase(action_name: str) -> str:
        return {
            ActionType.LIKE_POST.value: "点赞了",
            ActionType.UNLIKE_POST.value: "取消点赞了",
            ActionType.DISLIKE_POST.value: "表达了负面反馈于",
            ActionType.UNDO_DISLIKE_POST.value: "撤回了对",
            ActionType.REPOST.value: "转发了",
            ActionType.QUOTE_POST.value: "引用了",
            ActionType.CREATE_COMMENT.value: "评论了",
            ActionType.FOLLOW.value: "关注了",
            ActionType.DO_NOTHING.value: "阅读但没有互动于",
            ActionType.REPORT_POST.value: "举报了",
        }.get(action_name, f"以“{zh_action_label(action_name)}”回应了")

    @classmethod
    def format_action_history_content(
        cls,
        *,
        locale: str,
        agent_name: str,
        action_name: str,
        arguments: dict[str, Any] | None = None,
        target_post: dict[str, Any] | None = None,
        target_context: str | None = None,
        action_reason: str | None = None
    ) -> str:
        """Build human-readable action memory without platform post ids."""
        arguments = arguments or {}
        content = str(arguments.get("content") or "").strip()
        report_reason = str(arguments.get("report_reason") or "").strip()
        reason = str(
            action_reason if action_reason is not None else ""
        ).strip()
        context = str(target_context or "").strip()
        author = str(
            (target_post or {}).get("user_name")
            or (target_post or {}).get("author_name")
            or ""
        ).strip()
        has_target = bool(context or author)

        if is_zh_locale(locale):
            action_label = zh_action_label(action_name)
            if action_name == ActionType.CREATE_POST.value:
                record = (
                    f"{agent_name} 发布了一条内容"
                    f"：“{content}”。" if content else f"{agent_name} 发布了一条新内容。"
                )
            elif action_name == ActionType.FOLLOW.value:
                target = f"作者 {author}" if author else "相关作者"
                if context:
                    record = f"{agent_name} 因一篇内容关注了{target}，该内容上下文为：“{context}”。"
                else:
                    record = f"{agent_name} 关注了{target}。"
            elif has_target:
                target = f"作者 {author} 的一篇内容" if author else "一篇内容"
                if context:
                    target += f"，该内容上下文为：“{context}”"
                record = f"{agent_name} {cls._zh_action_phrase(action_name)}{target}。"
            else:
                record = f"{agent_name} 进行了“{action_label}”相关互动。"

            if content and action_name != ActionType.CREATE_POST.value:
                record += f"生成内容：“{content}”。"
            if report_reason:
                record += f"报告理由：“{report_reason}”。"
            if reason:
                record += f"行动理由：{reason}"
            return record

        if action_name == ActionType.CREATE_POST.value:
            record = (
                f"{agent_name} created a post saying: \"{content}\"."
                if content else f"{agent_name} created a new post."
            )
        elif action_name == ActionType.FOLLOW.value:
            target = f"author {author}" if author else "the related author"
            if context:
                record = (
                    f"{agent_name} followed {target} after reading a post with context: "
                    f"\"{context}\"."
                )
            else:
                record = f"{agent_name} followed {target}."
        elif has_target:
            target = f"a post by {author}" if author else "a post"
            if context:
                target += f" with context: \"{context}\""
            record = f"{agent_name} {cls._en_action_phrase(action_name)} {target}."
        else:
            record = f"{agent_name} took a social-platform action."

        if content and action_name != ActionType.CREATE_POST.value:
            record += f" Generated content: \"{content}\"."
        if report_reason:
            record += f" Report reason: \"{report_reason}\"."
        if reason:
            record += f" Action reason: {reason}"
        return record

    @classmethod
    def format_description_records(cls, records: list[Any]) -> str:
        return cls.format_records(records)

    # AgentCognition prompt-input builders
    # These methods prepare deterministic prompt inputs before AgentCognition
    # sends them to the LLM.

    @staticmethod
    def normalize_item_type(item_type: str) -> tuple[str, str, str]:
        if item_type == "notification":
            return "notification", "notifications", "通知"
        if item_type == "post":
            return "post", "posts", "帖子"
        return "item", "items", "内容"

    @classmethod
    def format_batch_summary_items(
        cls,
        *,
        locale: str,
        item_contents: list[str],
        item_type: str,
    ) -> tuple[str, str, str, str]:
        item_type_en, item_type_plural, item_type_zh = cls.normalize_item_type(item_type)
        numbered = "\n\n".join(
            (
                f"[{item_type_zh} {i}]\n{content}"
                if is_zh_locale(locale)
                else f"[{item_type_en.capitalize()} {i}]\n{content}"
            )
            for i, content in enumerate(item_contents)
        )
        return item_type_en, item_type_plural, item_type_zh, numbered

    @staticmethod
    def records_to_memory_content(records: list[Any]) -> str:
        return "\n".join(record.content for record in records)

    @staticmethod
    def records_to_indexed_statements(records: list[Any]) -> str:
        return "\n".join(
            f"{idx}. {record.content}" for idx, record in enumerate(records)
        )

    # Memory/Record display helpers
    # This area is reserved for deterministic Record presentation shared by
    # memory views and prompt builders.

    @classmethod
    def format_observation_record_content(
        cls,
        *,
        locale: str,
        obs_type: str,
        user_name: str,
        summary: str,
        post_attr: dict[str, Any] | str | None = None,
    ) -> str:
        if is_zh_locale(locale):
            obs_label = "帖子观察" if obs_type == "post" else "通知观察"
            content = f"[{obs_label}，来自 {user_name}] 摘要：{summary}"
            return content
        content = f"[{obs_type.capitalize()} by {user_name}] Summary: {summary}"
        return content
