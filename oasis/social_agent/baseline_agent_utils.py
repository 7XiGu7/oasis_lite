from __future__ import annotations

import asyncio
import json
from typing import Any

import chromadb

from oasis.localization import is_zh_locale, platform_display_name
from oasis.social_agent.social_context_builder import SocialContextBuilder
from oasis.social_agent.agent_action import SocialAction
from oasis.social_agent.agent_cognition import AgentCognition
from oasis.social_agent.agent_common import (
    AGENT_ACTION_LOG_PATH,
    _serialize_obj,
    agent_log,
)
from oasis.social_agent.agent_environment import SocialEnvironment
from oasis.social_agent.agent_memory import Memory, Record
from oasis.social_platform import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.typing import ActionType


def _fallback_action_type_names() -> frozenset[str]:
    try:
        return frozenset(action.value for action in ActionType)
    except TypeError:
        return frozenset()


TOOL_ACTION_NAMES = (
    frozenset(SocialAction._action_function_names)
    or _fallback_action_type_names()
)
POST_ID_ACTIONS = frozenset({
    ActionType.LIKE_POST.value,
    ActionType.REPOST.value,
    ActionType.QUOTE_POST.value,
    ActionType.CREATE_COMMENT.value,
    ActionType.FOLLOW.value,
    ActionType.DO_NOTHING.value,
    ActionType.UNLIKE_POST.value,
    ActionType.DISLIKE_POST.value,
    ActionType.UNDO_DISLIKE_POST.value,
    ActionType.REPORT_POST.value,
})


def initialize_baseline_agent(
    agent: Any,
    *,
    agent_id: int,
    user_info: UserInfo,
    model: Any,
    embedding_model_config: dict,
    client: chromadb.Client,
    channel: Channel | None = None,
    available_actions: list[ActionType] | None = None,
    max_content_length: int = 280,
    enable_thinking: bool | None = None,
) -> None:
    agent.llm_semaphore = asyncio.Semaphore(20)
    agent.model = model
    agent.social_agent_id = agent_id
    agent.user_info = user_info
    agent.channel = channel or Channel()
    agent.locale = getattr(user_info, "locale", "en")
    agent.platform_name = platform_display_name(
        getattr(user_info, "platform_name", None),
        agent.locale,
    )
    agent.enable_thinking = enable_thinking
    agent.action = SocialAction(
        agent_id,
        agent.channel,
        locale=agent.locale,
        max_content_length=max_content_length,
    )
    agent.env = SocialEnvironment(agent.action, agent.user_info, locale=agent.locale)
    agent.cognition_model = AgentCognition(agent_id, agent.model, agent.user_info)
    agent.memory = Memory(
        agent_id,
        agent.user_info.name,
        client,
        model,
        embedding_model_config,
    )
    agent.available_actions = [
        action.value if isinstance(action, ActionType) else str(action)
        for action in (available_actions or [])
    ]
    if available_actions:
        agent.action_tools = agent.action.get_openai_function_list_for_actions(
            available_actions
        )
    else:
        agent_log.info("No available actions defined, using all actions.")
        agent.action_tools = agent.action.get_openai_function_list_for_actions()
    agent.profile_attr = build_profile_attr(agent)
    other_info = (agent.user_info.profile or {}).get("other_info", {})
    agent.description = (
        other_info.get("user_profile")
        or agent.user_info.description
        or ""
    )
    agent.is_init = False
    agent.login_times = []
    agent.pending_description_records = []
    agent.runtime = BaselineAgentRuntime(agent)


def other_info(agent: Any) -> dict:
    return (agent.user_info.profile or {}).get("other_info", {})


def profile_value(agent: Any, key: str, default: Any = None) -> Any:
    return getattr(agent.user_info, key, None) or other_info(agent).get(key, default)


def format_profile_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_profile_attr(agent: Any) -> str:
    info = other_info(agent)
    traits = info.get("traits_info") or info.get("traits")
    if is_zh_locale(agent.locale):
        fields = [
            ("姓名", profile_value(agent, "name")),
            ("性别", profile_value(agent, "gender")),
            ("人格特征", traits),
        ]
        return " | ".join(
            f"{label}：{format_profile_value(value)}"
            for label, value in fields
            if value not in (None, "")
        )

    fields = [
        ("Name", profile_value(agent, "name")),
        ("Personality traits", traits),
    ]
    return " | ".join(
        f"{label}: {format_profile_value(value)}"
        for label, value in fields
        if value not in (None, "")
    )


def normalize_tool_action_name(action_name: str) -> str:
    if not isinstance(action_name, str):
        return action_name
    action_name = action_name.strip()
    if action_name in TOOL_ACTION_NAMES:
        return action_name
    if "<|" in action_name:
        candidate = action_name.split("<|", 1)[0].strip()
        if candidate in TOOL_ACTION_NAMES:
            return candidate
    return action_name


def format_available_actions(agent: Any) -> str:
    names = [tool.get("function", {}).get("name") for tool in agent.action_tools]
    names = [name for name in names if name]
    if is_zh_locale(agent.locale):
        return "、".join(names)
    return ", ".join(names)


def available_tool_names(agent: Any) -> set[str]:
    return {
        tool.get("function", {}).get("name")
        for tool in agent.action_tools
        if tool.get("function", {}).get("name")
    }


def format_records(records: list[Record], *, empty_text: str) -> str:
    if not records:
        return empty_text
    return "\n".join(
        f"{idx}. {record.content}"
        for idx, record in enumerate(records, 1)
    )


def format_observation_context(agent: Any) -> str:
    posts = agent.env.posts or []
    if not posts:
        return "当前没有可见帖子。" if is_zh_locale(agent.locale) else "No visible posts."

    blocks = []
    for idx, post in enumerate(posts, 1):
        comments = post.get("comments") or []

        if is_zh_locale(agent.locale):
            block = (
                f"第 {idx} 条\n"
                f"用户 ID：{post.get('user_id')} | 用户名：{post.get('user_name')}\n"
                f"帖子 ID：{post.get('post_id')} | 内容：“{post.get('content')}”\n"
                f"点赞数：{post.get('num_likes')} | 评论数：{len(comments)} | "
                f"转发数：{post.get('num_shares')} | "
                f"来源：{'已关注用户' if post.get('is_following') else '推荐'}"
                "\n"
            )
        else:
            block = (
                f"No.{idx}\n"
                f"User id: {post.get('user_id')} | User name: {post.get('user_name')}\n"
                f"Post id: {post.get('post_id')} | Content: \"{post.get('content')}\"\n"
                f"Likes: {post.get('num_likes')} | Comments: {len(comments)} | "
                f"Shares: {post.get('num_shares')} | "
                f"Source: {'Following' if post.get('is_following') else 'Recommendation'}"
                "\n"
            )
        blocks.append(block)
    return "\n\n".join(blocks)


def format_notifications(agent: Any) -> str:
    notifications = agent.env.notifications or []
    if not notifications:
        return "没有新通知。" if is_zh_locale(agent.locale) else "No new notifications."
    return "\n".join(
        f"{idx}. {item.get('content')}"
        for idx, item in enumerate(notifications, 1)
    )


class BaselineAgentRuntime:
    def __init__(self, agent: Any):
        self.agent = agent

    async def prepare_environment(self) -> None:
        agent = self.agent
        await agent.env.get_posts()
        agent.login_times.append(agent.env.current_time)
        agent.cognition_model.current_time = agent.env.current_time
        await agent.env.get_notifications()

    async def observe_summarized(self) -> None:
        await self._observe_items("post")
        await self._observe_items("notification")
        self.agent.is_init = True

    async def observe_no_reflection(self) -> None:
        await self._observe_items("post", no_reflection=True)
        await self._observe_items("notification", no_reflection=True)
        self.agent.is_init = True

    async def observe_recagent(self) -> None:
        await self._observe_items("post", recagent_memory=True)
        await self._observe_items("notification", recagent_memory=True)
        self.agent.is_init = True

    async def _observe_items(
        self,
        obs_type: str,
        *,
        recagent_memory: bool = False,
        no_reflection: bool = False,
    ) -> None:
        agent = self.agent
        items = agent.env.posts if obs_type == "post" else agent.env.notifications
        if not items:
            return

        raw_contents = [item["content"] for item in items]
        observation_items = await agent.cognition_model.batch_summarize_items_with_importance(
            raw_contents,
            agent.user_info.name,
            obs_type,
            agent.profile_attr + "\n" + agent.description,
        )

        records = []
        for item, observation_item in zip(items, observation_items):
            summary = str(observation_item.get("summary", ""))
            score = int(observation_item.get("importance", 5))
            summary = summary.replace("\n", " ").replace("\r", " ")
            post_id = item.get("post_id")
            if is_zh_locale(agent.locale):
                label = "帖子观察" if obs_type == "post" else "通知观察"
                content = (
                    f"[{label}，来自 {item.get('user_name')}] "
                    f"摘要：{summary}"
                )
            else:
                content = (
                    f"[{obs_type.capitalize()} by {item.get('user_name')}] "
                    f"Summary: {summary}"
                )

            if obs_type == "post" and not recagent_memory:
                seen = agent.memory.update_observation_record(
                    post_id,
                    agent.env.current_time,
                    new_content=content,
                )
                if seen:
                    continue

            record = Record(
                type=f"observation_{obs_type}",
                post_id=post_id,
                author_id=item.get("user_id"),
                author_name=item.get("user_name"),
                content=content,
                create_time=agent.env.current_time,
                access_time=agent.env.current_time,
                importance_score=score,
            )
            if recagent_memory:
                record.memory_layer = "short_term"
                await agent.memory.write_recagent_observation(record)
            elif no_reflection:
                await agent.memory.add_record_no_reflection(record)
            else:
                records.append(record)

        if records:
            await agent.memory.batch_add_observations(records)

    async def request_text_response(
        self,
        messages: list,
        *,
        enable_thinking: bool | None = None,
    ):
        agent = self.agent
        async with agent.llm_semaphore:
            parts = await agent.model.get_message_response_parts(
                messages,
                enable_thinking=enable_thinking,
            )
        assistant_output = parts.message
        if getattr(assistant_output, "content", None) is None:
            assistant_output.content = parts.content or ""
        messages.append(assistant_output)
        return assistant_output, parts.reasoning_content

    async def request_tool_response(
        self,
        messages: list,
        tools: list[dict] | None = None,
        *,
        enable_thinking: bool | None = None,
    ):
        agent = self.agent
        action_tools = agent.action_tools if tools is None else tools
        effective_enable_thinking = (
            agent.enable_thinking if enable_thinking is None else enable_thinking
        )
        async with agent.llm_semaphore:
            parts = await agent.model.get_tool_call_response_parts(
                messages,
                action_tools,
                enable_thinking=effective_enable_thinking,
            )
        assistant_output = parts.message
        if getattr(assistant_output, "content", None) is None:
            assistant_output.content = parts.content or ""
        tool_calls = parts.tool_calls
        if tool_calls is not None and len(tool_calls) > 1:
            raise ValueError(
                f"Expected at most 1 tool call, got {len(tool_calls)}."
            )
        postprocess_reasoning = getattr(
            agent,
            "postprocess_reasoning_content",
            None,
        )
        reasoning_content = parts.reasoning_content
        if callable(postprocess_reasoning):
            reasoning_content = postprocess_reasoning(
                reasoning_content,
                assistant_output,
            )
        messages.append(assistant_output)
        return assistant_output, reasoning_content

    async def run_tool_feedback_loop(
        self,
        messages: list,
        assistant_output: Any,
        *,
        variant: str,
        reasoning_content: str | None = None,
        enable_thinking: bool | None = None,
        tools: list[dict] | None = None,
        allowed_action_names: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> list:
        agent = self.agent
        max_retries = 5
        retry_count = 0
        previous_arguments = None
        allowed_action_names = (
            set(allowed_action_names)
            if allowed_action_names is not None else None
        )

        while retry_count < max_retries:
            tool_calls = getattr(assistant_output, "tool_calls", None)
            if not tool_calls:
                await self._fallback_do_nothing(variant, reasoning_content)
                break

            tool_call = tool_calls[0]
            tool_call_id = tool_call.id
            raw_action_name = tool_call.function.name
            action_name = normalize_tool_action_name(raw_action_name)
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                feedback = self._feedback(
                    None,
                    action_name,
                    f"Invalid JSON arguments: {exc}",
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": feedback,
                })
                assistant_output, reasoning_content = await self.request_tool_response(
                    messages,
                    tools,
                    enable_thinking=enable_thinking,
                )
                retry_count += 1
                continue

            if (
                action_name not in TOOL_ACTION_NAMES
                or action_name not in available_tool_names(agent)
                or (
                    allowed_action_names is not None
                    and action_name not in allowed_action_names
                )
            ):
                if allowed_action_names is None:
                    error = f"Unsupported or unavailable tool name {raw_action_name!r}."
                else:
                    error = (
                        f"Unsupported, unavailable, or non-fixed tool name "
                        f"{raw_action_name!r}. The allowed action is one of "
                        f"{sorted(allowed_action_names)}."
                    )
                feedback = self._feedback(
                    arguments.get("post_id"),
                    action_name,
                    error,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": feedback,
                })
                assistant_output, reasoning_content = await self.request_tool_response(
                    messages,
                    tools,
                    enable_thinking=enable_thinking,
                )
                retry_count += 1
                continue

            if (
                previous_arguments is not None
                and previous_arguments.get("content")
                and previous_arguments.get("content") == arguments.get("content")
            ):
                feedback = self._feedback(
                    arguments.get("post_id"),
                    action_name,
                    "The generated content repeats the previous failed content.",
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": feedback,
                })
                assistant_output, reasoning_content = await self.request_tool_response(
                    messages,
                    tools,
                    enable_thinking=enable_thinking,
                )
                retry_count += 1
                continue

            previous_arguments = arguments
            target_post = self._target_post(arguments.get("post_id"))
            if action_name in POST_ID_ACTIONS and target_post is None:
                env_post_ids = [post.get("post_id") for post in (agent.env.posts or [])]
                result = (
                    f"当前环境中不存在该 post_id。可选帖子 ID：{env_post_ids}。"
                    if is_zh_locale(agent.locale)
                    else f"The post_id is not visible. Available post IDs: {env_post_ids}."
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": self._feedback(arguments.get("post_id"), action_name, result),
                })
                assistant_output, reasoning_content = await self.request_tool_response(
                    messages,
                    tools,
                    enable_thinking=enable_thinking,
                )
                retry_count += 1
                continue

            try:
                action_function = getattr(agent.action, action_name)
                result = await action_function(**arguments)
            except Exception as exc:
                result = {"success": False, "error": str(exc)}

            if not result.get("success", False):
                platform_error = str(
                    result.get("error", "Unknown platform execution failure.")
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": self._feedback(
                        arguments.get("post_id"),
                        action_name,
                        platform_error,
                    ),
                })
                assistant_output, reasoning_content = await self.request_tool_response(
                    messages,
                    tools,
                    enable_thinking=enable_thinking,
                )
                retry_count += 1
                continue

            await self._store_action_memory(
                action_name,
                arguments,
                result,
                target_post,
                variant=variant,
            )
            break

        if retry_count >= max_retries:
            agent_log.error(
                f"Agent {agent.social_agent_id} variant={variant} reached "
                f"maximum action retries."
            )
            await self._fallback_do_nothing(variant, reasoning_content)
        return messages

    def _feedback(self, post_id: Any, action_name: str, result: str) -> str:
        agent = self.agent
        if is_zh_locale(agent.locale):
            return (
                f"你刚才尝试对帖子 {post_id} 执行 "
                f"{action_name}，但该行动无法执行，原因是：{result}\n"
                "请改为调用一个当前可执行的工具，且最多调用一个工具。"
            )
        return (
            f"You tried to perform {action_name} on post {post_id}, but it cannot "
            f"be executed because: "
            f"{result}\nPlease call one currently feasible tool."
        )

    def _target_post(self, post_id: Any) -> dict | None:
        if post_id is None:
            return None
        for post in self.agent.env.posts or []:
            if str(post.get("post_id")) == str(post_id):
                return post
        return None

    def _target_post_context(
        self,
        post_id: Any,
        target_post: dict | None,
    ) -> str:
        observation = None
        if post_id is not None:
            observation = self.agent.memory.get_observation_summary_by_post_id(post_id)
        return SocialContextBuilder.post_context_from_target(
            locale=self.agent.locale,
            target_post=target_post,
            observation_memory=observation,
        )

    async def _fallback_do_nothing(
        self,
        variant: str,
        reasoning_content: str | None,
    ) -> None:
        agent = self.agent
        if ActionType.DO_NOTHING.value not in available_tool_names(agent):
            agent_log.info(
                f"Agent {agent.social_agent_id} did not call a tool and "
                "do_nothing is unavailable; ending without platform action."
            )
            return
        post_id = (agent.env.posts or [{}])[0].get("post_id", 0)
        result = await agent.action.do_nothing(post_id=post_id)
        if result.get("success", False):
            target_post = self._target_post(post_id)
            await self._store_action_memory(
                ActionType.DO_NOTHING.value,
                {"post_id": post_id},
                result,
                target_post,
                variant=variant,
            )

    async def _store_action_memory(
        self,
        action_name: str,
        arguments: dict,
        result: dict,
        target_post: dict | None,
        *,
        variant: str,
    ) -> None:
        agent = self.agent
        store_action_memory = getattr(agent, "store_action_memory", None)
        if store_action_memory is None:
            return
        await store_action_memory(
            action_name=action_name,
            arguments=arguments,
            result=result,
            target_post=target_post,
            variant=variant,
        )

    async def finalize(self, messages: list, *, variant: str) -> None:
        agent = self.agent
        final_record = {
            "agent_id": agent.social_agent_id,
            "variant": variant,
            "timestamp": agent.env.current_time,
            "messages": [_serialize_obj(message) for message in messages],
        }
        try:
            with open(AGENT_ACTION_LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(final_record, ensure_ascii=False) + "\n")
        except Exception as exc:
            agent_log.error(f"Failed to save baseline conversation: {exc}")
        await agent.cognition_model.save_cognition_records()
        agent.env.clear_env()

    async def perform_action_by_data(self, func_name, *args, **kwargs) -> Any:
        agent = self.agent
        func_name = func_name.value if isinstance(func_name, ActionType) else func_name
        function = getattr(agent.action, func_name, None)
        result = await function(*args, **kwargs)
        if isinstance(result, dict) and not result.get("success", False):
            agent_log.error(
                f"Agent {agent.social_agent_id} manual action {func_name} failed: {result}"
            )
        target_post = self._target_post(kwargs.get("post_id"))
        await self._store_action_memory(
            func_name,
            kwargs,
            result,
            target_post,
            variant="manual",
        )
        return result
