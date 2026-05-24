from __future__ import annotations

import ast
import json
from typing import Any

import pandas as pd

from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_platform import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.typing import ActionType


SUPPORTED_AGENT_VARIANTS = frozenset({"camel", "recagent_user"})


def connect_platform_channel(
    channel: Channel,
    agent_graph: AgentGraph | None = None,
) -> AgentGraph:
    if agent_graph is None:
        agent_graph = AgentGraph()
    for _, agent in agent_graph.get_agents():
        agent.channel = channel
        agent.env.action.channel = channel
    return agent_graph


async def generate_custom_agents(
    channel: Channel,
    agent_graph: AgentGraph | None = None,
) -> AgentGraph:
    if agent_graph is None:
        agent_graph = AgentGraph()

    agent_graph = connect_platform_channel(
        channel=channel,
        agent_graph=agent_graph,
    )

    import asyncio

    await asyncio.gather(*[
        agent.env.action.sign_up(
            user_name=agent.user_info.user_name,
            name=agent.user_info.name,
            bio=agent.user_info.description,
        )
        for _, agent in agent_graph.get_agents()
    ])

    await asyncio.gather(*[
        agent.env.action.create_post(
            content=agent.user_info.profile["other_info"]["previous_tweets"][0],
        )
        for _, agent in agent_graph.get_agents()
        if agent.user_info.profile["other_info"]["previous_tweets"]
    ])

    follow_tasks = []
    for _, agent in agent_graph.get_agents():
        following_agent_ids = agent.user_info.profile["other_info"][
            "following_id_list"
        ]
        follow_tasks.extend(
            agent.env.action.follow(agent_id, None)
            for agent_id in following_agent_ids
        )
    await asyncio.gather(*follow_tasks)

    return agent_graph


async def generate_twitter_agent_graph(
    profile_path: str,
    model,
    embedding_model_config=None,
    client=None,
    available_actions: list[ActionType] | None = None,
    locale: str = "en",
    platform_name: str | None = None,
    max_content_length: int = 280,
    agent_variant: str = "recagent_user",
    token_limit: int | None = None,
    message_window_size: int | None = None,
) -> AgentGraph:
    return await _generate_agent_graph(
        profile_path=profile_path,
        model=model,
        embedding_model_config=embedding_model_config,
        client=client,
        available_actions=available_actions,
        locale=locale,
        platform_name=platform_name,
        max_content_length=max_content_length,
        agent_variant=agent_variant,
        token_limit=token_limit,
        message_window_size=message_window_size,
        platform="twitter",
    )


async def generate_weibo_agent_graph(
    profile_path: str,
    model,
    embedding_model_config=None,
    client=None,
    available_actions: list[ActionType] | None = None,
    locale: str = "zh_cn",
    platform_name: str | None = "微博",
    max_content_length: int = 280,
    agent_variant: str = "recagent_user",
    token_limit: int | None = None,
    message_window_size: int | None = None,
) -> AgentGraph:
    return await _generate_agent_graph(
        profile_path=profile_path,
        model=model,
        embedding_model_config=embedding_model_config,
        client=client,
        available_actions=available_actions,
        locale=locale,
        platform_name=platform_name,
        max_content_length=max_content_length,
        agent_variant=agent_variant,
        token_limit=token_limit,
        message_window_size=message_window_size,
        platform="weibo",
    )


async def _generate_agent_graph(
    *,
    profile_path: str,
    model,
    embedding_model_config,
    client,
    available_actions: list[ActionType] | None,
    locale: str,
    platform_name: str | None,
    max_content_length: int,
    agent_variant: str,
    token_limit: int | None,
    message_window_size: int | None,
    platform: str,
) -> AgentGraph:
    normalized_variant = (agent_variant or "recagent_user").lower()
    if normalized_variant not in SUPPORTED_AGENT_VARIANTS:
        supported = ", ".join(sorted(SUPPORTED_AGENT_VARIANTS))
        raise ValueError(
            f"Unsupported agent_variant {agent_variant!r}. "
            f"Supported variants: {supported}."
        )
    if normalized_variant == "recagent_user" and client is None:
        raise ValueError("client must be provided for recagent_user.")

    agent_info = pd.read_csv(profile_path)
    agent_graph = AgentGraph()

    for agent_id, row in agent_info.iterrows():
        user_info = (
            _weibo_user_info(
                agent_id=agent_id,
                row=row,
                agent_info=agent_info,
                locale=locale,
                platform_name=platform_name,
            )
            if platform == "weibo"
            else _twitter_user_info(
                agent_id=agent_id,
                row=row,
                agent_info=agent_info,
                locale=locale,
                platform_name=platform_name,
            )
        )

        if normalized_variant == "camel":
            from oasis.social_agent.social_agent import SocialAgent

            agent = SocialAgent(
                agent_id=agent_id,
                user_info=user_info,
                model=model,
                agent_graph=agent_graph,
                available_actions=available_actions,
                token_limit=token_limit,
                message_window_size=message_window_size,
                max_content_length=max_content_length,
            )
        else:
            from oasis.social_agent.recagent_user_agent import RecAgentUserAgent

            agent = RecAgentUserAgent(
                agent_id=agent_id,
                user_info=user_info,
                model=model,
                embedding_model_config=embedding_model_config,
                available_actions=available_actions,
                client=client,
                max_content_length=max_content_length,
            )

        agent_graph.add_agent(agent)
    return agent_graph


def _twitter_user_info(
    *,
    agent_id: int,
    row: pd.Series,
    agent_info: pd.DataFrame,
    locale: str,
    platform_name: str | None,
) -> UserInfo:
    name = _optional_str(_first_present(row, ("name", "user_name", "用户名")))
    if not name:
        name = f"user_{agent_id}"

    description = _optional_str(
        _first_present(row, ("description", "bio", "user_profile", "描述"))
    )
    traits = _parse_traits(_first_present(row, ("personality_traits", "traits")))
    following_id_list = _parse_optional_list(
        _first_present(row, ("following_user_ids", "following_ids"))
    )
    previous_tweets = _parse_optional_list(
        _first_present(row, ("previous_tweets", "previous_posts"))
    )

    other_info: dict[str, Any] = {
        "user_profile": description,
        "traits": traits,
        "traits_info": _format_traits_info(traits),
        "following_id_list": following_id_list,
        "previous_tweets": previous_tweets,
    }

    if following_id_list and "name" in agent_info.columns:
        other_info["following_name_list"] = [
            str(agent_info["name"][followee_id])
            for followee_id in following_id_list
            if isinstance(followee_id, int) and followee_id in agent_info.index
        ]

    return UserInfo(
        name=name,
        user_name=name,
        description=description,
        profile={"nodes": [], "edges": [], "other_info": other_info},
        recsys_type="twitter",
        locale=locale,
        platform_name=platform_name,
    )


def _weibo_user_info(
    *,
    agent_id: int,
    row: pd.Series,
    agent_info: pd.DataFrame,
    locale: str,
    platform_name: str | None,
) -> UserInfo:
    name = _optional_str(_first_present(row, ("用户名", "name", "user_name")))
    if not name:
        name = f"user_{agent_id}"

    raw_description = _optional_str(
        _first_present(row, ("描述", "description", "bio", "user_profile"))
    )
    certification = _optional_str(_first_present(row, ("认证信息", "certification")))
    description = (
        f"认证信息：{certification}，描述：{raw_description}"
        if certification else raw_description
    )
    traits = _parse_traits(
        _first_present(row, ("personality", "personality_traits", "traits"))
    )
    following_id_list = _parse_optional_list(
        _first_present(
            row,
            ("following_simulation_user_ids", "following_user_ids", "following_ids"),
        )
    )
    previous_tweets = _parse_optional_list(
        _first_present(row, ("previous_tweets", "previous_posts"))
    )

    other_info: dict[str, Any] = {
        "user_profile": description,
        "traits": traits,
        "traits_info": _format_traits_info(traits),
        "following_id_list": following_id_list,
        "previous_tweets": previous_tweets,
    }

    name_column = "用户名" if "用户名" in agent_info.columns else "name"
    if following_id_list and name_column in agent_info.columns:
        other_info["following_name_list"] = [
            str(agent_info[name_column][followee_id])
            for followee_id in following_id_list
            if isinstance(followee_id, int) and followee_id in agent_info.index
        ]

    return UserInfo(
        name=name,
        user_name=name,
        gender=_optional_str(_first_present(row, ("gender", "性别"))),
        description=raw_description,
        profile={"nodes": [], "edges": [], "other_info": other_info},
        recsys_type="weibo",
        locale=locale,
        platform_name=platform_name,
    )


def _first_present(row: pd.Series, field_names: tuple[str, ...]) -> Any:
    for field_name in field_names:
        if field_name in row and not _is_missing(row[field_name]):
            return row[field_name]
    return None


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_traits(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or _is_missing(value):
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _format_traits_info(traits: dict[str, Any]) -> str:
    if not traits:
        return ""
    return "; ".join(f"{key}: {value}" for key, value in traits.items())


def _parse_optional_list(value: Any) -> list[Any]:
    if value is None or _is_missing(value):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _optional_str(value: Any) -> str:
    if value is None or _is_missing(value):
        return ""
    return str(value)
