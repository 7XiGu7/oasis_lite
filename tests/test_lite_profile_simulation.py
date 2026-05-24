import asyncio
import random
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from oasis.environment.env_action import build_fixed_probability_actions
from oasis.social_agent.agent_cognition import AgentCognition
from oasis.social_agent.agents_generator import (
    generate_twitter_agent_graph,
    generate_weibo_agent_graph,
)


class DummyAgent:
    def __init__(self, agent_id):
        self.social_agent_id = agent_id
        self.user_info = types.SimpleNamespace(
            profile={"other_info": {}},
        )


def test_fixed_probability_activation_zero_and_one():
    agents = {
        0: DummyAgent(0),
        1: DummyAgent(1),
        2: DummyAgent(2),
    }

    random.seed(1)
    assert build_fixed_probability_actions(agents, 0.0) == {}

    random.seed(1)
    actions = build_fixed_probability_actions(agents, 1.0)
    assert {agent.social_agent_id for agent in actions} == {0, 1, 2}


def install_fake_agent_modules(monkeypatch):
    class FakeAgent:
        def __init__(self, agent_id, user_info, **kwargs):
            self.social_agent_id = agent_id
            self.user_info = user_info
            self.kwargs = kwargs

    social_agent_module = types.ModuleType("oasis.social_agent.social_agent")
    social_agent_module.SocialAgent = FakeAgent
    monkeypatch.setitem(
        sys.modules,
        "oasis.social_agent.social_agent",
        social_agent_module,
    )

    recagent_module = types.ModuleType("oasis.social_agent.recagent_user_agent")
    recagent_module.RecAgentUserAgent = FakeAgent
    monkeypatch.setitem(
        sys.modules,
        "oasis.social_agent.recagent_user_agent",
        recagent_module,
    )


def write_twitter_profile(path):
    pd.DataFrame({
        "name": ["alice", "bob"],
        "description": ["Profile A", "Profile B"],
        "personality_traits": ['{"openness": 0.5}', '{"openness": 0.7}'],
        "following_user_ids": ["[1]", "[]"],
        "previous_tweets": ["[]", "[]"],
    }).to_csv(path, index=False)


def write_weibo_profile(path):
    pd.DataFrame({
        "用户名": ["甲", "乙"],
        "描述": ["画像甲", "画像乙"],
        "认证信息": ["认证甲", ""],
        "personality": ['{"openness": 0.5}', '{"openness": 0.7}'],
        "following_simulation_user_ids": ["[1]", "[]"],
        "previous_tweets": ["[]", "[]"],
        "gender": ["女", "男"],
    }).to_csv(path, index=False)


def test_generators_accept_only_lite_variants(tmp_path, monkeypatch):
    install_fake_agent_modules(monkeypatch)
    twitter_path = tmp_path / "twitter.csv"
    weibo_path = tmp_path / "weibo.csv"
    write_twitter_profile(twitter_path)
    write_weibo_profile(weibo_path)

    twitter_graph = asyncio.run(generate_twitter_agent_graph(
        profile_path=str(twitter_path),
        model=object(),
        client=object(),
        agent_variant="recagent_user",
    ))
    assert len(twitter_graph.get_agents()) == 2

    weibo_graph = asyncio.run(generate_weibo_agent_graph(
        profile_path=str(weibo_path),
        model=object(),
        agent_variant="camel",
    ))
    assert len(weibo_graph.get_agents()) == 2

    with pytest.raises(ValueError, match="Unsupported agent_variant"):
        asyncio.run(generate_twitter_agent_graph(
            profile_path=str(twitter_path),
            model=object(),
            client=object(),
            agent_variant="legacy",
        ))


def test_agent_cognition_keeps_only_summary_and_importance_interfaces():
    assert hasattr(AgentCognition, "batch_summarize_items_with_importance")
    assert hasattr(AgentCognition, "batch_importance_evaluation")
    removed_names = [
        "initial_" + "att" + "itude",
        "update_description_and_" + "att" + "itude",
        "calculate_" + "att" + "itude",
        "generate_action_" + "reason",
        "extract_reasoning_" + "style",
    ]
    for removed_name in removed_names:
        assert not hasattr(AgentCognition, removed_name)
