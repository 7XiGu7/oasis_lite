__version__ = "0.2.3"

from oasis.environment.env_action import LLMAction, ManualAction
from oasis.environment.make import make
from oasis.social_agent import generate_twitter_agent_graph, generate_weibo_agent_graph
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_platform.config import UserInfo
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType, DefaultPlatformType, RecsysConfig
from oasis.testing.show_db import print_db_contents

__all__ = [
    "ActionType",
    "AgentGraph",
    "DefaultPlatformType",
    "LLMAction",
    "ManualAction",
    "Platform",
    "RecAgentUserAgent",
    "RecsysConfig",
    "SocialAgent",
    "UserInfo",
    "generate_twitter_agent_graph",
    "generate_weibo_agent_graph",
    "make",
    "print_db_contents",
]


def __getattr__(name: str):
    if name == "SocialAgent":
        from oasis.social_agent.social_agent import SocialAgent

        return SocialAgent
    if name == "RecAgentUserAgent":
        from oasis.social_agent.recagent_user_agent import RecAgentUserAgent

        return RecAgentUserAgent
    raise AttributeError(name)
