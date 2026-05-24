from .agent_cognition import AgentCognition
from .agent_graph import AgentGraph
from .agents_generator import generate_twitter_agent_graph, generate_weibo_agent_graph

__all__ = [
    "AgentCognition",
    "AgentGraph",
    "Memory",
    "RecAgentUserAgent",
    "SocialAgent",
    "generate_twitter_agent_graph",
    "generate_weibo_agent_graph",
]


def __getattr__(name: str):
    if name == "SocialAgent":
        from .social_agent import SocialAgent

        return SocialAgent
    if name == "RecAgentUserAgent":
        from .recagent_user_agent import RecAgentUserAgent

        return RecAgentUserAgent
    if name == "Memory":
        from .agent_memory import Memory

        return Memory
    raise AttributeError(name)
