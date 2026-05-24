#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(override=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tiny OASIS environment and real-LLM smoke tests.",
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--activation-prob", type=float, default=0.0)
    parser.add_argument("--llm-timeout", type=float, default=30.0)
    parser.add_argument("--llm-backend", default="dashscope")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--skip-real-llm", action="store_true")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--keep", action="store_true")
    return parser.parse_args()


def make_profile(name: str, description: str, following: list[int]) -> dict:
    return {
        "nodes": [],
        "edges": [],
        "other_info": {
            "user_profile": description,
            "traits": {"openness": 0.6},
            "traits_info": "openness: 0.6",
            "following_id_list": following,
            "previous_tweets": [f"hello from {name}"],
        },
    }


class SmokeAgent:
    def __init__(self, agent_id: int, name: str, description: str, following: list[int]):
        from oasis.social_agent.agent_action import SocialAction
        from oasis.social_agent.agent_environment import SocialEnvironment
        from oasis.social_platform import Channel
        from oasis.social_platform.config import UserInfo

        self.social_agent_id = agent_id
        self.channel = Channel()
        self.user_info = UserInfo(
            name=name,
            user_name=name,
            description=description,
            profile=make_profile(name, description, following),
            recsys_type="twitter",
        )
        self.action = SocialAction(agent_id, self.channel, max_content_length=120)
        self.env = SocialEnvironment(self.action, self.user_info)

    async def perform_action_by_llm(self):
        raise RuntimeError("light_smoke_test should not call the LLM")


def require_success(name: str, result: dict[str, Any]) -> dict[str, Any]:
    if not result.get("success"):
        raise RuntimeError(f"{name} failed: {result}")
    return result


async def exercise_platform_actions(platform) -> dict[str, Any]:
    created = {}

    alice_post = require_success(
        "create_post",
        await platform.create_post(
            0,
            ("smoke: alice writes a richer post mentioning bob", [1]),
        ),
    )["post_id"]
    created["post"] = alice_post
    bob_post = require_success(
        "create_post",
        await platform.create_post(
            1,
            ("smoke: bob adds a second target post mentioning alice", [0]),
        ),
    )["post_id"]
    created["bob_post"] = bob_post

    comment_id = require_success(
        "create_comment",
        await platform.create_comment(
            1,
            (alice_post, "smoke: bob comments back mentioning alice", [0]),
        ),
    )["comment_id"]
    created["comment"] = comment_id

    created["like_post"] = require_success(
        "like_post",
        await platform.like_post(1, alice_post),
    )["like_id"]
    require_success("unlike_post", await platform.unlike_post(1, alice_post))

    created["dislike_post"] = require_success(
        "dislike_post",
        await platform.dislike_post(1, alice_post),
    )["dislike_id"]
    require_success(
        "undo_dislike_post",
        await platform.undo_dislike_post(1, alice_post),
    )

    created["like_comment"] = require_success(
        "like_comment",
        await platform.like_comment(0, comment_id),
    )["comment_like_id"]
    require_success(
        "unlike_comment",
        await platform.unlike_comment(0, comment_id),
    )

    created["comment_dislike"] = require_success(
        "dislike_comment",
        await platform.dislike_comment(0, comment_id),
    )["comment_dislike_id"]
    require_success(
        "undo_dislike_comment",
        await platform.undo_dislike_comment(0, comment_id),
    )

    created["repost"] = require_success(
        "repost",
        await platform.repost(1, alice_post),
    )["post_id"]
    created["quote"] = require_success(
        "quote_post",
        await platform.quote_post(
            1,
            (alice_post, "smoke: bob quotes alice with a note", [0]),
        ),
    )["post_id"]
    created["remaining_like"] = require_success(
        "remaining_like_post",
        await platform.like_post(0, bob_post),
    )["like_id"]
    created["remaining_dislike"] = require_success(
        "remaining_dislike_post",
        await platform.dislike_post(0, created["quote"]),
    )["dislike_id"]
    created["remaining_comment_like"] = require_success(
        "remaining_like_comment",
        await platform.like_comment(1, comment_id),
    )["comment_like_id"]
    created["remaining_comment_dislike"] = require_success(
        "remaining_dislike_comment",
        await platform.dislike_comment(1, comment_id),
    )["comment_dislike_id"]

    created["follow"] = require_success(
        "follow",
        await platform.follow(1, 0),
    )["follow_id"]
    require_success("unfollow", await platform.unfollow(1, 0))

    created["mute"] = require_success(
        "mute",
        await platform.mute(0, 1),
    )["mute_id"]
    require_success("unmute", await platform.unmute(0, 1))
    created["remaining_mute"] = require_success(
        "remaining_mute",
        await platform.mute(1, 0),
    )["mute_id"]

    created["report"] = require_success(
        "report_post",
        await platform.report_post(
            1,
            (alice_post, "smoke test report reason"),
        ),
    )["report_id"]

    created["group"] = require_success(
        "create_group",
        await platform.create_group(0, "smoke-group"),
    )["group_id"]
    require_success(
        "join_group",
        await platform.join_group(1, created["group"]),
    )
    created["group_message"] = require_success(
        "send_to_group",
        await platform.send_to_group(
            1,
            (created["group"], "smoke: group message from bob"),
        ),
    )["message_id"]
    require_success(
        "listen_from_group",
        await platform.listen_from_group(0),
    )
    require_success(
        "leave_group",
        await platform.leave_group(1, created["group"]),
    )

    require_success("search_posts", await platform.search_posts(0, "smoke"))
    require_success("search_user", await platform.search_user(0, "bob"))
    require_success("trend", await platform.trend(0))
    require_success("view_comments", await platform.view_comments(0, alice_post))

    return created


async def run_smoke(args: argparse.Namespace) -> None:
    import oasis
    from oasis.environment.env_action import build_fixed_probability_actions
    from oasis.social_agent.agent_graph import AgentGraph
    from oasis.social_platform.typing import RecsysConfig, RecsysType

    work_dir = Path(args.work_dir) if args.work_dir else Path(
        tempfile.mkdtemp(prefix="oasis-light-smoke-")
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / "tiny_oasis.db"
    if db_path.exists():
        db_path.unlink()
    os.environ["OASIS_DB_PATH"] = str(db_path)

    agent_graph = AgentGraph()
    agent_graph.add_agent(SmokeAgent(0, "alice", "likes careful tests", [1]))
    agent_graph.add_agent(SmokeAgent(1, "bob", "likes tiny demos", []))

    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=str(db_path),
        semaphore=1,
        max_content_length=120,
        recsys_config=RecsysConfig(
            recsys_type=RecsysType.RANDOM,
            refresh_rec_post_count=1,
            max_rec_post_len=1,
            following_post_count=1,
        ),
    )

    await env.reset()
    created = await exercise_platform_actions(env.platform)

    agents_by_id = dict(env.agent_graph.get_agents())
    for _ in range(args.steps):
        user_actions = build_fixed_probability_actions(
            agents_by_id,
            args.activation_prob,
        )
        await env.step(user_actions)

    conn = env.platform.db
    counts = {}
    for table in (
        "user",
        "post",
        "comment",
        "like",
        "dislike",
        "comment_like",
        "comment_dislike",
        "follow",
        "mute",
        "report",
        "chat_group",
        "group_members",
        "group_messages",
        "mention",
        "interaction",
        "rec",
        "trace",
    ):
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    trace_action_counts = dict(
        conn.execute(
            "SELECT action, COUNT(*) FROM trace GROUP BY action ORDER BY action"
        ).fetchall()
    )
    required_trace_actions = (
        "create_post",
        "create_comment",
        "like_post",
        "unlike_post",
        "dislike_post",
        "undo_dislike_post",
        "repost",
        "quote_post",
        "follow",
        "unfollow",
        "mute",
        "unmute",
        "report_post",
        "create_group",
        "join_group",
        "send_to_group",
        "leave_group",
        "search_posts",
        "search_user",
        "trend",
        "like_comment",
        "unlike_comment",
        "dislike_comment",
        "undo_dislike_comment",
        "view_comment",
    )
    missing_actions = [
        action for action in required_trace_actions
        if trace_action_counts.get(action, 0) < 1
    ]
    if missing_actions:
        raise RuntimeError(f"Missing trace actions: {missing_actions}")

    await env.close()

    print("ENV_SMOKE_TEST_OK")
    print(f"activation_prob={args.activation_prob}")
    print(f"work_dir={work_dir}")
    print(f"db_path={db_path}")
    print(
        "created="
        + ", ".join(f"{name}:{value}" for name, value in created.items())
    )
    print(
        "counts="
        + ", ".join(f"{table}:{count}" for table, count in counts.items())
    )
    print(
        "trace_actions="
        + ", ".join(
            f"{action}:{count}" for action, count in trace_action_counts.items()
        )
    )

    if not args.skip_real_llm:
        await run_real_llm_smoke(args)
        print("REAL_LLM_SMOKE_TEST_OK")

    if args.work_dir is None and not args.keep:
        shutil.rmtree(work_dir, ignore_errors=True)


async def run_real_llm_smoke(args: argparse.Namespace) -> None:
    from oasis.social_agent.llm_model import LLMModel

    backend = args.llm_backend
    model_name = args.llm_model
    base_url = args.llm_base_url
    if backend == "dashscope":
        model_name = model_name or os.getenv("DASHSCOPE_MODEL_NAME", "qwen-plus")
        base_url = base_url or os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        os.environ["IS_VLLM"] = "false"
    else:
        model_name = model_name or os.getenv("MODEL_NAME")

    llm = LLMModel(
        backend=backend,
        model_name=model_name,
        base_url=base_url,
        temperature=0.0,
        top_p=1.0,
        enable_thinking=False,
        usage_log_path=None,
        timeout=args.llm_timeout,
    )
    llm.retry_attempts = 1
    parts = await llm.get_message_response_parts(
        [
            {
                "role": "user",
                "content": "Reply with exactly: oasis-lite-ok",
            }
        ],
        enable_thinking=False,
        operation_name="light_smoke_real_llm",
    )
    content = (parts.content or "").strip()
    if not content:
        raise RuntimeError("Real LLM call returned empty content")
    print(f"llm_backend={llm.backend}")
    print(f"llm_model={llm.model_name}")
    print(f"llm_base_url={llm.base_url}")
    print(f"llm_response={content[:200]}")


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.activation_prob != 0:
        print(
            "WARNING: activation-prob is not 0, environment smoke may call "
            "perform_action_by_llm."
        )
    try:
        asyncio.run(run_smoke(args))
    except Exception as exc:
        print("SMOKE_TEST_FAILED")
        print(f"error_type={type(exc).__name__}")
        print(f"error={exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
