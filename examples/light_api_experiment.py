#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(override=True)


DEFAULT_PROFILES = [
    {
        "name": "Maya Chen",
        "description": (
            "Urban planner who cares about climate adaptation, public transit, "
            "and practical civic experiments."
        ),
        "personality_traits": json.dumps(
            {
                "openness": 0.78,
                "conscientiousness": 0.72,
                "extroversion": 0.45,
                "agreeableness": 0.61,
            }
        ),
        "following_user_ids": "[1]",
        "previous_tweets": json.dumps(
            [
                "A shaded bus stop can change how a whole block feels in July.",
            ]
        ),
    },
    {
        "name": "Jordan Ellis",
        "description": (
            "Indie software builder who likes small tools, clear interfaces, "
            "and cautious takes on AI products."
        ),
        "personality_traits": json.dumps(
            {
                "openness": 0.66,
                "conscientiousness": 0.58,
                "extroversion": 0.55,
                "agreeableness": 0.5,
            }
        ),
        "following_user_ids": "[2]",
        "previous_tweets": json.dumps(
            [
                "The best productivity app is usually the one you keep using.",
            ]
        ),
    },
    {
        "name": "Riley Stone",
        "description": (
            "Community college media teacher who enjoys local journalism, "
            "student projects, and constructive debate."
        ),
        "personality_traits": json.dumps(
            {
                "openness": 0.7,
                "conscientiousness": 0.63,
                "extroversion": 0.68,
                "agreeableness": 0.74,
            }
        ),
        "following_user_ids": "[0]",
        "previous_tweets": json.dumps(
            [
                "Student newsletters are tiny laboratories for public trust.",
            ]
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny real-API OASIS experiment with generated agent content.",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--activation-prob", type=float, default=1.0)
    parser.add_argument(
        "--agent-variant",
        choices=["camel", "recagent_user"],
        default="camel",
    )
    parser.add_argument("--platform", choices=["twitter"], default="twitter")
    parser.add_argument("--work-dir", default="outputs/light_api_experiment")
    parser.add_argument("--profile-path", default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--simulation-concurrency", type=int, default=3)
    parser.add_argument("--max-content-length", type=int, default=280)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--chroma-path", default=None)
    return parser.parse_args()


def clip_probability(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_default_profiles(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "description",
        "personality_traits",
        "following_user_ids",
        "previous_tweets",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(DEFAULT_PROFILES)


def default_api_key(backend: str) -> str | None:
    if backend == "openai":
        return os.getenv("OPENAI_API_KEY")
    return os.getenv("DASHSCOPE_API_KEY")


def default_base_url(backend: str) -> str:
    if backend == "openai":
        return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def resolve_backend_config(args: argparse.Namespace) -> tuple[str, str, str]:
    backend = args.backend or "dashscope"
    if backend == "dashscope":
        model_name = (
            args.model_name
            or os.getenv("DASHSCOPE_MODEL_NAME")
            or "qwen-plus"
        )
        base_url = args.base_url or default_base_url(backend)
        return backend, model_name, base_url
    model_name = args.model_name or os.getenv("MODEL_NAME", "qwen-plus")
    base_url = args.base_url or default_base_url(backend)
    return backend, model_name, base_url


class UsageRecorder:
    def __init__(self, path: Path):
        self.path = path
        self.requests: list[dict[str, Any]] = []

    def record(self, payload: dict[str, Any]) -> None:
        self.requests.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                **payload,
            }
        )
        self.write()

    def summary(self) -> dict[str, Any]:
        totals = Counter()
        for item in self.requests:
            usage = item.get("request_usage") or {}
            totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            totals["completion_tokens"] += int(
                usage.get("completion_tokens") or 0
            )
            totals["total_tokens"] += int(usage.get("total_tokens") or 0)
        return {
            "request_count": len(self.requests),
            "prompt_tokens": totals["prompt_tokens"],
            "completion_tokens": totals["completion_tokens"],
            "total_tokens": totals["total_tokens"],
            "requests": self.requests,
        }

    def write(self) -> None:
        write_json(self.path, self.summary())


def build_camel_model(
    *,
    backend: str,
    model_name: str,
    base_url: str,
    temperature: float,
    top_p: float,
    timeout: float,
):
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    api_key = default_api_key(backend)
    if not api_key:
        raise ValueError(
            f"Missing API key for backend={backend}. Set DASHSCOPE_API_KEY "
            "or OPENAI_API_KEY in .env."
        )

    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=model_name,
        api_key=api_key,
        url=base_url,
        timeout=timeout,
        max_retries=1,
        model_config_dict={"temperature": temperature, "top_p": top_p},
    )


def install_usage_recorder(agent_graph, recorder: UsageRecorder) -> None:
    for _, agent in agent_graph.get_agents():
        if hasattr(agent, "on_request_usage"):
            agent.on_request_usage = recorder.record


def fetch_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts = {}
    for table in (
        "user",
        "post",
        "comment",
        "like",
        "dislike",
        "follow",
        "mention",
        "interaction",
        "rec",
        "trace",
    ):
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return counts


def fetch_trace_action_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return dict(
        conn.execute(
            "SELECT action, COUNT(*) FROM trace GROUP BY action ORDER BY action"
        ).fetchall()
    )


def fetch_samples(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    conn.row_factory = sqlite3.Row
    post_rows = conn.execute(
        "SELECT post_id, user_id, original_post_id, content, quote_content, "
        "created_at, num_likes, num_shares FROM post "
        "ORDER BY post_id DESC LIMIT 10"
    ).fetchall()
    comment_rows = conn.execute(
        "SELECT comment_id, post_id, user_id, content, created_at, num_likes "
        "FROM comment ORDER BY comment_id DESC LIMIT 10"
    ).fetchall()
    trace_rows = conn.execute(
        "SELECT user_id, created_at, action, info FROM trace "
        "ORDER BY created_at DESC LIMIT 15"
    ).fetchall()
    return {
        "posts": [dict(row) for row in post_rows],
        "comments": [dict(row) for row in comment_rows],
        "recent_trace": [dict(row) for row in trace_rows],
    }


def generated_text_count(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM post WHERE created_at > 0 AND "
        "((content IS NOT NULL AND content != '') OR "
        "(quote_content IS NOT NULL AND quote_content != ''))"
    ).fetchone()[0] + conn.execute(
        "SELECT COUNT(*) FROM comment WHERE created_at > 0 AND "
        "content IS NOT NULL AND content != ''"
    ).fetchone()[0]


async def run_experiment(args: argparse.Namespace) -> None:
    import oasis
    from oasis import ActionType, generate_twitter_agent_graph
    from oasis.environment.env_action import build_fixed_probability_actions
    from oasis.social_platform.typing import RecsysConfig, RecsysType

    if args.steps < 1:
        raise ValueError("--steps must be at least 1")

    work_dir = Path(args.work_dir)
    if args.fresh and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    db_path = work_dir / "light_api_experiment.db"
    profile_path = Path(args.profile_path) if args.profile_path else work_dir / "profiles.csv"
    usage_path = work_dir / "experiment_usage.json"
    summary_path = work_dir / "experiment_summary.json"
    chroma_path = (
        Path(args.chroma_path)
        if args.chroma_path
        else work_dir / "chroma_db"
    )

    if args.profile_path is None or not profile_path.exists():
        write_default_profiles(profile_path)

    if args.fresh and db_path.exists():
        db_path.unlink()

    backend, model_name, base_url = resolve_backend_config(args)
    os.environ["IS_VLLM"] = "false"
    os.environ["OASIS_DB_PATH"] = str(db_path)
    os.environ["OASIS_LLM_USAGE_LOG_PATH"] = str(usage_path)

    usage_recorder = UsageRecorder(usage_path)
    agent_model = build_camel_model(
        backend=backend,
        model_name=model_name,
        base_url=base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
    )

    client = None
    embedding_model_config = None
    if args.agent_variant == "recagent_user":
        import chromadb
        embedding_model = (
            args.embedding_model
            or os.getenv("EMBEDDING_MODEL")
            or "BAAI/bge-m3"
        )
        embedding_model_config = {
            "model_name_or_path": embedding_model,
            "device": args.embedding_device,
        }
        chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_path))

    agent_graph = await generate_twitter_agent_graph(
        profile_path=str(profile_path),
        model=agent_model,
        embedding_model_config=embedding_model_config,
        available_actions=ActionType.get_default_twitter_actions(),
        client=client,
        max_content_length=args.max_content_length,
        agent_variant=args.agent_variant,
        token_limit=6000 if args.agent_variant == "camel" else None,
    )
    install_usage_recorder(agent_graph, usage_recorder)

    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=str(db_path),
        semaphore=args.simulation_concurrency,
        max_content_length=args.max_content_length,
        recsys_config=RecsysConfig(
            recsys_type=RecsysType.RANDOM,
            refresh_rec_post_count=2,
            max_rec_post_len=5,
            following_post_count=2,
        ),
    )

    env_started = False
    total_activated = 0
    try:
        await env.reset()
        env_started = True
        agents_by_id = dict(env.agent_graph.get_agents())
        activation_prob = clip_probability(args.activation_prob)

        print(f"work_dir={work_dir}")
        print(f"profile_path={profile_path}")
        print(f"db_path={db_path}")
        print(f"llm_usage_path={usage_path}")
        print(f"backend={backend}")
        print(f"model_name={model_name}")
        print(f"base_url={base_url}")
        print(f"agent_variant={args.agent_variant}")
        print(f"activation_prob={activation_prob}")

        for step_index in range(args.steps):
            actions = build_fixed_probability_actions(agents_by_id, activation_prob)
            activated = len(actions)
            total_activated += activated
            print(f"step={step_index + 1}/{args.steps} activated_agents={activated}")
            await env.step(actions)

        conn = env.platform.db
        counts = fetch_counts(conn)
        trace_action_counts = fetch_trace_action_counts(conn)
        samples = fetch_samples(conn)
        usage_summary = usage_recorder.summary()
        text_count = generated_text_count(conn)
        agent_llm_calls_performed = usage_summary["request_count"] > 0

        if total_activated < 1:
            raise RuntimeError("No agents were activated; increase --activation-prob.")
        if not agent_llm_calls_performed:
            raise RuntimeError("No real agent LLM API calls were recorded.")

        summary = {
            "status": "ok",
            "config": {
                "platform": args.platform,
                "agent_variant": args.agent_variant,
                "steps": args.steps,
                "activation_prob": activation_prob,
                "backend": backend,
                "model_name": model_name,
                "base_url": base_url,
                "simulation_concurrency": args.simulation_concurrency,
            },
            "work_dir": str(work_dir),
            "db_path": str(db_path),
            "profile_path": str(profile_path),
            "llm_usage_path": str(usage_path),
            "agent_llm_calls_performed": agent_llm_calls_performed,
            "activated_agents_total": total_activated,
            "generated_text_count": text_count,
            "counts": counts,
            "trace_actions": trace_action_counts,
            "usage": usage_summary,
            "samples": samples,
        }
        write_json(summary_path, summary)
        usage_recorder.write()

        print("LIGHT_API_EXPERIMENT_OK")
        print(f"agent_llm_calls_performed={str(agent_llm_calls_performed).lower()}")
        print(f"activated_agents_total={total_activated}")
        print(f"llm_request_count={usage_summary['request_count']}")
        print(f"generated_text_count={text_count}")
        if text_count < 1:
            print("WARNING: no generated post/comment text was created.")
        print(f"summary_path={summary_path}")
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
    finally:
        if env_started:
            await env.close()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_experiment(args))
    except Exception as exc:
        print("LIGHT_API_EXPERIMENT_FAILED")
        print(f"error_type={type(exc).__name__}")
        print(f"error={exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
