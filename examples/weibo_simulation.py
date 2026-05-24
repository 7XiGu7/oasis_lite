#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import os
import time
from contextlib import contextmanager
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv

from checkpoint_utils import (
    CheckpointError,
    CheckpointManager,
    ensure_simulation_log_paths,
    ensure_simulation_tables,
    remove_path,
    restore_agent_states,
    restore_checkpoint_environment,
    restore_random_state,
    save_agent_states,
    save_progress,
)


load_dotenv(override=True)

DEFAULT_OUTPUT_DIR = Path(os.getenv("OASIS_OUTPUT_DIR", "outputs"))
DEFAULT_MODEL_DIR = Path(os.getenv("OASIS_MODEL_DIR", "models"))

@contextmanager
def timed_stage(name: str):
    start = time.perf_counter()
    print(f"[startup] {name} started")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[startup] {name} finished in {elapsed:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or resume a profile-only Weibo simulation.",
    )
    parser.add_argument("--profile-path", required=True, help="Agent profile CSV.")
    parser.add_argument("--steps", type=int, default=24, help="Simulation steps.")
    parser.add_argument(
        "--activation-prob",
        type=float,
        default=0.1,
        help="Fixed per-step activation probability for each eligible user.",
    )
    parser.add_argument(
        "--agent-variant",
        default="recagent_user",
        choices=["recagent_user", "camel"],
        help="Supported agent implementation.",
    )
    parser.add_argument("--variant", default="oasis_lite_weibo")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--chroma-path", default=None)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max-content-length", type=int, default=280)
    parser.add_argument("--simulation-concurrency", type=int, default=60)
    parser.add_argument("--memory-embedding-device", default="cuda:6")
    parser.add_argument(
        "--memory-embedding-model",
        default=os.getenv("EMBEDDING_MODEL", str(DEFAULT_MODEL_DIR / "bge-m3")),
    )
    parser.add_argument("--recsys-device", default="cuda:6")
    parser.add_argument(
        "--recsys-model",
        default=os.getenv("RECSYS_MODEL", str(DEFAULT_MODEL_DIR / "twhin-bert-base")),
    )
    parser.add_argument("--agent-context-token-limit", type=int, default=6000)
    parser.add_argument("--agent-message-window-size", type=int, default=None)
    return parser.parse_args()


def _default_api_key(backend: str) -> str | None:
    return {
        "dashscope": os.getenv("DASHSCOPE_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
    }[backend]


def _default_base_url(backend: str) -> str:
    if backend == "dashscope":
        return os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


def build_camel_model(backend: str, model_name: str, temperature: float, top_p: float):
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    api_key = _default_api_key(backend)
    if not api_key:
        raise ValueError(
            f"Missing API key for backend={backend}. Set the matching env var."
        )

    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=model_name,
        api_key=api_key,
        url=_default_base_url(backend),
        model_config_dict={"temperature": temperature, "top_p": top_p},
    )


def clip_probability(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


async def main():
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")

    is_camel_agent = args.agent_variant == "camel"
    profile_stem = Path(args.profile_path).stem
    run_variant = f"{args.variant}_{profile_stem}_{args.agent_variant}"
    db_path = args.db_path or str(DEFAULT_OUTPUT_DIR / "data" / f"{run_variant}.db")
    chromadb_path = (
        None
        if is_camel_agent
        else args.chroma_path
        or str(DEFAULT_OUTPUT_DIR / "data" / "chroma_db" / run_variant)
    )
    checkpoint_dir = (
        args.checkpoint_dir
        or str(DEFAULT_OUTPUT_DIR / "checkpoints" / run_variant)
    )

    os.environ["OASIS_DB_PATH"] = os.path.abspath(db_path)
    checkpoint = CheckpointManager(run_variant, checkpoint_dir)
    resume_state = None

    if args.fresh:
        remove_path(db_path)
        if chromadb_path is not None:
            remove_path(chromadb_path)
        remove_path(checkpoint_dir)
        print("Cleared previous simulation state.")
    elif checkpoint.has_checkpoint():
        with timed_stage("restore_to"):
            resume_state = checkpoint.restore_to(db_path, chromadb_path)
        if resume_state.get("total_steps") != args.steps:
            raise CheckpointError(
                "Checkpoint total_steps does not match --steps. Use --fresh."
            )
        restored_environment = restore_checkpoint_environment(resume_state)
        print(
            "Checkpoint detected, resuming from "
            f"step {resume_state['next_step']}/{args.steps}."
        )
        if restored_environment.get("OASIS_LLM_USAGE_LOG_PATH"):
            print(
                "Restored LLM usage log path: "
                f"{restored_environment['OASIS_LLM_USAGE_LOG_PATH']}"
            )
    else:
        remove_path(db_path)
        if chromadb_path is not None:
            remove_path(chromadb_path)
        remove_path(checkpoint_dir)
        print("No checkpoint detected; starting from scratch.")

    simulation_log_paths = ensure_simulation_log_paths(run_variant)
    print(f"LLM usage log: {simulation_log_paths['OASIS_LLM_USAGE_LOG_PATH']}")
    print(f"Run: {run_variant}")
    print(f"Profile path: {args.profile_path}")
    print(f"Activation probability: {clip_probability(args.activation_prob)}")
    print(f"Agent variant: {args.agent_variant}")
    print(f"Simulation concurrency: {args.simulation_concurrency}")

    with timed_stage("import oasis"):
        import oasis
        from oasis import ActionType, generate_weibo_agent_graph
        from oasis.environment.env_action import build_fixed_probability_actions
        from oasis.social_agent.llm_model import LLMModel

    available_actions = ActionType.get_default_weibo_actions()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    if chromadb_path is not None:
        Path(chromadb_path).parent.mkdir(parents=True, exist_ok=True)

    memory_embedding_model_config = None
    if not is_camel_agent:
        memory_embedding_model_config = {
            "model_name_or_path": args.memory_embedding_model,
            "device": args.memory_embedding_device,
        }

    client = None
    if not is_camel_agent:
        with timed_stage("import/chroma client"):
            import chromadb

            client = chromadb.PersistentClient(path=chromadb_path)

    if is_camel_agent:
        agent_model = build_camel_model(
            os.getenv("BACKEND", "openai"),
            os.getenv("MODEL_NAME", "qwen-plus"),
            args.temperature,
            args.top_p,
        )
    else:
        agent_model = LLMModel(
            temperature=args.temperature,
            top_p=args.top_p,
            max_memory_tokens=1000,
        )

    with timed_stage("agent graph"):
        agent_graph = await generate_weibo_agent_graph(
            profile_path=args.profile_path,
            model=agent_model,
            embedding_model_config=memory_embedding_model_config,
            available_actions=available_actions,
            client=client,
            max_content_length=args.max_content_length,
            agent_variant=args.agent_variant,
            token_limit=(
                args.agent_context_token_limit if is_camel_agent else None
            ),
            message_window_size=(
                args.agent_message_window_size if is_camel_agent else None
            ),
        )

    recsys_config = oasis.RecsysConfig.weibo(
        available_device=args.recsys_device,
        model_name_or_path=args.recsys_model,
        embedding_batch_size=256,
        openai_embedding_batch_size=100,
        coarse_filter_size=4000,
    )
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.WEIBO,
        recsys_config=recsys_config,
        max_content_length=args.max_content_length,
        database_path=db_path,
        semaphore=args.simulation_concurrency,
    )

    env_started = False
    try:
        with timed_stage("env.reset"):
            await env.reset(initialize_social_state=resume_state is None)
        env_started = True

        conn = env.platform.db
        ensure_simulation_tables(conn)
        agents_by_id = dict(env.agent_graph.get_agents())

        if resume_state is None:
            env.platform.sandbox_clock.time_step = 0
            save_agent_states(conn, run_variant, env)
            save_progress(conn, run_variant, args.steps, 0, 1, "ready")
            current_state = checkpoint.save_ready(
                env,
                db_path,
                args.steps,
                last_completed_step=0,
                next_step=1,
                chromadb_path=chromadb_path,
            )
        else:
            with timed_stage("restore_agent_states"):
                restore_agent_states(conn, run_variant, env)
            env.platform.sandbox_clock.time_step = int(
                resume_state["sandbox_clock_time_step"]
            )
            restore_random_state(resume_state.get("random_state"))
            current_state = resume_state

        start_step = int(current_state["next_step"])
        if start_step < 1 or start_step > args.steps + 1:
            raise CheckpointError("Checkpoint next_step is out of range.")

        for step_index in range(start_step - 1, args.steps):
            step_number = step_index + 1
            current_state = checkpoint.mark_running(current_state, step_number)
            save_progress(
                conn,
                run_variant,
                args.steps,
                last_completed_step=step_number - 1,
                next_step=step_number,
                status="running",
            )

            user_actions = build_fixed_probability_actions(
                agents_by_id,
                args.activation_prob,
            )

            print("=" * 50)
            print(f"Simulation progress: {step_number}/{args.steps}")
            print(f"Activated users: {len(user_actions)}")
            print("=" * 50)

            await env.step(user_actions)
            save_agent_states(conn, run_variant, env)
            save_progress(
                conn,
                run_variant,
                args.steps,
                last_completed_step=step_number,
                next_step=step_number + 1,
                status="ready",
            )
            current_state = checkpoint.save_ready(
                env,
                db_path,
                args.steps,
                last_completed_step=step_number,
                next_step=step_number + 1,
                chromadb_path=chromadb_path,
            )
    finally:
        if env_started:
            await env.close()


if __name__ == "__main__":
    asyncio.run(main())
