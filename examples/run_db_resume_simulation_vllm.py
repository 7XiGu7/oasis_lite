"""Resume an OASIS social simulation from an initialized SQLite DB.

The YAML config is the source of truth for paths, model settings, platform
settings, agent defaults, validation policy, and execution mode. The script
never writes to the input DB; it copies the input DB to the configured output
DB before smoke or run execution.
"""

#  python run_db_resume_simulation_vllm.py --config /home/yy/exp1/oasis_lite/examples/configs/db_resume_vllm.yaml

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import inspect
import io
import logging
import os
import random
import shutil
import sqlite3
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

CURRENT_DIR = Path(__file__).resolve().parent


def find_repo_root() -> Path:
    candidates = [CURRENT_DIR, Path.cwd().resolve()]
    for base in candidates:
        for path in (base, *base.parents):
            if (path / "oasis").is_dir() and (
                (path / "requirements.txt").exists()
                or (path / "pyproject.toml").exists()
                or (path / ".git").exists()
            ):
                return path
    return CURRENT_DIR.parent


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIRED_TABLES = ("user", "post", "follow", "trace", "rec", "top_news")
TIME_STEP_TABLES = ("trace", "post", "comment", "follow", "interaction")
STEP_DELTA_TABLES = ("trace", "post", "comment", "interaction")
VALID_MODES = {"validate", "smoke", "run"}
DEFAULT_ATTITUDE = 5.0
DEFAULT_TRAITS = {"openness": 0.5}
DEFAULT_TRAITS_INFO = "openness: 0.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume an OASIS simulation from an initialized DB.")
    parser.add_argument("--config",
                        required=True,
                        help="YAML config file for this resume task.")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Compatibility alias for execution.mode=validate. "
                        "The YAML file is not modified.")
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def require_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid config section: {section}")
    return value


def optional_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Invalid config section: {section}")
    return value


def resolve_repo_path(path: str | None, *, must_exist: bool = False) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (
        REPO_ROOT / candidate).resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def apply_logging_config(execution_config: dict[str, Any]) -> Path:
    log_dir = resolve_repo_path(execution_config.get("log_dir"))
    if log_dir is None:
        log_dir = (REPO_ROOT / "outputs" / "logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.environ["OASIS_LOG_DIR"] = str(log_dir)
    os.environ["OASIS_AGENT_LOG_PATH"] = str(
        log_dir / f"social.agent-{timestamp}.log")
    os.environ["OASIS_AGENT_ACTION_RECORD_PATH"] = str(
        log_dir / f"agent_action_records_{timestamp}.jsonl")
    os.environ["OASIS_AGENT_COGNITION_RECORD_PATH"] = str(
        log_dir / f"agent_cognition_records_{timestamp}.jsonl")
    os.environ["OASIS_ENV_LOG_PATH"] = str(log_dir / "oasis-env.log")
    os.environ["OASIS_PLATFORM_LOG_PATH"] = str(log_dir / "social-platform.log")
    os.environ["OASIS_LLM_LOG_PATH"] = str(
        log_dir / f"oasis.llm-{timestamp}.log")
    os.environ["OASIS_LLM_USAGE_LOG_PATH"] = str(
        log_dir / f"llm-usage-summary-{timestamp}.json")

    warnings_path = log_dir / "camel-warnings.log"
    camel_logger = logging.getLogger("camel")
    camel_logger.setLevel(logging.WARNING)
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")).resolve() == warnings_path
        for handler in camel_logger.handlers
    ):
        handler = logging.FileHandler(warnings_path, encoding="utf-8")
        handler.setLevel(logging.WARNING)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        camel_logger.addHandler(handler)
    return log_dir


def task_name(task_config: dict[str, Any]) -> str:
    raw_name = str(task_config.get("name") or "db_resume").strip()
    return raw_name or "db_resume"


def configured_paths(task_config: dict[str, Any],
                     paths_config: dict[str, Any]) -> tuple[Path, Path, Path]:
    input_db = resolve_repo_path(paths_config.get("input_db"), must_exist=True)
    if input_db is None:
        raise ValueError("paths.input_db is required.")

    output_dir = resolve_repo_path(
        paths_config.get("output_dir")
        or f"outputs/db_resume/{task_name(task_config)}")
    assert output_dir is not None
    output_db = resolve_repo_path(paths_config.get("output_db"))
    if output_db is None:
        output_db = (output_dir / "resume.db").resolve()
    return input_db, output_dir, output_db


def as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def as_int_or_none(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer or null.") from exc


def as_list(value: Any, field_name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    raise ValueError(f"{field_name} must be a list.")


def enum_value(enum_cls: Any, name: str, field_name: str) -> Any:
    text = str(name).strip()
    normalized = text.upper().replace("-", "_")
    for item in enum_cls:
        if item.name == normalized or item.value == text:
            return item
    valid = ", ".join(f"{item.name.lower()} ({item.value})" for item in enum_cls)
    raise ValueError(f"Unsupported {field_name}: {name}. Valid values: {valid}")


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table, )).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def table_count(conn: sqlite3.Connection, table: str) -> int | None:
    if not table_exists(conn, table):
        return None
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def db_count_snapshot(db_path: Path,
                      tables: Iterable[str] = STEP_DELTA_TABLES
                      ) -> dict[str, int]:
    with sqlite3.connect(db_path) as conn:
        return {
            table: (table_count(conn, table) or 0)
            for table in tables
        }


def count_delta(before: dict[str, int],
                after: dict[str, int]) -> dict[str, int]:
    return {
        table: after.get(table, 0) - before.get(table, 0)
        for table in before
    }


def format_count_delta(delta: dict[str, int]) -> str:
    return ", ".join(f"{table}+{count}" for table, count in delta.items())


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def load_dotenv_file(dotenv_path: Path, *, override: bool = False) -> bool:
    if not dotenv_path.exists():
        return False
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and (override or key not in os.environ):
            os.environ[key] = value
    return True


def install_oasis_show_db_stub() -> None:
    """Avoid importing oasis.testing.show_db, which hardcodes a log path."""
    if "oasis.testing.show_db" in sys.modules:
        return

    testing_pkg = sys.modules.setdefault("oasis.testing",
                                         types.ModuleType("oasis.testing"))
    testing_pkg.__path__ = [str(REPO_ROOT / "oasis/testing")]

    show_db_module = types.ModuleType("oasis.testing.show_db")

    def print_db_contents(db_file: str) -> None:
        print(f"print_db_contents is disabled in resume mode: {db_file}")

    show_db_module.print_db_contents = print_db_contents
    sys.modules["oasis.testing.show_db"] = show_db_module


def infer_next_time_step(conn: sqlite3.Connection) -> int:
    values: list[int] = []
    for table in TIME_STEP_TABLES:
        if not table_exists(conn, table):
            continue
        columns = table_columns(conn, table)
        if "created_at" not in columns:
            continue
        value = conn.execute(f"SELECT MAX(created_at) FROM {table}").fetchone()[0]
        if value is None:
            continue
        try:
            values.append(int(float(value)))
        except (TypeError, ValueError):
            continue
    return max(values) + 1 if values else 0


def validate_required_tables(conn: sqlite3.Connection) -> None:
    missing = [table for table in REQUIRED_TABLES if not table_exists(conn, table)]
    if missing:
        raise ValueError(f"Input DB missing required tables: {missing}")


def validate_user_mapping(conn: sqlite3.Connection, *,
                          require_user_agent_id_equal: bool,
                          require_contiguous_agent_ids: bool) -> None:
    if require_user_agent_id_equal:
        mismatch = conn.execute(
            "SELECT COUNT(*) FROM user WHERE user_id IS NULL OR agent_id IS NULL "
            "OR user_id != agent_id").fetchone()[0]
        if mismatch:
            raise ValueError(
                "Current OASIS resume path requires user.user_id == "
                "user.agent_id; mismatched rows: "
                f"{mismatch}")

    if not require_contiguous_agent_ids:
        return

    rows = conn.execute("SELECT agent_id FROM user ORDER BY agent_id").fetchall()
    agent_ids = [int(row[0]) for row in rows]
    expected = list(range(len(agent_ids)))
    if agent_ids != expected:
        raise ValueError(
            "Current OASIS graph backend expects contiguous agent IDs starting "
            f"at 0. Found first IDs {agent_ids[:10]} and count "
            f"{len(agent_ids)}.")


def validate_follow_refs(conn: sqlite3.Connection) -> None:
    bad_refs = conn.execute("""
        SELECT COUNT(*)
        FROM follow
        WHERE follower_id IS NULL
           OR followee_id IS NULL
           OR follower_id NOT IN (SELECT user_id FROM user)
           OR followee_id NOT IN (SELECT user_id FROM user)
    """).fetchone()[0]
    if bad_refs:
        raise ValueError(f"follow table contains missing user refs: {bad_refs}")


def validate_runtime_schema(conn: sqlite3.Connection) -> None:
    required_columns = {
        "post": {"post_attr"},
        "interaction": {"interaction_type"},
    }
    missing: list[str] = []
    for table, columns in required_columns.items():
        if not table_exists(conn, table):
            missing.append(table)
            continue
        existing = table_columns(conn, table)
        missing.extend(
            f"{table}.{column}" for column in sorted(columns - existing))
    if missing:
        raise ValueError(
            "Input DB schema is not compatible with the current platform code; "
            f"missing: {missing}")


def summarize_db(db_path: Path,
                 *,
                 validate_db: bool,
                 require_user_agent_id_equal: bool,
                 require_contiguous_agent_ids: bool) -> dict[str, Any]:
    with connect_readonly(db_path) as conn:
        if validate_db:
            validate_required_tables(conn)
            validate_runtime_schema(conn)
            validate_user_mapping(
                conn,
                require_user_agent_id_equal=require_user_agent_id_equal,
                require_contiguous_agent_ids=require_contiguous_agent_ids,
            )
            validate_follow_refs(conn)

        counts = {
            table: table_count(conn, table)
            for table in REQUIRED_TABLES + ("comment", "interaction",
                                            "agent_state")
        }
        next_step = infer_next_time_step(conn)
    return {"counts": counts, "next_step": next_step}


def prepare_output_db(input_db: Path, output_db: Path, overwrite: bool) -> None:
    if not input_db.exists():
        raise FileNotFoundError(f"Input DB does not exist: {input_db}")
    if input_db.resolve() == output_db.resolve():
        raise ValueError(
            "paths.input_db and paths.output_db must be different files. "
            "Resume mode keeps input DB read-only and writes to output DB.")
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output DB already exists: {output_db}. Set "
                "paths.overwrite_output: true to replace it.")
        output_db.unlink()
    shutil.copy2(input_db, output_db)


def ensure_agent_state(conn: sqlite3.Connection, topic: str,
                       updated_at: int) -> int:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_state (
            agent_id INTEGER PRIMARY KEY,
            topic TEXT,
            attitude REAL,
            updated_at INTEGER
        )
    """)
    rows = conn.execute("SELECT agent_id FROM user ORDER BY agent_id").fetchall()
    inserted = 0
    for (agent_id, ) in rows:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO agent_state "
            "(agent_id, topic, attitude, updated_at) VALUES (?, ?, ?, ?)",
            (int(agent_id), topic, DEFAULT_ATTITUDE, updated_at),
        )
        inserted += cursor.rowcount
    conn.execute(
        "UPDATE agent_state SET "
        "topic=COALESCE(topic, ?), "
        "attitude=COALESCE(attitude, ?), "
        "updated_at=COALESCE(updated_at, ?) "
        "WHERE agent_id IN (SELECT agent_id FROM user)",
        (topic, DEFAULT_ATTITUDE, updated_at),
    )
    conn.commit()
    return inserted


def load_agent_states(conn: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    if not table_exists(conn, "agent_state"):
        return {}
    rows = conn.execute(
        "SELECT agent_id, topic, attitude, updated_at FROM agent_state"
    ).fetchall()
    states: dict[int, dict[str, Any]] = {}
    for agent_id, topic, attitude, updated_at in rows:
        states[int(agent_id)] = {
            "topic": topic,
            "attitude": attitude,
            "updated_at": updated_at,
        }
    return states


def persist_agent_states(db_path: Path, env: Any, updated_at: int,
                         default_topic: str) -> int:
    rows = []
    for agent_id, agent in env.agent_graph.get_agents():
        other_info = agent.user_info.profile.setdefault("other_info", {})
        rows.append((
            int(agent_id),
            other_info.get("topic") or default_topic,
            other_info.get("attitude", DEFAULT_ATTITUDE),
            int(updated_at),
        ))

    sql = (
        "INSERT OR REPLACE INTO agent_state "
        "(agent_id, topic, attitude, updated_at) VALUES (?, ?, ?, ?)"
    )
    platform_conn = getattr(getattr(env, "platform", None), "db", None)
    if platform_conn is not None:
        platform_conn.executemany(sql, rows)
        platform_conn.commit()
    else:
        with sqlite3.connect(db_path) as conn:
            conn.executemany(sql, rows)
            conn.commit()
    return len(rows)


def apply_model_env(model_config: dict[str, Any]) -> str | None:
    platform_name = str(model_config.get("platform", "vllm")).lower()
    if platform_name == "vllm":
        return None

    defaults = {
        "qwen": ("QWEN_API_KEY", "DASHSCOPE_API_KEY"),
        "openai": ("OPENAI_API_KEY", None),
        "deepseek": ("DEEPSEEK_API_KEY", None),
    }
    default_api_key_env, default_fallback = defaults.get(platform_name,
                                                         (None, None))
    api_key_env = model_config.get("api_key_env") or default_api_key_env
    fallback_api_key_env = (
        model_config.get("fallback_api_key_env") or default_fallback)
    if not api_key_env:
        return None
    if os.environ.get(api_key_env):
        return None
    if fallback_api_key_env and os.environ.get(fallback_api_key_env):
        os.environ[api_key_env] = os.environ[fallback_api_key_env]
        return (
            f"Using {fallback_api_key_env} as process-local fallback for "
            f"{api_key_env}.")
    raise ValueError(
        f"Missing model API key environment variable: {api_key_env}"
        + (f" (fallback {fallback_api_key_env} is also empty)."
           if fallback_api_key_env else "."))


def configured_model_api_key(model_config: dict[str, Any]) -> str | None:
    api_key = model_config.get("api_key")
    if api_key not in (None, ""):
        return str(api_key)

    api_key_env = model_config.get("api_key_env")
    if api_key_env:
        return os.environ.get(str(api_key_env))
    return None


def model_factory_runtime_kwargs(model_config: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for key in ("timeout", "max_retries"):
        if key in model_config and model_config[key] is not None:
            kwargs[key] = model_config[key]
    return kwargs


def build_model(model_config: dict[str, Any]):
    apply_model_env(model_config)

    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType

    platform_name = str(model_config.get("platform", "vllm")).lower()
    type_name = model_config.get("type")
    url = model_config.get("url") or model_config.get("base_url")
    extra_config = model_config.get("config") or {}
    if not isinstance(extra_config, dict):
        raise ValueError("model.config must be a mapping.")

    if platform_name == "vllm":
        model_type = type_name or os.getenv("MODEL_NAME")
        if not model_type:
            raise ValueError(
                "model.type is required for model.platform: vllm. Use the "
                "same name as vLLM --served-model-name, or the model path if "
                "you did not set --served-model-name.")
        base_url = (
            url
            or os.getenv("VLLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "http://127.0.0.1:8000/v1"
        )
        os.environ["IS_VLLM"] = "true"
        os.environ["MODEL_NAME"] = str(model_type)
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_type,
            api_key=configured_model_api_key(model_config) or "EMPTY",
            url=base_url,
            model_config_dict=extra_config,
            **model_factory_runtime_kwargs(model_config),
        )

    platform_map = {
        "qwen": ModelPlatformType.QWEN,
        "openai": ModelPlatformType.OPENAI,
        "deepseek": ModelPlatformType.DEEPSEEK,
        "vllm": ModelPlatformType.VLLM,
    }
    default_types = {
        "qwen": ModelType.QWEN_PLUS,
        "openai": ModelType.GPT_4O_MINI,
        "deepseek": "deepseek-chat",
    }
    if platform_name not in platform_map:
        raise ValueError(f"Unsupported model.platform: {platform_name}")

    kwargs: dict[str, Any] = {
        "model_platform": platform_map[platform_name],
        "model_type": type_name or default_types.get(platform_name),
    }
    if url:
        kwargs["url"] = url
    if extra_config:
        kwargs["model_config_dict"] = extra_config
    return ModelFactory.create(**kwargs)


def twitter_actions(action_names: list[str] | None):
    from oasis.social_platform.typing import ActionType

    if not action_names:
        return ActionType.get_default_twitter_actions()
    return [
        enum_value(ActionType, action_name, "agents.available_actions")
        for action_name in action_names
    ]


def validate_action_alignment(action_names: list[Any]) -> None:
    from oasis.social_agent.agent_action import SocialAction
    from oasis.social_platform.platform import Platform
    from oasis.social_platform.typing import ActionType

    enum_values = {action.value for action in ActionType}
    tool_names = set(SocialAction._action_function_names)
    missing_from_tools: list[str] = []
    missing_from_platform: list[str] = []
    invalid_values: list[str] = []

    for action in action_names:
        action_name = action.value if isinstance(action, ActionType) else str(action)
        if action_name not in enum_values:
            invalid_values.append(action_name)
        if action_name not in tool_names:
            missing_from_tools.append(action_name)
        if not hasattr(Platform, action_name):
            missing_from_platform.append(action_name)

    errors = []
    if invalid_values:
        errors.append(f"not in ActionType: {sorted(set(invalid_values))}")
    if missing_from_tools:
        errors.append(
            "missing SocialAction tool method/schema: "
            f"{sorted(set(missing_from_tools))}")
    if missing_from_platform:
        errors.append(
            "missing Platform execution method: "
            f"{sorted(set(missing_from_platform))}")
    if errors:
        raise ValueError(
            "agents.available_actions is not aligned across ActionType, "
            "SocialAction, and Platform: " + "; ".join(errors))


def format_traits_info(traits: Any) -> str:
    if isinstance(traits, dict):
        return "; ".join(f"{key}: {value}" for key, value in traits.items())
    return str(traits or "")


def profile_defaults(agents_config: dict[str, Any]) -> dict[str, Any]:
    default_profile = agents_config.get("default_profile") or {}
    if not isinstance(default_profile, dict):
        raise ValueError("agents.default_profile must be a mapping.")
    traits = copy.deepcopy(default_profile.get("traits", DEFAULT_TRAITS))
    traits_info = default_profile.get("traits_info") or format_traits_info(traits)
    return {
        "traits": traits,
        "traits_info": traits_info or DEFAULT_TRAITS_INFO,
        "previous_tweets": as_list(default_profile.get("previous_tweets"),
                                   "agents.default_profile.previous_tweets"),
        "activity_level": as_list(default_profile.get("activity_level"),
                                  "agents.default_profile.activity_level"),
        "is_robot": as_bool(default_profile.get("is_robot"), default=False),
    }


def build_agent_graph_from_db(db_path: Path, model: Any, task_config: dict[str,
                                                                          Any],
                              agents_config: dict[str, Any]):
    from oasis.social_agent.agent import SocialAgent
    from oasis.social_agent.agent_graph import AgentGraph
    from oasis.social_platform.config import UserInfo

    topic = str(task_config.get("topic") or "the current discussion")
    recsys_type = str(agents_config.get("recsys_type") or "twitter")
    available_actions = twitter_actions(agents_config.get("available_actions"))
    defaults = profile_defaults(agents_config)
    graph = AgentGraph()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        states = load_agent_states(conn)
        users = conn.execute("""
            SELECT user_id, agent_id, user_name, name, bio
            FROM user
            ORDER BY agent_id
        """).fetchall()

    for row in users:
        agent_id = int(row["agent_id"])
        state = states.get(agent_id, {})
        bio = row["bio"] or ""
        profile = {
            "nodes": [],
            "edges": [],
            "other_info": {
                "user_profile": bio,
                "traits": copy.deepcopy(defaults["traits"]),
                "traits_info": defaults["traits_info"],
                "topic": state.get("topic") or topic,
                "attitude": state.get("attitude", DEFAULT_ATTITUDE),
                "is_robot": defaults["is_robot"],
                "following_id_list": [],
                "following_name_list": [],
                "previous_tweets": copy.deepcopy(defaults["previous_tweets"]),
                "activity_level": copy.deepcopy(defaults["activity_level"]),
            },
        }
        user_info = UserInfo(
            user_name=row["user_name"],
            name=row["name"] or row["user_name"],
            description=bio,
            profile=profile,
            recsys_type=recsys_type,
        )
        agent = SocialAgent(
            agent_id=agent_id,
            user_info=user_info,
            model=model,
            agent_graph=graph,
            available_actions=available_actions,
        )
        graph.add_agent(agent)
    return graph


def sync_graph_from_db(db_path: Path, graph: Any) -> int:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT follower_id, followee_id FROM follow").fetchall()
        names = {
            int(user_id): str(name or user_name or f"user_{user_id}")
            for user_id, user_name, name in conn.execute(
                "SELECT user_id, user_name, name FROM user")
        }

    follow_map: dict[int, list[int]] = {}
    edge_count = 0
    for follower_id, followee_id in rows:
        follower = int(follower_id)
        followee = int(followee_id)
        if follower not in graph.agent_mappings:
            continue
        if followee not in graph.agent_mappings:
            continue
        graph.add_edge(follower, followee)
        follow_map.setdefault(follower, []).append(followee)
        edge_count += 1

    for agent_id, agent in graph.get_agents():
        following_ids = follow_map.get(int(agent_id), [])
        other_info = agent.user_info.profile.setdefault("other_info", {})
        other_info["following_id_list"] = following_ids
        other_info["following_name_list"] = [
            names.get(fid, f"user_{fid}") for fid in following_ids
        ]
    return edge_count


def connect_agents_to_env_channel(env: Any) -> None:
    for _, agent in env.agent_graph.get_agents():
        agent.channel = env.channel
        if hasattr(agent, "action"):
            agent.action.channel = env.channel
        if hasattr(agent, "env") and hasattr(agent.env, "action"):
            agent.env.action.channel = env.channel
        action_tools = getattr(agent, "action_tools", None)
        if action_tools is not None and hasattr(agent, "action"):
            action_names = [
                tool.get_function_name()
                for tool in action_tools
                if hasattr(tool, "get_function_name")
            ]
            agent.action_tools = agent.action.get_function_tool_list_for_actions(
                action_names)


def selected_agents(env: Any,
                    agent_limit: int | None) -> Iterable[tuple[int, object]]:
    agents = env.agent_graph.get_agents()
    if agent_limit is not None:
        return agents[:agent_limit]
    return agents


def llm_action_kwargs(simulation_config: dict[str, Any]) -> dict[str, Any]:
    value = simulation_config.get("llm_action_kwargs") or {}
    if not isinstance(value, dict):
        raise ValueError("simulation.llm_action_kwargs must be a mapping.")
    return dict(value)


def make_llm_action(simulation_config: dict[str, Any]):
    from oasis.environment.env_action import LLMAction

    kwargs = llm_action_kwargs(simulation_config)
    if not kwargs:
        return LLMAction()
    signature = inspect.signature(LLMAction)
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return LLMAction(**accepted)


def build_llm_actions(env: Any, simulation_config: dict[str, Any]):
    activation = str(simulation_config.get("activation", "random")).lower()
    probability = float(simulation_config.get("probability", 0.1))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("simulation.probability must be between 0 and 1.")
    agent_limit = as_int_or_none(simulation_config.get("agent_limit"),
                                 "simulation.agent_limit")
    agents = list(selected_agents(env, agent_limit))

    if activation == "all":
        active_agents = [agent for _, agent in agents]
    elif activation == "random":
        active_agents = [agent for _, agent in agents if random.random() <= probability]
    else:
        raise ValueError("simulation.activation must be 'random' or 'all'.")
    return {agent: make_llm_action(simulation_config) for agent in active_agents}


def close_platform(platform: Any) -> None:
    cursor = getattr(platform, "db_cursor", None)
    conn = getattr(platform, "db", None)
    if cursor is not None:
        cursor.close()
    if conn is not None:
        conn.close()


def platform_allowed_kwargs() -> set[str]:
    from oasis.social_platform.platform import Platform

    signature = inspect.signature(Platform.__init__)
    return set(signature.parameters) - {"self", "db_path", "channel"}


def validate_platform_kwargs(platform_config: dict[str, Any]) -> None:
    kwargs = platform_config.get("kwargs") or {}
    if not isinstance(kwargs, dict):
        raise ValueError("platform.kwargs must be a mapping.")
    unsupported = sorted(set(kwargs) - platform_allowed_kwargs())
    reserved = sorted(set(kwargs) & {"self", "db_path", "channel"})
    if unsupported or reserved:
        allowed = ", ".join(sorted(platform_allowed_kwargs()))
        raise ValueError(
            "Unsupported platform.kwargs keys: "
            f"{unsupported + reserved}. Supported keys: {allowed}")


def validate_llm_action_kwargs(simulation_config: dict[str, Any]) -> None:
    from oasis.environment.env_action import LLMAction

    kwargs = llm_action_kwargs(simulation_config)
    if not kwargs:
        return
    signature = inspect.signature(LLMAction)
    unsupported = sorted(set(kwargs) - set(signature.parameters))
    if unsupported:
        supported = ", ".join(signature.parameters) or "(none)"
        raise ValueError(
            "Unsupported simulation.llm_action_kwargs keys for current "
            f"LLMAction: {unsupported}. Supported keys: {supported}")


def validate_simulation_config(simulation_config: dict[str, Any]) -> None:
    steps = int(simulation_config.get("steps", 0))
    if steps < 0:
        raise ValueError("simulation.steps must be >= 0.")

    semaphore = int(simulation_config.get("semaphore", 20))
    if semaphore <= 0:
        raise ValueError("simulation.semaphore must be > 0.")

    agent_limit = as_int_or_none(simulation_config.get("agent_limit"),
                                 "simulation.agent_limit")
    if agent_limit is not None and agent_limit < 0:
        raise ValueError("simulation.agent_limit must be >= 0 or null.")

    activation = str(simulation_config.get("activation", "random")).lower()
    if activation not in {"all", "random"}:
        raise ValueError("simulation.activation must be 'random' or 'all'.")

    probability = float(simulation_config.get("probability", 0.1))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("simulation.probability must be between 0 and 1.")


def tool_call_result(tool_call: Any) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get("result")
    return getattr(tool_call, "result", None)


def tool_call_name(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("tool_name") or tool_call.get("name") or "")
    return str(
        getattr(tool_call, "tool_name", None)
        or getattr(tool_call, "name", None)
        or "")


def tool_result_failed(result: Any) -> bool:
    if isinstance(result, dict):
        return result.get("success") is False
    if isinstance(result, str):
        return result.startswith("Tool execution failed:")
    return False


def summarize_step_results(results: list[Any]) -> dict[str, int]:
    summary = {
        "api_or_model_failures": 0,
        "no_tool_call": 0,
        "tool_execution_failures": 0,
        "tool_calls": 0,
    }
    for result in results:
        if isinstance(result, BaseException):
            summary["api_or_model_failures"] += 1
            continue
        if isinstance(result, dict) and result.get("success") is False:
            summary["no_tool_call"] += 1
            continue

        info = getattr(result, "info", None)
        tool_calls = []
        if isinstance(info, dict):
            tool_calls = info.get("tool_calls") or []
        if not tool_calls:
            summary["no_tool_call"] += 1
            continue

        summary["tool_calls"] += len(tool_calls)
        for tool_call in tool_calls:
            if tool_result_failed(tool_call_result(tool_call)):
                summary["tool_execution_failures"] += 1
    return summary


def format_step_result_summary(summary: dict[str, int]) -> str:
    return (
        f"tool_calls={summary['tool_calls']}, "
        f"api_or_model_failures={summary['api_or_model_failures']}, "
        f"tool_execution_failures={summary['tool_execution_failures']}, "
        f"no_tool_call={summary['no_tool_call']}"
    )


def validate_agents_config(agents_config: dict[str, Any]) -> None:
    configured_actions = agents_config.get("available_actions")
    if configured_actions:
        actions = as_list(configured_actions, "agents.available_actions")
    else:
        actions = twitter_actions(None)
    validate_action_alignment(actions)
    profile_defaults(agents_config)


def build_platform(output_db: Path, platform_config: dict[str, Any]) -> Any:
    from oasis.social_platform.channel import Channel
    from oasis.social_platform.platform import Platform
    from oasis.social_platform.typing import DefaultPlatformType

    platform_type = str(platform_config.get("type", "twitter")).lower()
    if platform_type != "twitter":
        raise ValueError(
            "This DB resume script currently supports platform.type: twitter.")

    validate_platform_kwargs(platform_config)
    platform_kwargs = dict(platform_config.get("kwargs") or {})
    defaults = {
        "recsys_type": "twhin-bert",
        "refresh_rec_post_count": 2,
        "max_rec_post_len": 2,
        "following_post_count": 3,
        "time_top_n_comments": 3,
    }
    defaults.update(platform_kwargs)
    enum_value(DefaultPlatformType, platform_type, "platform.type")
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        return Platform(db_path=str(output_db), channel=Channel(), **defaults)


def platform_semaphore(simulation_config: dict[str, Any]) -> int:
    return int(simulation_config.get("semaphore", 20))


def print_summary(config_path: Path, mode: str, input_db: Path, output_db: Path,
                  summary: dict[str, Any], start_step: int | None) -> None:
    counts = summary["counts"]
    next_step = summary["next_step"]
    effective_step = next_step if start_step is None else start_step
    print(f"Config: {config_path}")
    print(f"Mode: {mode}")
    print(f"Input DB: {input_db}")
    print(f"Output DB: {output_db}")
    print("Table counts:")
    for table in ("user", "post", "follow", "trace", "rec", "top_news",
                  "comment", "interaction", "agent_state"):
        value = counts.get(table)
        if value is not None:
            print(f"  {table}: {value}")
    print(f"Inferred next step: {next_step}")
    print(f"Effective start step: {effective_step}")


def validate_config_against_runtime(config: dict[str, Any]) -> str | None:
    execution_config = require_section(config, "execution")
    mode = str(execution_config.get("mode", "validate")).lower()
    if mode not in VALID_MODES:
        raise ValueError(f"execution.mode must be one of {sorted(VALID_MODES)}.")

    platform_config = require_section(config, "platform")
    platform_type = str(platform_config.get("type", "twitter")).lower()
    if platform_type != "twitter":
        raise ValueError("Only platform.type: twitter is currently supported.")
    validate_platform_kwargs(platform_config)

    validate_agents_config(require_section(config, "agents"))
    simulation_config = require_section(config, "simulation")
    validate_simulation_config(simulation_config)
    validate_llm_action_kwargs(simulation_config)
    return apply_model_env(require_section(config, "model"))


async def execute_steps(output_db: Path, task_config: dict[str, Any],
                        model_config: dict[str, Any],
                        platform_config: dict[str, Any],
                        agents_config: dict[str, Any],
                        simulation_config: dict[str, Any],
                        effective_step: int, topic: str) -> None:
    install_oasis_show_db_stub()
    from oasis.environment.make import make

    os.environ["OASIS_DB_PATH"] = str(output_db)
    model = build_model(model_config)
    agent_graph = build_agent_graph_from_db(output_db, model, task_config,
                                            agents_config)
    edge_count = sync_graph_from_db(output_db, agent_graph)
    platform = build_platform(output_db, platform_config)

    env = make(
        agent_graph=agent_graph,
        platform=platform,
        database_path=str(output_db),
        semaphore=platform_semaphore(simulation_config),
    )
    env_started = False
    try:
        connect_agents_to_env_channel(env)
        env.platform.sandbox_clock.time_step = effective_step
        env.platform_task = asyncio.create_task(env.platform.running())
        env_started = True
        print(f"Rebuilt agents: {len(env.agent_graph.get_agents())}")
        print(f"Synced follow edges: {edge_count}")
        print(f"Starting resume simulation at time step: {effective_step}")

        steps = int(simulation_config.get("steps", 0))
        total_active_agents = 0
        total_result_summary = {
            "api_or_model_failures": 0,
            "no_tool_call": 0,
            "tool_execution_failures": 0,
            "tool_calls": 0,
        }
        for step_idx in range(steps):
            current_step = env.platform.sandbox_clock.time_step
            actions = build_llm_actions(env, simulation_config)
            active_agents = len(actions)
            total_active_agents += active_agents
            before_counts = db_count_snapshot(output_db)
            print(f"Running step {step_idx + 1}/{steps} at time step "
                  f"{current_step}; active agents: {active_agents}.")
            results = await env.step(actions)
            after_counts = db_count_snapshot(output_db)
            delta = count_delta(before_counts, after_counts)
            result_summary = summarize_step_results(results or [])
            for key, value in result_summary.items():
                total_result_summary[key] += value
            print(f"Step {step_idx + 1}/{steps} DB delta: "
                  f"{format_count_delta(delta)}.")
            print(f"Step {step_idx + 1}/{steps} result summary: "
                  f"{format_step_result_summary(result_summary)}.")
            persist_agent_states(output_db, env,
                                 env.platform.sandbox_clock.time_step, topic)
        print(f"Resume run summary: active agent executions="
              f"{total_active_agents}; "
              f"{format_step_result_summary(total_result_summary)}.")
    finally:
        if env_started:
            await env.close()


def smoke_simulation_config(simulation_config: dict[str, Any],
                            smoke_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(simulation_config)
    if "steps" in smoke_config:
        config["steps"] = smoke_config["steps"]
    if "agent_limit" in smoke_config:
        config["agent_limit"] = smoke_config["agent_limit"]
    return config


async def run_resume(config: dict[str, Any], config_path: Path,
                     dry_run: bool) -> None:
    config = copy.deepcopy(config)
    execution_config = require_section(config, "execution")
    if dry_run:
        execution_config["mode"] = "validate"

    if as_bool(execution_config.get("load_dotenv"), default=True):
        if load_dotenv_file(REPO_ROOT / ".env"):
            print(f"Loaded env file: {REPO_ROOT / '.env'}")
    log_dir = apply_logging_config(execution_config)
    print(f"Log dir: {log_dir}")

    task_config = require_section(config, "task")
    paths_config = require_section(config, "paths")
    platform_config = require_section(config, "platform")
    agents_config = require_section(config, "agents")
    model_config = require_section(config, "model")
    simulation_config = require_section(config, "simulation")
    resume_config = require_section(config, "resume")
    smoke_test_config = optional_section(config, "smoke_test")

    mode = str(execution_config.get("mode", "validate")).lower()
    input_db, _output_dir, output_db = configured_paths(task_config, paths_config)
    overwrite_output = as_bool(paths_config.get("overwrite_output"),
                               default=False)
    start_step = as_int_or_none(resume_config.get("start_step"),
                                "resume.start_step")
    validate_db = as_bool(resume_config.get("validate_db"), default=True)
    require_equal = as_bool(
        resume_config.get("require_user_agent_id_equal"), default=True)
    require_contiguous = as_bool(
        resume_config.get("require_contiguous_agent_ids"), default=True)
    initialize_agent_state = as_bool(
        resume_config.get("initialize_agent_state"), default=True)
    stop_on_error = as_bool(execution_config.get("stop_on_validation_error"),
                            default=True)

    validation_errors: list[str] = []
    try:
        env_message = validate_config_against_runtime(config)
        if env_message:
            print(env_message)
    except Exception as exc:  # noqa: BLE001
        validation_errors.append(str(exc))

    input_summary: dict[str, Any] | None = None
    try:
        input_summary = summarize_db(
            input_db,
            validate_db=validate_db,
            require_user_agent_id_equal=require_equal,
            require_contiguous_agent_ids=require_contiguous,
        )
    except Exception as exc:  # noqa: BLE001
        validation_errors.append(str(exc))

    if input_db.resolve() == output_db.resolve():
        validation_errors.append(
            "paths.input_db and paths.output_db must be different files.")

    if validation_errors:
        print("Validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
        if stop_on_error:
            raise ValueError("Validation failed.")
        if mode == "validate":
            print("Validation finished with errors; no output DB was written.")
            return

    if input_summary is None:
        raise ValueError("Input DB summary is unavailable.")

    print_summary(config_path, mode, input_db, output_db, input_summary,
                  start_step)

    if mode == "validate":
        print("Validation complete. No output DB was written.")
        return

    prepare_output_db(input_db, output_db, overwrite_output)
    effective_step = input_summary["next_step"] if start_step is None else start_step
    topic = str(task_config.get("topic") or "the current discussion")
    if initialize_agent_state:
        with sqlite3.connect(output_db) as conn:
            inserted_states = ensure_agent_state(conn, topic, effective_step)
        print(f"Copied DB to: {output_db}")
        print(f"agent_state initialized rows: {inserted_states}")
    else:
        print(f"Copied DB to: {output_db}")
        print("agent_state initialization skipped by resume config.")

    if mode == "smoke":
        if as_bool(smoke_test_config.get("build_platform"), default=True):
            install_oasis_show_db_stub()
            platform = build_platform(output_db, platform_config)
            close_platform(platform)
            print("Smoke test built and closed platform from config.")
        if not as_bool(smoke_test_config.get("run_steps"), default=False):
            print("Smoke test complete. No LLM steps were run.")
            return
        smoke_config = smoke_simulation_config(simulation_config,
                                               smoke_test_config)
        await execute_steps(output_db, task_config, model_config,
                            platform_config, agents_config, smoke_config,
                            effective_step, topic)
        print(f"Smoke steps complete. Results appended to: {output_db}")
        return

    if mode == "run":
        steps = int(simulation_config.get("steps", 0))
        if steps < 0:
            raise ValueError("simulation.steps must be >= 0.")
        if steps == 0:
            print("simulation.steps is 0; DB copy and agent_state setup complete.")
            return
        await execute_steps(output_db, task_config, model_config,
                            platform_config, agents_config, simulation_config,
                            effective_step, topic)
        print(f"Resume complete. Results appended to: {output_db}")
        return

    raise ValueError(f"Unsupported execution.mode: {mode}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    asyncio.run(run_resume(config, config_path, args.dry_run))


if __name__ == "__main__":
    main()
