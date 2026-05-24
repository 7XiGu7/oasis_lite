#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Checkpoint persistence helpers for platform simulations.

这个版本使用“完整快照”续跑：每个时间步成功结束后保存 SQLite
和可选 ChromaDB 快照；如果在某个时间步中断，下次启动会丢弃当前工作
目录里的半步数据，并从最近一次 ready 快照继续。
"""

import base64
import json
import os
import pickle
import random
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

SIMULATION_SCHEMA_VERSION = 1
LLM_USAGE_LOG_ENV = "OASIS_LLM_USAGE_LOG_PATH"
DEFAULT_LLM_USAGE_LOG_DIR = Path("/data/lijiantong/Data/oasis/log")
LOG_PATH_ENV_TEMPLATES = {
    "OASIS_ENV_LOG_PATH": "oasis-{variant}-{timestamp}.log",
    "OASIS_LLM_LOG_PATH": "oasis.llm-{variant}-{timestamp}.log",
    "OASIS_PLATFORM_LOG_PATH": "social.twitter-{variant}-{timestamp}.log",
    "OASIS_AGENT_LOG_PATH": "social.agent-{variant}-{timestamp}.log",
    "OASIS_AGENT_ACTION_RECORD_PATH": "agent_action_records_{variant}_{timestamp}.jsonl",
    "OASIS_AGENT_COGNITION_RECORD_PATH": "agent_cognition_records_{variant}_{timestamp}.jsonl",
    LLM_USAGE_LOG_ENV: "llm-usage-summary-{variant}-{timestamp}.json",
}
LOG_PATH_ENV_NAMES = tuple(LOG_PATH_ENV_TEMPLATES)
LOG_TRUNCATE_ENV_NAMES = tuple(
    env_name
    for env_name, filename_template in LOG_PATH_ENV_TEMPLATES.items()
    if filename_template.endswith((".log", ".jsonl"))
)


class CheckpointError(RuntimeError):
    """Raised when checkpoint metadata or snapshots are unusable."""

    pass


def remove_path(path: str | Path) -> None:
    """Remove a file or directory if it exists."""

    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def encode_random_state() -> str:
    """Serialize Python's global random state for deterministic resume."""

    return base64.b64encode(pickle.dumps(random.getstate())).decode("ascii")


def restore_random_state(encoded_state: str | None) -> None:
    """Restore Python's global random state saved by ``encode_random_state``."""

    if not encoded_state:
        return
    random.setstate(pickle.loads(base64.b64decode(encoded_state.encode("ascii"))))


def safe_log_variant(variant: str) -> str:
    """Make the variant name safe for use in generated log filenames."""

    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in variant
    )


def ensure_simulation_log_paths(variant: str) -> dict[str, str]:
    """Ensure all simulation log environment variables point to stable paths."""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_variant = safe_log_variant(variant)
    paths: dict[str, str] = {}
    for env_name, filename_template in LOG_PATH_ENV_TEMPLATES.items():
        path = os.getenv(env_name)
        if not path:
            filename = filename_template.format(
                variant=safe_variant,
                timestamp=timestamp,
            )
            path = str(DEFAULT_LLM_USAGE_LOG_DIR / filename)
            os.environ[env_name] = path
        paths[env_name] = path
    return paths


def ensure_llm_usage_log_path(variant: str) -> str:
    """Return the LLM usage summary path, creating default log paths if needed."""

    return ensure_simulation_log_paths(variant)[LLM_USAGE_LOG_ENV]


def capture_checkpoint_environment() -> dict[str, str]:
    """Persist log-related environment paths into the checkpoint state."""

    return {
        env_name: value
        for env_name in LOG_PATH_ENV_NAMES
        if (value := os.getenv(env_name))
    }


def restore_checkpoint_environment(state: dict[str, Any]) -> dict[str, str]:
    """Restore log-related environment paths from a checkpoint state."""

    environment = state.get("environment")
    if not isinstance(environment, dict):
        return {}

    restored: dict[str, str] = {}
    for env_name in LOG_PATH_ENV_NAMES:
        path = environment.get(env_name)
        if isinstance(path, str) and path:
            os.environ[env_name] = path
            restored[env_name] = path
    return restored


def capture_log_offsets(
    environment: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Capture byte offsets for append-only text logs before a step starts."""

    offsets: dict[str, dict[str, Any]] = {}
    for env_name in LOG_TRUNCATE_ENV_NAMES:
        path = (
            environment.get(env_name)
            if isinstance(environment, dict)
            else os.getenv(env_name)
        )
        if not isinstance(path, str) or not path:
            continue

        log_path = Path(path)
        exists = log_path.exists()
        offsets[env_name] = {
            "path": path,
            "offset": log_path.stat().st_size if exists else 0,
            "exists": exists,
        }
    return offsets


def truncate_logs_to_offsets(offsets: Any) -> None:
    """Truncate .log/.jsonl files back to captured step-start offsets."""

    if not isinstance(offsets, dict):
        return

    for env_name, item in offsets.items():
        if env_name not in LOG_TRUNCATE_ENV_NAMES or not isinstance(item, dict):
            continue
        path = item.get("path")
        if not isinstance(path, str) or not path.endswith((".log", ".jsonl")):
            continue

        log_path = Path(path)
        if not bool(item.get("exists")):
            remove_path(log_path)
            continue
        if not log_path.exists():
            continue

        offset = int(item.get("offset", 0))
        current_size = log_path.stat().st_size
        if current_size > offset:
            with open(log_path, "rb+") as f:
                f.truncate(offset)


class CheckpointManager:
    """Manage full SQLite/Chroma snapshots for step-level resume.

    Invariant: ``state.json`` points at the latest complete ``ready`` snapshot.
    During ``save_ready`` the older snapshot is temporarily kept as
    ``*.previous`` so a crash while writing a new snapshot can still fall back
    to the last complete time step.
    """

    def __init__(self, variant: str, checkpoint_dir: str | Path):
        """Bind this manager to one simulation variant and checkpoint folder."""

        self.variant = variant
        self.checkpoint_dir = Path(checkpoint_dir)
        self.state_path = self.checkpoint_dir / "state.json"
        self.db_snapshot_path = self.checkpoint_dir / "db_snapshot.db"
        self.chroma_snapshot_path = self.checkpoint_dir / "chroma_snapshot"
        self.db_previous_path = self.checkpoint_dir / "db_snapshot.previous.db"
        self.chroma_previous_path = self.checkpoint_dir / "chroma_snapshot.previous"

    def has_checkpoint(self) -> bool:
        """Return whether a checkpoint state file exists."""

        return self.state_path.exists()

    def load_state(self) -> dict[str, Any]:
        """Load and validate checkpoint metadata.

        If the previous run stopped while ``status == running``, the latest
        stable data is the snapshot before that time step. When a crash happened
        during snapshot replacement, ``*.previous`` is restored first.
        """

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as exc:
            raise CheckpointError(
                f"无法读取 checkpoint state.json，请使用 --fresh 重新开始：{exc}"
            ) from exc

        if state.get("schema_version") != SIMULATION_SCHEMA_VERSION:
            raise CheckpointError("checkpoint 版本不匹配，请使用 --fresh 重新开始。")
        if state.get("variant") != self.variant:
            raise CheckpointError("checkpoint variant 不匹配，请使用 --fresh 重新开始。")
        if state.get("status") not in {"ready", "running"}:
            raise CheckpointError("checkpoint 状态异常，请使用 --fresh 重新开始。")
        has_chroma_snapshot = state.get("has_chroma_snapshot", True)
        if state["status"] == "running":
            self._restore_previous_snapshots_if_present()
        if not self.db_snapshot_path.exists():
            raise CheckpointError("checkpoint DB 快照不完整，请使用 --fresh 重新开始。")
        if has_chroma_snapshot and not self.chroma_snapshot_path.exists():
            raise CheckpointError("checkpoint 快照不完整，请使用 --fresh 重新开始。")
        return state

    def write_state(self, state: dict[str, Any]) -> None:
        """Atomically write ``state.json`` after refreshing ``updated_at``."""

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        tmp_path = self.state_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.state_path)

    def restore_to(
        self,
        db_path: str | Path,
        chromadb_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Replace live SQLite/Chroma data with the saved ready snapshot.

        This is the operation that clears all partial DB and vector-store writes
        from the interrupted time step before the main simulation loop reruns
        that same step from its beginning.
        """

        state = self.load_state()
        has_chroma_snapshot = state.get("has_chroma_snapshot", True)
        remove_path(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.db_snapshot_path, db_path)
        if has_chroma_snapshot:
            if chromadb_path is None:
                raise CheckpointError(
                    "checkpoint 包含 Chroma 快照，但当前未提供 Chroma 路径。"
                )
            remove_path(chromadb_path)
            Path(chromadb_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.chroma_snapshot_path, chromadb_path)
        if state["status"] == "running":
            truncate_logs_to_offsets(state.get("log_offsets_before_step"))
            state["status"] = "ready"
            state.pop("running_step", None)
            state.pop("log_offsets_before_step", None)
            self.write_state(state)
        return state

    def mark_running(self, state: dict[str, Any], running_step: int) -> dict[str, Any]:
        """Record that ``running_step`` has started but has not completed yet.

        The log offsets let resume remove .log/.jsonl lines written by a
        half-finished step before that step is rerun.
        """

        running_state = dict(state)
        running_state["status"] = "running"
        running_state["running_step"] = running_step
        running_state["log_offsets_before_step"] = capture_log_offsets(
            running_state.get("environment")
        )
        self.write_state(running_state)
        return running_state

    def save_ready(
        self,
        env,
        db_path: str | Path,
        total_steps: int,
        last_completed_step: int,
        next_step: int,
        chromadb_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Save a complete checkpoint after one time step finishes.

        ``next_step`` is the next step to execute on resume. Therefore, if the
        process interrupts later while this step is running, the restored state
        still starts at the beginning of that interrupted step.
        """

        env.platform.db.commit()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._backup_sqlite(db_path)
        has_chroma_snapshot = chromadb_path is not None
        if has_chroma_snapshot:
            self._copy_chroma(chromadb_path)
        else:
            self._remove_chroma_snapshots()
        state = {
            "schema_version": SIMULATION_SCHEMA_VERSION,
            "variant": self.variant,
            "status": "ready",
            "has_chroma_snapshot": has_chroma_snapshot,
            "total_steps": total_steps,
            "last_completed_step": last_completed_step,
            "next_step": next_step,
            "sandbox_clock_time_step": int(env.platform.sandbox_clock.time_step),
            "random_state": encode_random_state(),
            "environment": capture_checkpoint_environment(),
        }
        self.write_state(state)
        self._remove_previous_snapshots()
        return state

    def _backup_sqlite(self, db_path: str | Path) -> None:
        """Copy the live SQLite DB into ``db_snapshot.db`` safely."""

        tmp_path = self.checkpoint_dir / "db_snapshot.db.tmp"
        remove_path(tmp_path)
        source = sqlite3.connect(db_path)
        target = sqlite3.connect(tmp_path)
        try:
            with target:
                source.backup(target)
        finally:
            target.close()
            source.close()
        remove_path(self.db_previous_path)
        if self.db_snapshot_path.exists():
            os.replace(self.db_snapshot_path, self.db_previous_path)
        os.replace(tmp_path, self.db_snapshot_path)

    def _copy_chroma(self, chromadb_path: str | Path) -> None:
        """Copy the live Chroma directory into ``chroma_snapshot`` safely."""

        tmp_path = self.checkpoint_dir / "chroma_snapshot.tmp"
        remove_path(tmp_path)
        if Path(chromadb_path).exists():
            shutil.copytree(chromadb_path, tmp_path)
        else:
            tmp_path.mkdir(parents=True, exist_ok=True)
        remove_path(self.chroma_previous_path)
        if self.chroma_snapshot_path.exists():
            os.replace(self.chroma_snapshot_path, self.chroma_previous_path)
        os.replace(tmp_path, self.chroma_snapshot_path)

    def _restore_previous_snapshots_if_present(self) -> None:
        """Rollback half-written snapshot replacements after an interruption."""

        if self.db_previous_path.exists():
            remove_path(self.db_snapshot_path)
            os.replace(self.db_previous_path, self.db_snapshot_path)
        if self.chroma_previous_path.exists():
            remove_path(self.chroma_snapshot_path)
            os.replace(self.chroma_previous_path, self.chroma_snapshot_path)

    def _remove_previous_snapshots(self) -> None:
        """Drop fallback snapshots once the new ready state is durable."""

        remove_path(self.db_previous_path)
        remove_path(self.chroma_previous_path)

    def _remove_chroma_snapshots(self) -> None:
        """Drop Chroma snapshots for SQLite-only checkpoint variants."""

        remove_path(self.chroma_snapshot_path)
        remove_path(self.chroma_previous_path)


def ensure_simulation_tables(conn: sqlite3.Connection) -> None:
    """Create metadata tables used by checkpoint progress."""

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS simulation_progress (
            variant TEXT PRIMARY KEY,
            total_steps INTEGER NOT NULL,
            last_completed_step INTEGER NOT NULL,
            next_step INTEGER NOT NULL,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS simulation_agent_state (
            variant TEXT NOT NULL,
            agent_id INTEGER NOT NULL,
            description TEXT,
            is_init INTEGER NOT NULL,
            login_times_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (variant, agent_id)
        );
        """
    )
    conn.commit()


def save_progress(
    conn: sqlite3.Connection,
    variant: str,
    total_steps: int,
    last_completed_step: int,
    next_step: int,
    status: str,
) -> None:
    """Upsert human-readable progress metadata into the live SQLite DB."""

    conn.execute(
        """
        INSERT INTO simulation_progress (
            variant, total_steps, last_completed_step, next_step, status, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(variant) DO UPDATE SET
            total_steps = excluded.total_steps,
            last_completed_step = excluded.last_completed_step,
            next_step = excluded.next_step,
            status = excluded.status,
            updated_at = excluded.updated_at
        """,
        (
            variant,
            total_steps,
            last_completed_step,
            next_step,
            status,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


def save_agent_states(conn: sqlite3.Connection, variant: str, env) -> None:
    """Persist per-agent mutable fields needed after rebuilding the graph."""

    now = datetime.now().isoformat(timespec="seconds")
    rows = []
    for agent_id, agent in env.agent_graph.get_agents():  # type: ignore[attr-defined]
        rows.append(
            (
                variant,
                agent_id,
                getattr(agent, "description", None),
                int(bool(getattr(agent, "is_init", False))),
                json.dumps(getattr(agent, "login_times", []), ensure_ascii=False),
                now,
            )
        )
    conn.executemany(
        """
        INSERT INTO simulation_agent_state (
            variant, agent_id, description, is_init,
            login_times_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(variant, agent_id) DO UPDATE SET
            description = excluded.description,
            is_init = excluded.is_init,
            login_times_json = excluded.login_times_json,
            updated_at = excluded.updated_at
        """,
        rows,
    )
    conn.commit()


def restore_agent_states(conn: sqlite3.Connection, variant: str, env) -> None:
    """Restore mutable agent fields into the freshly constructed agent graph."""

    rows = conn.execute(
        """
        SELECT agent_id, description, is_init, login_times_json
        FROM simulation_agent_state
        WHERE variant = ?
        """,
        (variant,),
    ).fetchall()
    if not rows:
        raise CheckpointError("checkpoint 缺少 agent 状态，请使用 --fresh 重新开始。")

    agents_by_id = dict(env.agent_graph.get_agents())  # type: ignore[attr-defined]
    for agent_id, description, is_init, login_times_json in rows:
        agent = agents_by_id.get(agent_id)
        if agent is None:
            continue
        agent.description = description
        agent.is_init = bool(is_init)
        agent.login_times = json.loads(login_times_json)
