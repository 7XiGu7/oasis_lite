import asyncio
import importlib.util
import os
import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OASIS_ROOT = PROJECT_ROOT / "oasis"


def load_agent_environment(monkeypatch):
    oasis_pkg = types.ModuleType("oasis")
    oasis_pkg.__path__ = [str(OASIS_ROOT)]
    monkeypatch.setitem(sys.modules, "oasis", oasis_pkg)

    social_agent_pkg = types.ModuleType("oasis.social_agent")
    social_agent_pkg.__path__ = [str(OASIS_ROOT / "social_agent")]
    monkeypatch.setitem(sys.modules, "oasis.social_agent", social_agent_pkg)

    social_platform_pkg = types.ModuleType("oasis.social_platform")
    social_platform_pkg.__path__ = [str(OASIS_ROOT / "social_platform")]
    monkeypatch.setitem(sys.modules, "oasis.social_platform", social_platform_pkg)

    agent_action = types.ModuleType("oasis.social_agent.agent_action")
    agent_action.SocialAction = type("SocialAction", (), {})
    monkeypatch.setitem(sys.modules, "oasis.social_agent.agent_action", agent_action)

    database = types.ModuleType("oasis.social_platform.database")
    database.get_db_path = lambda: os.environ["OASIS_DB_PATH"]
    monkeypatch.setitem(sys.modules, "oasis.social_platform.database", database)

    config = types.ModuleType("oasis.social_platform.config")
    config.UserInfo = type("UserInfo", (), {})
    monkeypatch.setitem(sys.modules, "oasis.social_platform.config", config)

    typing_module = types.ModuleType("oasis.social_platform.typing")

    class ActionType:
        FOLLOW = SimpleNamespace(value="follow")
        CREATE_COMMENT = SimpleNamespace(value="create_comment")
        QUOTE_POST = SimpleNamespace(value="quote_post")

    typing_module.ActionType = ActionType
    monkeypatch.setitem(sys.modules, "oasis.social_platform.typing", typing_module)

    localization = types.ModuleType("oasis.localization")
    localization.is_zh_locale = lambda locale: False
    localization.zh_action_label = lambda action: action
    monkeypatch.setitem(sys.modules, "oasis.localization", localization)

    spec = importlib.util.spec_from_file_location(
        "oasis.social_agent.agent_environment",
        OASIS_ROOT / "social_agent" / "agent_environment.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def create_notification_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(
        """
        CREATE TABLE user (
            user_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );

        CREATE TABLE interaction (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            post_id INTEGER,
            comment_id INTEGER,
            original_user_id INTEGER,
            is_mention INTEGER,
            interaction_type TEXT NOT NULL,
            created_at DATETIME NOT NULL
        );
        """
    )
    cursor.execute("INSERT INTO user (user_id, name) VALUES (?, ?)", (1, "Target"))
    cursor.execute("INSERT INTO user (user_id, name) VALUES (?, ?)", (2, "Other"))
    for user_id in range(101, 114):
        cursor.execute(
            "INSERT INTO user (user_id, name) VALUES (?, ?)",
            (user_id, f"User {user_id}"),
        )

    for offset, user_id in enumerate(range(101, 113), start=1):
        cursor.execute(
            """
            INSERT INTO interaction (
                user_id, post_id, comment_id, original_user_id, is_mention,
                interaction_type, created_at
            ) VALUES (?, NULL, NULL, 1, 0, 'follow', ?)
            """,
            (user_id, f"2026-05-11 12:{offset:02d}:00"),
        )

    cursor.execute(
        """
        INSERT INTO interaction (
            user_id, post_id, comment_id, original_user_id, is_mention,
            interaction_type, created_at
        ) VALUES (113, NULL, NULL, 1, 0, 'follow', '2026-05-11 12:12:00')
        """
    )
    cursor.execute(
        """
        INSERT INTO interaction (
            user_id, post_id, comment_id, original_user_id, is_mention,
            interaction_type, created_at
        ) VALUES (113, NULL, NULL, 2, 0, 'follow', '2026-05-11 13:00:00')
        """
    )
    conn.commit()
    conn.close()


def remaining_interaction_ids(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT interaction_id FROM interaction ORDER BY interaction_id"
    )
    ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return ids


def test_get_notifications_queries_latest_ten_for_current_agent(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "notifications.db"
    create_notification_db(db_path)
    monkeypatch.setenv("OASIS_DB_PATH", str(db_path))
    module = load_agent_environment(monkeypatch)

    env = module.SocialEnvironment(
        SimpleNamespace(agent_id=1),
        user_info=SimpleNamespace(name="Target", locale="en"),
        locale="en",
    )

    notifications = asyncio.run(env.get_notifications())

    assert len(notifications) == module.MAX_NOTIFICATIONS_PER_AGENT
    assert [item["user_id"] for item in notifications] == [
        113,
        112,
        111,
        110,
        109,
        108,
        107,
        106,
        105,
        104,
    ]
    assert all("Target" in item["content"] for item in notifications)

    # Only the queried latest 10 rows for agent 1 are deleted. Older rows and
    # another agent's notification remain for later processing.
    assert remaining_interaction_ids(db_path) == [1, 2, 3, 14]
