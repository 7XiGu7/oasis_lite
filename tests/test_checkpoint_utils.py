import importlib.util
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_UTILS_PATH = PROJECT_ROOT / "examples" / "checkpoint_utils.py"


def load_checkpoint_utils():
    spec = importlib.util.spec_from_file_location(
        "checkpoint_utils_for_test",
        CHECKPOINT_UTILS_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ensure_simulation_log_paths_sets_managed_environment(monkeypatch, tmp_path):
    utils = load_checkpoint_utils()
    for env_name in utils.LOG_PATH_ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(utils, "DEFAULT_LLM_USAGE_LOG_DIR", tmp_path)

    paths = utils.ensure_simulation_log_paths("variant/test")

    assert set(paths) == set(utils.LOG_PATH_ENV_NAMES)
    for env_name, path in paths.items():
        assert os.environ[env_name] == path
        assert path.startswith(str(tmp_path))
        assert "variant_test" in path
    usage_log_path = paths[utils.LLM_USAGE_LOG_ENV]
    assert os.environ[utils.LLM_USAGE_LOG_ENV] == usage_log_path


def test_restore_checkpoint_environment_restores_managed_log_paths(monkeypatch):
    utils = load_checkpoint_utils()
    for env_name in utils.LOG_PATH_ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)

    environment = {
        env_name: f"/tmp/{env_name}.log"
        for env_name in utils.LOG_PATH_ENV_NAMES
    }

    restored = utils.restore_checkpoint_environment(
        {
            "environment": environment,
        }
    )

    assert restored == environment
    for env_name, path in environment.items():
        assert os.environ[env_name] == path
