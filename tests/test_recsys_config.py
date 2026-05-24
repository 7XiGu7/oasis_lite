import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OASIS_ROOT = PROJECT_ROOT / "oasis"

oasis_pkg = types.ModuleType("oasis")
oasis_pkg.__path__ = [str(OASIS_ROOT)]
sys.modules.setdefault("oasis", oasis_pkg)

social_platform_pkg = types.ModuleType("oasis.social_platform")
social_platform_pkg.__path__ = [str(OASIS_ROOT / "social_platform")]
sys.modules.setdefault("oasis.social_platform", social_platform_pkg)

sentence_transformers_stub = types.ModuleType("sentence_transformers")
sentence_transformers_stub.SentenceTransformer = lambda *args, **kwargs: None
sys.modules.setdefault("sentence_transformers", sentence_transformers_stub)

process_recsys_posts_stub = types.ModuleType(
    "oasis.social_platform.process_recsys_posts"
)
process_recsys_posts_stub.generate_post_vector = lambda *args, **kwargs: None
process_recsys_posts_stub.generate_post_vector_openai = (
    lambda *args, **kwargs: None
)
sys.modules.setdefault(
    "oasis.social_platform.process_recsys_posts",
    process_recsys_posts_stub,
)


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, item):
        if isinstance(item, FakeTensor):
            item = item.values
        return FakeTensor(self.values[item])

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def fake_topk(values, k, dim=1, largest=True, sorted=True):
    array = values.values if isinstance(values, FakeTensor) else np.asarray(values)
    indices = np.argsort(array, axis=dim)
    if largest:
        indices = np.flip(indices, axis=dim)
    indices = np.take(indices, np.arange(k), axis=dim)
    top_values = np.take_along_axis(array, indices, axis=dim)
    return FakeTensor(top_values), FakeTensor(indices)


torch_stub = types.ModuleType("torch")
torch_stub.cuda = SimpleNamespace(is_available=lambda: False)
torch_stub.device = lambda value: value
torch_stub.tensor = lambda values: FakeTensor(values)
torch_stub.topk = fake_topk
sys.modules.setdefault("torch", torch_stub)


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


typing_module = load_module(
    "oasis.social_platform.typing",
    OASIS_ROOT / "social_platform" / "typing.py",
)
RecsysConfig = typing_module.RecsysConfig


def load_recsys_module():
    return load_module(
        "oasis.social_platform.recsys",
        OASIS_ROOT / "social_platform" / "recsys.py",
    )


def test_recsys_config_resolves_adaptive_embedding_batch_size():
    assert RecsysConfig.twitter("cuda:0").resolved_embedding_batch_size() == 256
    assert RecsysConfig.twitter("cuda:6").resolved_embedding_batch_size() == 256
    assert RecsysConfig.twitter("cpu").resolved_embedding_batch_size() == 64
    assert RecsysConfig.twitter("cpu").resolved_openai_embedding_batch_size() == 100


def test_recsys_config_accepts_explicit_model_path():
    config = RecsysConfig.twitter(
        available_device="cuda:6",
        model_name_or_path="/data/local/twhin",
        embedding_batch_size=32,
    )

    assert config.model_name_or_path == "/data/local/twhin"
    assert config.resolved_embedding_batch_size() == 32


def test_twhin_model_loading_uses_configured_model_path(monkeypatch):
    recsys = load_recsys_module()

    calls = {"tokenizer": [], "model": []}

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(pretrained_model_name_or_path, **kwargs):
            calls["tokenizer"].append((pretrained_model_name_or_path, kwargs))
            return SimpleNamespace(kind="tokenizer")

    class FakeModel:
        @staticmethod
        def from_pretrained(pretrained_model_name_or_path, **kwargs):
            calls["model"].append((pretrained_model_name_or_path, kwargs))
            return FakeLoadedModel()

    class FakeLoadedModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.is_eval = True
            return self

    monkeypatch.setattr(recsys, "_twhin_tokenizers", {})
    monkeypatch.setattr(recsys, "_twhin_models", {})
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeTokenizer, AutoModel=FakeModel),
    )

    tokenizer, model = recsys.get_recsys_model(
        available_device="cuda:6",
        recsys_type="twhin-bert",
        model_name_or_path="/data/local/twhin",
    )

    assert tokenizer.kind == "tokenizer"
    assert model.device == "cuda:6"
    assert model.is_eval is True
    assert calls["tokenizer"][0][0] == "/data/local/twhin"
    assert calls["model"][0][0] == "/data/local/twhin"


def test_rec_sys_personalized_twh_passes_batch_and_coarse_filter(monkeypatch):
    recsys = load_recsys_module()

    captured = {}

    monkeypatch.setattr(recsys, "t_items", {1: "p1", 2: "p2", 3: "p3"})
    monkeypatch.setattr(recsys, "date_score", [1.0, 1.0, 1.0])
    monkeypatch.setattr(recsys, "user_profiles", ["bio"])
    monkeypatch.setattr(recsys, "user_previous_post", {0: ""})
    monkeypatch.setattr(recsys, "user_previous_post_all", {0: []})
    monkeypatch.setattr(recsys, "u_items", {0: 0})
    monkeypatch.setattr(
        recsys,
        "get_recsys_model",
        lambda *args, **kwargs: (SimpleNamespace(), SimpleNamespace()),
    )

    def fake_coarse_filtering(input_list, scale):
        captured["coarse_filter_size"] = scale
        return input_list[:2], range(2)

    def fake_generate_post_vector(model, tokenizer, texts, device, batch_size):
        captured["embedding_batch_size"] = batch_size
        captured["texts"] = texts
        return np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

    monkeypatch.setattr(recsys, "coarse_filtering", fake_coarse_filtering)
    monkeypatch.setattr(recsys, "generate_post_vector", fake_generate_post_vector)

    result = recsys.rec_sys_personalized_twh(
        user_table=[{"user_id": 0, "agent_id": 0, "bio": "bio", "num_followers": 0}],
        post_table=[
            {"post_id": 1, "user_id": 1, "content": "p1", "created_at": 0},
            {"post_id": 2, "user_id": 2, "content": "p2", "created_at": 0},
            {"post_id": 3, "user_id": 3, "content": "p3", "created_at": 0},
        ],
        latest_post_count=0,
        trace_table=[],
        rec_matrix=[[]],
        max_rec_post_len=1,
        current_time=0,
        available_device="cpu",
        model_name_or_path="/data/local/twhin",
        embedding_batch_size=7,
        coarse_filter_size=11,
    )

    assert captured["embedding_batch_size"] == 7
    assert captured["coarse_filter_size"] == 11
    assert captured["texts"] == ["bio", "p1", "p2"]
    assert result == [[1]]
