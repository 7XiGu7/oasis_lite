# OASIS Lite Runtime

This repository is a lightweight, API-first OASIS runtime for small social
simulation experiments. It keeps the core platform, agent, recommendation, and
checkpoint code needed for Twitter/Weibo-style simulations, plus a smoke test
that can validate the environment without triggering agent LLM calls.

## Setup

Create a Python 3.10 environment and install the pinned direct dependencies:

```bash
git clone git@github.com:7XiGu7/oasis_lite.git
cd oasis_lite
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

If you already use the checked local environment, run commands with
`.venv/bin/python`.

## Configuration

Copy the example file and fill in local secrets:

```bash
cp .env.example .env
chmod 600 .env
```

For DashScope-compatible API calls, the usual minimum is:

```dotenv
BACKEND=dashscope
MODEL_NAME=qwen-plus
DASHSCOPE_API_KEY=your-dashscope-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
IS_VLLM=false
```

Do not commit `.env`, local tokens, model folders, databases, or runtime
outputs.

## Lightweight Verification

Run validation commands from the repository root:

```bash
cd oasis_lite
```

There are three recommended validation runs:

1. No-API environment smoke: validates DB writes and core platform actions with
   `activation_prob=0.0`.
2. API smoke: repeats the environment smoke and makes one real DashScope call to
   verify credentials and remote API connectivity.
3. Light API experiment: runs a 3-agent, 5-step real simulation with
   `activation_prob=1.0`; this is the run that produces actual agent-generated
   experiment results.

Run the richer environment smoke test without any real LLM/API call:

```bash
.venv/bin/python examples/light_smoke_test.py \
  --steps 1 \
  --skip-real-llm \
  --work-dir outputs/light_smoke_no_api
```

Expected markers:

```text
ENV_SMOKE_TEST_OK
smoke_mode=env_only
REAL_LLM_API_CALL_SKIPPED
api_call_performed=false
db_path=outputs/light_smoke_no_api/tiny_oasis.db
summary_path=outputs/light_smoke_no_api/smoke_summary.json
```

The script uses `activation_prob=0.0` by default, so the agent LLM action path is
not called. It still directly validates core platform actions: posts, comments,
likes, unlikes, dislikes, undo dislikes, reposts, quotes, follow/unfollow,
mute/unmute, reports, groups, search, trend, and comment viewing. It prints
SQLite table counts and trace action counts for manual inspection. This mode
does not instantiate the LLM client and writes `real_llm_smoke_skipped.json`.

Run the environment smoke test plus one real DashScope API call:

```bash
.venv/bin/python examples/light_smoke_test.py \
  --steps 1 \
  --work-dir outputs/light_smoke_api
```

Expected markers:

```text
ENV_SMOKE_TEST_OK
smoke_mode=env_plus_real_llm
REAL_LLM_SMOKE_TEST_OK
REAL_LLM_API_CALL_OK
api_call_performed=true
llm_marker_matched=true
```

The real API check defaults to `backend=dashscope`, `model=qwen-plus`, and sets
`IS_VLLM=false` so it uses the remote API instead of a local vLLM endpoint. It
asks the model for a fixed marker (`oasis-lite-ok`), checks the marker in the
response, writes `real_llm_smoke.json`, and writes token usage to
`real_llm_usage_summary.json`.

## Light API Experiment

Run a tiny real-agent simulation that produces actual experimental outputs:

```bash
.venv/bin/python examples/light_api_experiment.py \
  --work-dir outputs/light_api_experiment \
  --fresh
```

This defaults to 3 Twitter agents, 5 steps, `activation_prob=1.0`, and the
`camel` agent path, so every activated agent goes through real API tool calling
and generates real content in the database. It writes
`experiment_summary.json`, `experiment_usage.json`, and
`light_api_experiment.db`.

Expected markers:

```text
LIGHT_API_EXPERIMENT_OK
agent_llm_calls_performed=true
activation_prob=1.0
```

The script also supports `--agent-variant recagent_user` if you want the
embedding-backed memory path. In that case, set `EMBEDDING_MODEL` in `.env`
to your local model directory.

## Tests

Run the unit tests:

```bash
.venv/bin/python -m pytest -q
```

## Profile Simulations

Twitter-style CSV columns can include:

```text
name,description,personality_traits,following_user_ids,previous_tweets
```

Weibo-style CSV columns can include:

```text
用户名,描述,认证信息,personality,following_simulation_user_ids,previous_tweets,gender
```

Example small Twitter run:

```bash
.venv/bin/python examples/twitter_simulation.py \
  --profile-path profiles/twitter_profiles.csv \
  --steps 2 \
  --activation-prob 0.05 \
  --recsys-device cpu \
  --fresh
```

Example small Weibo run:

```bash
.venv/bin/python examples/weibo_simulation.py \
  --profile-path profiles/weibo_profiles.csv \
  --steps 2 \
  --activation-prob 0.05 \
  --recsys-device cpu \
  --fresh
```

By default, simulation DBs, Chroma data, checkpoints, and logs are written under
`outputs/`. You can override them with `--db-path`, `--checkpoint-dir`,
`--chroma-path`, and the `OASIS_*_LOG_PATH` environment variables.

## Inspecting Outputs

The smoke test prints the SQLite DB path. To inspect a DB quickly:

```bash
.venv/bin/sqlite3 outputs/light_smoke_no_api/tiny_oasis.db ".tables"
.venv/bin/sqlite3 outputs/light_smoke_no_api/tiny_oasis.db \
  "SELECT action, COUNT(*) FROM trace GROUP BY action ORDER BY action;"
```

The `trace` table records platform actions. The `interaction` table records
notifications such as mentions, comments, quotes, reposts, likes, and follows.
