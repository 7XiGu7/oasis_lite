# OASIS Lite Runtime

This repository is a lightweight, API-first OASIS runtime for small social
simulation experiments. It keeps the core platform, agent, recommendation, and
checkpoint code needed for Twitter/Weibo-style simulations, plus a smoke test
that can validate the environment without triggering agent LLM calls.

## Setup

Create a Python 3.10 environment and install the pinned direct dependencies:

```bash
cd /data/lijiantong/oasis_lite/oasis-lite-runtime-20260522-105953
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
db_path=outputs/light_smoke_no_api/tiny_oasis.db
```

The script uses `activation_prob=0.0` by default, so the agent LLM action path is
not called. It still directly validates core platform actions: posts, comments,
likes, unlikes, dislikes, undo dislikes, reposts, quotes, follow/unfollow,
mute/unmute, reports, groups, search, trend, and comment viewing. It prints
SQLite table counts and trace action counts for manual inspection.

Run the same smoke test plus one real DashScope API call:

```bash
.venv/bin/python examples/light_smoke_test.py \
  --steps 1 \
  --work-dir outputs/light_smoke_api
```

Expected markers:

```text
ENV_SMOKE_TEST_OK
REAL_LLM_SMOKE_TEST_OK
```

The real API check defaults to `backend=dashscope`, `model=qwen-plus`, and sets
`IS_VLLM=false` so it uses the remote API instead of a local vLLM endpoint.

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
  --profile-path /path/to/twitter_profiles.csv \
  --steps 2 \
  --activation-prob 0.05 \
  --recsys-device cpu \
  --fresh
```

Example small Weibo run:

```bash
.venv/bin/python examples/weibo_simulation.py \
  --profile-path /path/to/weibo_profiles.csv \
  --steps 2 \
  --activation-prob 0.05 \
  --recsys-device cpu \
  --fresh
```

By default, simulation DBs and checkpoints are written under
`/data/lijiantong/Data/oasis/data` and
`/data/lijiantong/Saved/oasis/checkpoints`. You can override them with
`--db-path`, `--checkpoint-dir`, and `--chroma-path`.

## Inspecting Outputs

The smoke test prints the SQLite DB path. To inspect a DB quickly:

```bash
.venv/bin/sqlite3 outputs/light_smoke_no_api/tiny_oasis.db ".tables"
.venv/bin/sqlite3 outputs/light_smoke_no_api/tiny_oasis.db \
  "SELECT action, COUNT(*) FROM trace GROUP BY action ORDER BY action;"
```

The `trace` table records platform actions. The `interaction` table records
notifications such as mentions, comments, quotes, reposts, likes, and follows.
