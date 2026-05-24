# Contributing

## Workflow

Use short-lived feature branches from `main`:

```bash
git checkout main
git pull --ff-only
git checkout -b feat/short-description
```

Keep commits focused and use imperative messages, for example:

```text
Fix platform trace recording
Add lightweight smoke coverage
```

## Local Checks

Before opening a pull request, run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python examples/light_smoke_test.py --steps 1 --skip-real-llm --work-dir outputs/light_smoke_no_api
```

When changing API or LLM integration code, also run:

```bash
.venv/bin/python examples/light_smoke_test.py --steps 1 --work-dir outputs/light_smoke_api
```

## Secrets And Outputs

Never commit:

- `.env`
- `.venv/`
- `outputs/`
- SQLite databases
- log files
- local tokens, API keys, model folders, or checkpoints

Use `.env.example` for new configuration variables and keep real values only in
local ignored files.

## Code Style

Prefer small, conservative changes that match the current code structure. Add
tests or smoke coverage when touching platform action behavior, persistence,
checkpointing, recommendation, or LLM integration paths.
