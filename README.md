# routerbench

A benchmark framework for LLM router services — the kind of service that
takes a prompt and decides which underlying model (small/fast vs.
large/capable, or across providers) should actually handle it.

It measures the four things that matter for a router specifically, not just
for a single model:

1. **Routing accuracy** — did it pick the right model for each prompt?
2. **Latency & throughput** — response time and req/s under increasing concurrency.
3. **Cost efficiency** — $/request, and savings vs. always using the most expensive model.
4. **Reliability** — error/timeout rates, retry recovery, and behavior under fault injection and adversarial inputs.

## Architecture

```
routerbench/
  config.py        # YAML-driven config (dataclasses, no pydantic dependency)
  models.py         # Core data types: RouterResponse, DatasetItem, RequestResult, ScenarioResult
  client.py         # RouterClient: thin async HTTP adapter around the router under test
  dataset.py        # Loads benchmark cases from JSONL
  scorer.py         # Pluggable quality scoring (Scorer protocol; heuristic default)
  scenarios/        # One module per benchmark scenario
    accuracy.py      #   routing accuracy + quality + cost, one pass over the dataset
    load.py           #   latency/throughput swept across concurrency levels
    reliability.py     #   baseline + fault-injected + adversarial-edge-case traffic
  metrics/          # Pure functions turning RequestResult lists into stats
    latency.py, cost.py, routing.py, reliability.py
  runner.py         # Orchestrates: config -> client -> scenarios -> results
  report.py         # Aggregates results into JSON / Markdown / HTML reports
  cli.py            # `routerbench run --config ...`

mock_router/        # A fake router (FastAPI) so the framework runs end-to-end
                     # with no external dependency. Point configs/*.yaml at your
                     # real router's base_url instead when you're ready.
configs/default.yaml
data/sample_prompts.jsonl
tests/
```

### Design principles

- **Adapter, not assumption.** `RouterClient` doesn't hardcode a response
  shape. Every field it reads (model used, content, token counts) is a
  configurable dot-path in `RouterConfig`, so it works against an
  OpenAI-compatible endpoint, a bespoke router API, or anything in between —
  point it at your service and adjust the paths in the config, no code
  changes needed.
- **Scenarios are independent and composable.** Each one produces its own
  `ScenarioResult` (raw per-request results + computed metrics). Enable/disable
  and tune each independently in the config; add a new scenario by writing one
  function with the same shape.
- **Metrics are pure functions over `RequestResult` lists.** They don't know
  about HTTP, concurrency, or scenarios — easy to unit test and easy to reuse
  (e.g. cost math is shared between the accuracy scenario and any future one).
  Throughput is computed from real measured wall-clock time per batch, not
  reconstructed from individual latencies (which would be wrong once requests
  overlap).
- **Scoring is pluggable.** The default `HeuristicScorer` only checks for
  required substrings — good enough to catch outright routing/quality
  failures on objective QA without needing API keys. Swap in a real
  LLM-as-judge by implementing the `Scorer` protocol.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[mock,dev]"

# Terminal 1: start the mock router (stand-in for your real service)
uvicorn mock_router.server:app --port 8000

# Terminal 2: run the benchmark against it
routerbench run --config configs/default.yaml
```

This prints a summary table per scenario to the terminal and writes
JSON/HTML reports to `reports/`.

## Pointing it at your real router

Edit `configs/default.yaml`:

```yaml
router:
  base_url: "https://your-router.internal"
  chat_path: "/v1/chat/completions"
  api_key_env: "ROUTER_API_KEY"     # read from env var, never hardcoded
  response_model_path: "model"                       # adjust to your response shape
  response_content_path: "choices.0.message.content"
  response_input_tokens_path: "usage.prompt_tokens"
  response_output_tokens_path: "usage.completion_tokens"

pricing:
  <your-model-names>:
    input_per_1k: ...
    output_per_1k: ...
```

Then replace `data/sample_prompts.jsonl` with your own dataset (same JSONL
schema — see below) so `expected_model` and `expected_answer_contains`
reflect your router's actual intended behavior.

## Dataset format

One JSON object per line:

```json
{"id": "qa-1", "category": "simple_qa", "prompt": "What is the capital of France?", "expected_model": "small-model", "expected_answer_contains": ["Paris"]}
```

- `expected_model` is optional — omit it for items you don't have a routing
  ground truth for (they still count toward latency/cost/quality, just not accuracy).
- `expected_answer_contains` is optional — omit it for open-ended prompts
  that can't be scored by substring match (they still count toward routing
  accuracy/latency/cost, just not quality).

## Running tests

```bash
pytest
```

Tests cover metrics math, the scorer, dataset loading, and the HTTP client's
retry/timeout/error handling (mocked with `respx`, no network needed).

## Extending

- **New scenario**: add `scenarios/my_scenario.py` following the shape of
  `scenarios/load.py`, wire it into `runner.py` and `config.py`.
- **New metric**: add a pure function in `metrics/` taking a
  `list[RequestResult]` and returning a dict.
- **Real LLM-judge scoring**: implement `Scorer.score()` in `scorer.py` using
  `RouterClient` (or any LLM API) to grade `content` against the prompt, then
  select it via `scorer.mode` in config.
