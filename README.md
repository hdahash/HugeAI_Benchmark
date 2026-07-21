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
  scoring/          # Pluggable quality scoring (see "Measuring quality" below)
    base.py          #   async Scorer protocol + NullScorer
    heuristic.py      #   substring-match scoring for objective QA
    code_exec.py       #   sandboxed execution of generated code against test cases
    llm_judge.py        #   LLM-as-judge scoring for open-ended items
    composite.py         #   dispatches each item to whichever strategy actually applies
  scenarios/        # One module per benchmark scenario
    accuracy.py      #   routing accuracy + quality + cost + routing regret, one pass over the dataset
    load.py           #   latency/throughput swept across concurrency levels
    reliability.py     #   baseline + fault-injected + adversarial-edge-case traffic
  metrics/          # Pure functions turning RequestResult lists into stats
    latency.py, cost.py, routing.py, reliability.py, regret.py
  runner.py         # Orchestrates: config -> client(s) -> scenarios -> results
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
- **Scoring is pluggable and dispatches per item.** See "Measuring quality"
  below — verifiable items (code, objective QA) are checked deterministically;
  everything else falls back to an LLM judge if one is configured.

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

`configs/hugeai.yaml` / `data/hugeai_prompts.jsonl` are a ready-made example
of this pointed at a real service. `scripts/run_hugeai.sh` sets up the venv,
installs dependencies, and runs it in one step:

```bash
export HUGEAI_API_KEY="sk-..."     # your real key -- never hardcode it, never commit it
./scripts/run_hugeai.sh            # defaults to configs/hugeai.yaml
./scripts/run_hugeai.sh configs/my-other-config.yaml   # or pass a different config
```

Run it from a machine that can actually reach your router (your own
machine, a CI runner) — it does nothing special about network access.

### Two things worth checking for any real router

- **Rate limiting** (`rate_limit_rps` in the config): a full run fires 100+
  requests across three scenarios in quick succession. If your router has a
  requests-per-minute cap, set this comfortably under it (see
  `configs/hugeai.yaml` for a worked example) or the load/reliability
  scenarios will report your own self-inflicted 429s as if they were real
  capacity limits.
- **Server-side caching** (`router.cache_bust`): if identical prompts get
  cached and returned near-instantly (common, and sometimes not even
  disableable by the client — see `configs/hugeai.yaml`'s comment on
  `FORCE_CACHE_ENABLED`), the load scenario's "throughput" numbers end up
  measuring cache-hit latency instead of real backend capacity, since the
  same 24-item dataset gets reused across the accuracy pass and every
  concurrency level. Setting `cache_bust: true` appends a fresh nonce to
  every prompt so the router never sees a repeat — same task difficulty,
  no possibility of a cache hit.

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
- `metadata.test_cases` (+ `metadata.entry_point`) is optional — for coding
  items, gives ground-truth execution-based scoring instead of a proxy. See
  "Measuring quality" below.
- `expected_answer_excludes` is optional — substrings that must NOT appear,
  symmetric to `expected_answer_contains`. Useful for testing whether an
  injected instruction can override a server-side policy (e.g. checking a
  known "no emojis" policy actually holds under a prompt trying to break it —
  see `data/hugeai_prompts.jsonl`'s `security` category), or whether a PII
  value leaks into the response unredacted.
- `conversation` is optional — prior turns before `prompt`, e.g.
  `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`.
  `prompt` is always the final user turn actually sent. Omit for the common
  single-turn case. See `data/hugeai_prompts.jsonl`'s `multi_turn` category.

## Measuring quality

Not every prompt can be graded the same way, so `scorer.mode: "auto"` (the
default) dispatches each item to whichever strategy actually applies to it,
verifiable checks before subjective ones:

1. **`metadata.test_cases` present → code execution.** The model's generated
   code is run in a sandboxed subprocess against the test cases and scored
   pass-rate — ground truth, not a proxy. See the `code_exec:` config block
   and the SECURITY note in `routerbench/scoring/code_exec.py` before pointing
   this at anything but disposable/ephemeral infrastructure.
2. **`expected_answer_contains` and/or `expected_answer_excludes` present →
   substring match.** For objective QA where a keyword is enough to catch
   outright failures, or for checking forbidden content stays out.
3. **Otherwise, if `judge.enabled` → LLM-as-judge.** For open-ended items
   (reasoning, creative writing, long-context summarization) with no
   verifiable ground truth. Configured via `judge:` in the YAML config —
   `model`, `base_url`, etc. default to inheriting from `router:`, so by
   default the router judges its own output with a different model forced;
   point it at a separate, stronger judge model for real use. Rubrics are
   keyed by category (`judge.rubric.<category>`), with `judge.default_rubric`
   as the fallback.
4. **Otherwise → unscored (`None`).**

Quality scoring only runs in the accuracy scenario — the load and reliability
scenarios reuse the same prompts purely to measure timing/error behavior, and
re-grading them every time would multiply judge cost/latency for no benefit.

### Routing regret: quality relative to cost

Routing accuracy and cost alone can't tell "cheap and just as good" apart
from "cheap and worse" — a router that routes everything to the biggest
model gets a great quality score and is still a bad router. Setting
`accuracy.compute_routing_regret: true` (with `baseline_model` set) fires
each prompt a second time forced to the baseline model, and compares:

- **`avg_quality_regret`** — how much quality the router gave up vs. always
  using the baseline (0 or negative = router matched or beat it).
- **`total_cost_savings_usd`** — how much the router saved by not always
  using the baseline.
- **`worst_regret_items`** — the specific prompts where routing hurt quality
  the most, for debugging routing logic.

This roughly doubles request volume for the accuracy scenario, so it's opt-in.

## Comparing multiple providers

Since routerbench is provider-agnostic, comparing your router against
competing services is just running the same dataset against multiple
configs and lining up the results — `routerbench compare-run` does this in
one step:

```bash
routerbench compare-run \
  --configs configs/hugeai.yaml configs/openrouter.yaml configs/martian.yaml \
  --labels hugeai openrouter martian
```

This runs each config **sequentially** (never concurrently — each provider
has its own rate limits tuned for its own account; running them at once
would just make each look artificially slower under contention with the
others) against the same dataset, then writes a side-by-side
`comparison-*.json`/`.html` report: quality, cost, routing accuracy,
reliability, and latency/throughput at every concurrency level, one column
per provider. It warns if the configs point at different datasets, since
that would make the comparison meaningless.

If you'd rather run providers independently (different days, different
machines) and compare afterward, `routerbench compare` does the same
aggregation from already-saved `report-*.json` files instead of running
anything:

```bash
routerbench compare --reports reports/hugeai-report.json reports/openrouter-report.json \
  --labels hugeai openrouter
```

**Provider configs included as a starting point:**
- `configs/openrouter.yaml` — real, ready to use (OpenRouter's API is
  well-documented and stable). Requests `"openrouter/auto"`, their own
  meta-model for automatic routing — the fair comparison point against
  hugeai's `"auto"`.
- `configs/martian.yaml`, `configs/not_diamond.yaml`, `configs/portkey.yaml`,
  `configs/litellm_proxy.yaml` — **templates, not verified configs.** I
  don't have confident, current knowledge of these services' exact
  endpoints/auth schemes (they change over time, and LiteLLM Proxy is
  self-hosted with no fixed URL at all), so these have the routerbench-side
  settings fully wired up but placeholder `base_url`/`model` values clearly
  marked `REPLACE_ME` — each file's header comment explains exactly what to
  verify against that provider's current docs before running it. Not
  Diamond in particular may need a different integration entirely (see that
  file's comment) if their API is a routing *recommendation* rather than a
  full pass-through completions endpoint.

All provider configs point at the same `data/hugeai_prompts.jsonl` by
default — keep it that way for a fair comparison, or swap in your own
shared dataset across all of them.

## Running tests

```bash
pytest
```

Tests cover metrics math, all four scoring strategies, dataset loading, and
the HTTP client's retry/timeout/error handling (mocked with `respx` /
subprocess-only, no network needed).

## Extending

- **New scenario**: add `scenarios/my_scenario.py` following the shape of
  `scenarios/load.py`, wire it into `runner.py` and `config.py`.
- **New metric**: add a pure function in `metrics/` taking a
  `list[RequestResult]` and returning a dict.
- **New scoring strategy**: implement the async `Scorer` protocol
  (`routerbench/scoring/base.py`) and wire it into `CompositeScorer`'s
  dispatch order in `routerbench/scoring/__init__.py::build_scorer`.
