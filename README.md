# YouTube Fact Checker

An AI-assisted fact-checking prototype. Paste a YouTube link and it:

1. gets the video's transcript,
2. picks out up to **5 checkable factual claims**,
3. searches the web for evidence on each claim,
4. asks an LLM to judge what each source says,
5. cross-checks the LLM's reading of each source with a **local NLI model** (no API call), and
6. returns one of **three verdicts** and a **confidence score** for every claim, with the sources it relied on.

> **This is an academic prototype, not a production system.** Verdicts reflect the
> strength of the *retrieved evidence*, not absolute truth, and the confidence score
> is not a probability that the claim is true (see [Confidence score](#confidence-score)).

---

## How it works

```
YouTube URL
   │
   ▼
Transcript ──── official captions (youtube-transcript-api)
   │            └─ if none: download audio (yt-dlp) + local Whisper (faster-whisper)
   ▼
Claim extraction ──── Groq LLM → max 5 claims, each with search queries
   │                  (timestamps come from the transcript, never from the LLM)
   ▼
Evidence search ──── Tavily, queries run in parallel
   │                 de-duplicate by publisher → rank by credibility → max 5 sources/claim
   ▼
Verdicts ──── Groq LLM, claims batched per call. For every source it returns structured
   │          judgments (relevance, stance, directness) and a verdict. No confidence number.
   ▼
NLI second opinion ──── local DeBERTa MNLI judges the strongest (claim, snippet) pairs as
   │                        entailment / neutral / contradiction (one batched call, CPU)
   ▼
Confidence ──── computed in Python from the judgments + source credibility
   │            + a small, bounded adjustment for NLI agreement with the LLM's stance
   ▼
Streamlit UI ──── verdict, confidence bar, summary, explanation, cited sources
```

### Verdicts

| Verdict | Meaning |
|---|---|
| 🟢 **TRUE** | relevant evidence clearly confirms the claim as stated |
| 🔴 **FALSE** | relevant evidence clearly contradicts the claim |
| ⚪ **UNVERIFIABLE** | no relevant evidence, not enough to judge, or the evidence is **mixed / only partly supports** the claim — always confidence **0** |

Mixed or partly supporting evidence is deliberately folded into UNVERIFIABLE. This keeps the
label set to three outcomes (like FEVER's supported / refuted / not-enough-info) and avoids the
inconsistent "partially true" vs "misleading" boundary.

### Guard rails on the LLM

- It may only cite sources that were actually retrieved for *that* claim. The LLM returns **only
  the source URLs**; title, publisher and snippet are filled in from our own retrieved data, so a
  quote can never be fabricated. URLs that don't match a retrieved source are dropped.
- A definite verdict with no valid citation is downgraded to **UNVERIFIABLE**.
- An unrecognised verdict label is treated as **UNVERIFIABLE**.
- It never invents timestamps: it points at numbered transcript segments and the real
  start/end times are looked up from the transcript.
- If its JSON is malformed, the call is retried once with a stricter reminder; after that the
  claim is marked UNVERIFIABLE instead of crashing.

### When there are no sources

| Situation | Result |
|---|---|
| The search itself failed (technical problem) | UNVERIFIABLE, confidence 0 |
| Search worked but returned **nothing at all** (e.g. a name that looks invented or mis-transcribed) | FALSE with a fixed, low confidence of **0.45** — a deliberate heuristic, not produced by the formula |
| Search returned results but none were usable | UNVERIFIABLE, confidence 0 |

### Source credibility

Sources are scored **after** retrieval (the whole web is searched, nothing is whitelisted),
then ranked so the strongest evidence is shown to the LLM first. Unknown domains are kept but
rank lowest.

| Tier | Type | Examples | Weight in score |
|---|---|---|---|
| 1 | Government / official | nasa.gov, who.int, cdc.gov, nih.gov, un.org | 1.00 |
| 2 | Scientific / academic | nature.com, sciencedirect.com, britannica.com, arxiv.org | 0.85 |
| 3 | Reputable news | reuters.com, apnews.com, bbc.com, theguardian.com | 0.65 |
| 4 | Fact-check / reference | snopes.com, politifact.com, factcheck.org, wikipedia | 0.45 |
| 5 | Unrated | everything else | 0.25 |

The domain lists live in `factcheck/evidence/credibility.py`.

---

## Confidence score

The LLM judges *what each source says*; **Python computes the number**, so the same judgments
always give the same score and every part of it can be inspected.

For each source, the LLM returns (all on small fixed scales, which are more consistent than free-form decimals):

- **relevance** 0–3 — how directly it addresses the claim
- **stance** — `supports`, `contradicts`, `partial` or `neutral`
- **directness** 0–3 — how explicitly it states the specific fact

Each is normalised to 0–1 and combined with the source's credibility tier into a "mass"
`w = relevance × credibility × directness`. Only sources that are relevant, direct and
non-neutral count. Sources whose stance matches the verdict are the *aligned* ones
(TRUE → `supports`, FALSE → `contradicts`).

Four components, each 0–1:

| Component | Weight | Definition |
|---|---|---|
| **Relevance** | 0.20 | mean relevance of the aligned sources |
| **Credibility** | 0.25 | relevance-weighted mean credibility of the aligned sources |
| **Strength** | 0.35 | *dominance* × mean directness, where dominance = aligned mass ÷ (aligned + opposing mass) |
| **Agreement** | 0.20 | independent aligned domains ÷ 3, capped at 1 |

```
confidence = 0.20·relevance + 0.25·credibility + 0.35·strength + 0.20·agreement
```

Edge cases, handled explicitly:

- **Conflicting evidence** — if opposing evidence exists and dominance is below 0.75,
  confidence is **capped at 0.60**, however good the supporting sources are.
- **Never fully certain** — confidence is capped at **0.95**.
- **No usable evidence** — a TRUE/FALSE verdict with no aligned source is downgraded to UNVERIFIABLE (0).
- **UNVERIFIABLE** — always 0.
- Two sources from the same registrable domain (e.g. `news.bbc.co.uk` and `www.bbc.co.uk`)
  count as **one** independent source.

### NLI second opinion

A local MNLI model independently judges the strongest `(claim, source snippet)` pairs
(premise = snippet, hypothesis = claim). Its agreement with the LLM's stance nudges the
confidence up or down. **The LLM never sets the final number**, and the NLI model never changes
the verdict label.

| LLM stance | NLI label it should match |
|---|---|
| `supports` | entailment |
| `contradicts` | contradiction |
| `partial` | neutral (no agreement signal) |
| `neutral` | neutral (not counted) |

Per aligned source, agreement `a` is in [-1, 1]:

```
supports     a = P(entailment)    − P(contradiction)
contradicts  a = P(contradiction) − P(entailment)
partial      a = 0
|a| < 0.20   a = 0        (neutral / ambiguous NLI gives no boost)
```

Agreement is averaged over the NLI-scored aligned sources (at most `NLI_MAX_PAIRS_PER_CLAIM`
per claim), weighted by the same mass `w`, then applied:

```
A    = Σ(w·a) / Σ(w)
adj  = +0.08 · A   if A ≥ 0        (max boost +0.08)
       +0.15 · A   if A < 0        (max penalty −0.15)
confidence = clamp( min(raw + adj, 0.60 if conflict-capped), 0, 0.95 )
```

- **Speed:** NLI runs only on the top aligned sources per claim, ranked by relevance then
  directness. All pairs in a Groq batch go through one batched call, sorted by length to reduce
  padding. Identical `(snippet, claim)` pairs are inferred once and remembered for the life of the
  process. The model is loaded once per process.
- Disagreement costs more than agreement earns, on purpose.
- The conflict cap and the 0.95 ceiling still apply after the adjustment.
- NLI only looks at aligned sources, so it can't rescue a verdict with no aligned evidence.
- If NLI is disabled or unavailable (missing packages, download failure), the adjustment is 0 and
  the score is exactly the original formula. The failure is logged, not raised.
- Per-source NLI labels and probabilities are in `confidence_breakdown["per_source"][i]["nli"]`
  and the aggregate is in `confidence_breakdown["nli"]` (`agreement`, `adjustment`, `sources_used`).

**Worked example** — three sources back a TRUE verdict: a Tier-1 source (relevance 3, directness 3),
a Tier-2 source (3, 2) and a Tier-3 source (2, 2), from three different domains:

| relevance | credibility | strength | agreement | **confidence (before NLI)** |
|---|---|---|---|---|
| 0.889 | 0.856 | 0.778 | 1.000 | **0.864** |

Each claim card in the app has a **"How was this confidence calculated?"** expander showing
these components, and the same numbers are written to the log (see [Logging](#logging)).

> The weights and thresholds are heuristic design choices, not fitted to data. They are all
> in one place, `factcheck/scoring/constants.py`, and are easy to tune. The score expresses how well
> the evidence supports **the verdict reached**, not how likely the claim is to be true.

---

## The app

- **Analyze Video** — runs the pipeline. Results stay on screen across Streamlit reruns.
- **Cooldown** — after an analysis that needs fresh API calls, the button is disabled for
  **60 seconds** (to protect free-tier Groq/Tavily limits). The button's colour fills
  **left → right** like a progress bar, and it re-enables when it is full. The cooldown survives reruns.
- **Cached analyses skip the cooldown** — if the transcript, claims and every claim's verdict
  are already cached, no API calls are needed, so the button is never locked for that video.
- **Use cached results** (sidebar) — turn it off for a clean re-run.
- **Clear cache** (sidebar) — deletes the cache; a "Cache cleared." notice fades away by itself.

### Caching

A small JSON file at `.cache/fact_checker_cache.json` stores three things:

| Namespace | Key | Contents |
|---|---|---|
| `transcript` | video ID | segments + how they were obtained |
| `claims` | video ID | the extracted claims |
| `claim_results` | hash of the claim text | sources, search metadata and the verdict (with its confidence breakdown) |

Caching speeds up demos and makes evaluation fair: you can re-run the same claims and evidence
while changing only one stage. **Clear the cache after changing the verdict labels, the scoring
formula or the NLI setting** — older cached results keep their old verdicts and confidence.

---

## Evaluation

Two scripts run the pipeline in bulk and write an Excel file (needs `openpyxl`).

| Script | What it does |
|---|---|
| `benchmark.py` | Runs every YouTube URL in `urls.txt` through the full pipeline. Saves after each video, waits between videos for the free-tier limits and retries on Groq `429` rate-limit errors. |
| `claims_benchmark.py` | Runs 25 hand-written claims (15 known-false, 10 mixed-evidence) through search + verdicts, with the gold label pre-filled — used to balance the classes. |

```bash
pip install openpyxl
python benchmark.py urls.txt results.xlsx            # add --no-cache for a clean re-run
python claims_benchmark.py results_synthetic.xlsx
```

The **Claims** sheet has one row per claim: verdict, confidence, **confidence without NLI**,
NLI agreement and adjustment, sources and summary. Fill the `gold_label` column (dropdown:
TRUE / FALSE / UNVERIFIABLE); the `correct` column and the **Metrics** sheet (accuracy, per-class
precision and recall, mean confidence for correct vs wrong verdicts, with and without NLI) update
themselves. One run is enough for the NLI comparison, because the without-NLI confidence is
recomputed from the stored breakdown.

### Results (65 labelled claims)

40 claims came from 8 educational YouTube videos and 25 are synthetic stress-test claims.
Gold distribution: 36 TRUE, 16 FALSE, 13 UNVERIFIABLE.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| TRUE | 0.94 | 0.81 | 0.87 | 36 |
| FALSE | 0.71 | 0.94 | 0.81 | 16 |
| UNVERIFIABLE | 0.46 | 0.46 | 0.46 | 13 |

Accuracy **0.769** (50/65), macro-F1 **0.713**. Video claims alone: 0.775; synthetic claims: 0.760.

| Gold \ Predicted | TRUE | FALSE | UNVERIFIABLE |
|---|---|---|---|
| TRUE | 29 | 1 | 6 |
| FALSE | 0 | 15 | 1 |
| UNVERIFIABLE | 2 | 5 | 6 |

Mean confidence is about 0.66 for correct verdicts and 0.37 for incorrect ones. The NLI
adjustment was active on only 9 of these claims in this run (+0.040 mean, NLI never disagreed),
so its effect on correct-vs-wrong separation is **not yet established** — re-run with
`NLI_ENABLED=1` and `--no-cache` for a full comparison.

Main error types: true claims returned as UNVERIFIABLE (the snippets lack the exact figure),
and mixed-evidence claims judged FALSE (the FALSE / UNVERIFIABLE boundary is partly a labelling
choice). The benchmark is small, uses one set of labels and 25 of its claims are synthetic, so
treat the numbers as indicative.

---

## Setup

**Requirements:** Python 3.10+, [ffmpeg](https://ffmpeg.org/) on your PATH (only used by the Whisper
fallback), a [Groq](https://console.groq.com/) API key and a [Tavily](https://tavily.com/) API key.

```bash
python -m venv venv
# Windows:  venv\Scripts\activate        macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
pip install openpyxl        # only for benchmark.py / claims_benchmark.py
```

Create a `.env` file in the project root (never commit it):

```env
GROQ_API_KEY=your-groq-key
TAVILY_API_KEY=your-tavily-key
NLI_ENABLED=1
```

Run it:

```bash
streamlit run app.py
```

(Needs Streamlit ≥ 1.39. The first Whisper fallback downloads the small `tiny` model.)

**NLI model:** `requirements.txt` includes `torch`, `torchvision`, `transformers` and
`sentencepiece`. The first analysis downloads the NLI model (about 0.7 GB, cached by Hugging Face
afterwards) and loads it once per process, so that first claim is slower. It runs on CPU and needs
roughly 1.5 GB of RAM. `torchvision` must match your `torch` version; check with
`python -c "import torch, torchvision, transformers; print(torch.__version__, torchvision.__version__, transformers.__version__)"`.
To skip NLI entirely, set `NLI_ENABLED=0` (in `.env`, then restart the app).

**Free-tier rate limits:** Groq's free tier allows 8,000 tokens per minute for the default model, so
back-to-back verdict calls can return `429`. The SDK retries automatically and `benchmark.py` waits
and retries; lowering `max_tokens` in `factcheck/verdicts/checker.py` also helps.

### Settings

| Setting | Where | Default | Notes |
|---|---|---|---|
| `GROQ_API_KEY`, `TAVILY_API_KEY` | `.env` | — | required |
| `GROQ_MODEL` | `.env` | `openai/gpt-oss-120b` | model used for verdicts (claim extraction uses the same default) |
| `LOG_LEVEL` | `.env` | `INFO` | `DEBUG` also logs per-source scoring rows |
| `NLI_ENABLED` | `.env` / shell | `1` | `0` skips NLI; scoring falls back to the original formula |
| `NLI_MODEL` | `.env` / shell | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | any 3-way MNLI checkpoint; label order is read from the model config |
| `NLI_DEVICE` | `.env` / shell | `cpu` | e.g. `cuda` if available |
| `NLI_MAX_PAIRS_PER_CLAIM` | `factcheck/config.py` | 2 | top aligned snippets per claim sent to NLI (1 is faster) |
| `NLI_BATCH_SIZE`, `NLI_MAX_LENGTH` | `factcheck/config.py` | 16 / 512 | pairs per forward pass / max tokens (256 is faster) |
| `ANALYZE_COOLDOWN_SECONDS` | shell environment | `60` | `0` disables the cooldown, e.g. PowerShell: `$env:ANALYZE_COOLDOWN_SECONDS=0; streamlit run app.py` |
| `MAX_CLAIMS` | `factcheck/config.py` | 5 | hard cap on claims per video |
| `MAX_RESULTS_PER_QUERY`, `MAX_SOURCES_PER_CLAIM`, `MAX_CONCURRENT_SEARCHES`, `MAX_SNIPPET_CHARS` | `factcheck/config.py` | 3 / 5 / 2 / 800 | evidence retrieval limits |
| `BATCH_SIZE` | `factcheck/config.py` | 2 | claims per Groq verdict call |
| `WHISPER_MODEL_SIZE` | `factcheck/config.py` | `tiny` | speech-to-text model |
| scoring weights, caps, NLI boost/penalty/dead zone | `factcheck/scoring/constants.py` | see above | |
| UI cooldown / notice timing | `ui/config.py` | 60 s / 3.5 s | |

---

## Logging

Every stage logs a one-line, greppable record (`CACHE hit/miss`, `GROQ call/success/failure`,
`TAVILY …`, `SEARCH complete`, `BATCH start/complete`, `VERDICT`, …). Confidence adds one line per claim:

```
CONFIDENCE | claim_id=1 | relevance=0.89 | credibility=0.86 | strength=0.78 | agreement=1.00 | dominance=1.00 | weighted_sum=0.86 | conflict_capped=False | final=0.86
```

---

## Project layout

```
app.py                  Streamlit entry point (a few lines)
benchmark.py            bulk-run videos from urls.txt -> Excel with metrics
claims_benchmark.py     bulk-run synthetic FALSE / mixed claims -> Excel
urls.txt                YouTube URLs for benchmark.py

factcheck/              the whole backend (no Streamlit anywhere in here)
  config.py               every backend setting: keys, models, limits, batch size
  models.py               Claim, Source, CitedSource, Verdict (3 verdict labels)
  llm.py                  the one Groq client + logged chat call
  cache.py                JSON disk cache (.cache/ in the project root)
  logging_utils.py        the shared logger
  transcript/             getting the transcript
    segments.py             segment format shared by every source
    youtube.py              URL validation, video ID, official captions
    whisper.py              audio download + local Whisper fallback
  claims/                 claim extraction
    prompt.py               the extraction prompt
    extractor.py            extract_claims + validation of each proposed claim
  evidence/               evidence retrieval
    credibility.py          Tier 1-5 domain lists
    search.py               Tavily queries, parallel run, de-dup, ranking
  verdicts/               LLM verdicts
    prompt.py               the verdict prompt + the claim/source text builder
    checker.py              one Groq call per batch, with one retry
    parsing.py              JSON -> Verdict: real citations only, safe downgrades
    no_sources.py           verdicts when search found nothing (no LLM call)
  scoring/                deterministic confidence
    constants.py            weights, tier credibility, caps
    assessments.py          the LLM's per-source judgments + validation
    confidence.py           the formula (incl. the NLI adjustment)
    nli.py                  local NLI model (lazy, cached), stance mapping, agreement
  pipeline/               orchestration + caching - the only API the UI uses
    transcripts.py          transcript cache -> captions -> Whisper
    claims.py               claims cache / extract / save
    verdicts.py             evidence search + batching + caching
    recording.py            logging and caching of each finished verdict
    cache_check.py          "is this whole video already cached?"

ui/                     the front end, one concern per file
  config.py               cooldown length, notice duration, widget keys
  timing.py               pure cooldown arithmetic
  styles.py               CSS for the filling button and the fading notice
  state.py                everything kept in st.session_state
  sidebar.py              cache controls
  controls.py             URL box, Analyze button, cooldown logic
  analysis.py             runs the pipeline for one URL
  results.py              draws the results

tests/                  no network, no API keys needed
```

**Import direction is one-way:** `pipeline` -> `transcript`, `claims`, `evidence`, `verdicts` ->
`scoring`, `llm`, `models`, `config`, `cache`, `logging_utils`. Nothing imports upward, and the UI
only talks to `factcheck.pipeline` (plus the URL helpers in `factcheck.transcript`). The two
benchmark scripts are the only other callers of `factcheck.pipeline`.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests need no API keys or network access - Groq, Tavily, YouTube, Streamlit and the NLI
model are replaced with fakes (`NLI_ENABLED=0` is set for the test run, so the real model is never loaded).

| File | Covers |
|---|---|
| `test_scoring.py` | the confidence formula (hand-checked), conflict cap, edge cases, and the batched pipeline end to end with caching |
| `test_claims.py` | claim validation, real timestamps, the 5-claim cap, query fallbacks, JSON retry |
| `test_evidence.py` | de-duplication, caps, ranking, metadata, concurrency limit, failure handling |
| `test_verdicts.py` | prompt layout, citation validation, batching, retry, reordering |
| `test_nli.py` | NLI labels, Groq↔NLI mapping, agreement/disagreement effect, 0–0.95 bound, conflict cap, no-NLI fallback, top-pair selection, de-dup/memoisation, scoring unchanged by the optimisation (model mocked) |
| `test_transcript.py` | URL parsing, segment format, official captions |
| `test_cache_check.py` | the "fully cached" check behind the cooldown bypass |
| `test_ui.py` | cooldown lifecycle, cached bypass, fading notice |

## Limitations

- Evidence comes from **search-result snippets** (up to 800 characters), not whole pages.
- The LLM's per-source judgments are consistent but not perfectly deterministic (low temperature,
  not zero); only the arithmetic on top of them is.
- Claim extraction depends on transcript quality; Whisper's `tiny` model trades accuracy for speed.
- NLI sees the same short snippets the LLM does. MNLI models can be wrong on numbers, dates,
  negation and claims that need outside context, so NLI is a bounded second opinion
  (−0.15 to +0.08), not a verifier.
- Mixed evidence is reported as UNVERIFIABLE, so the system is deliberately conservative: it
  sometimes abstains on claims that are true, and low-credibility sources can still mislead it.
- The benchmark is small (65 claims), uses one set of labels, and includes synthetic claims.
- Maximum 5 claims per video, by design.
- Heuristic weights and credibility tiers reflect design judgment, not empirical calibration.
