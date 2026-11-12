# Contract-Based Production Autonomy (CBPA)

**Contract-Based Production Autonomy: A Governance Framework for LLM-Driven Intent-Based Manufacturing**

CBPA is a five-layer governance framework that lets manufacturing managers express high-level
production intent (throughput, quality, human well-being) as formal *outcome contracts*, treats
Large Language Models as *bounded semantic agents* that contribute translation, generation,
orchestration, explanation, and reasoning across all five layers, and interposes a *verification
firewall* that certifies candidates through deterministic constraint checking and stochastic
simulation before software dispatch. Underlying controllers (MES, PLC, ROS 2, digital twins)
are treated as black boxes, and the prototype checks supported contracted quantities at its
declared observation points. Continuous physical stopping has not been validated.

The design principle is *deep integration at semantic boundaries, strict exclusion from
safety-critical computation*: LLMs are concentrated where natural-language understanding and
contextual reasoning connect managerial requests to executable contracts (intent translation,
candidate generation, escalation explanation), while verification, guard enforcement, and
adaptation triggers remain formal and deterministic.

This repository is the software and data artefact accompanying the paper. It contains the runnable
prototype, the integrated execution stack (Isaac Sim, Eclipse BaSyx AAS, OPC-UA), the experiment
configurations, and the archived result artefacts behind every data-derived table and figure.

## Paper

> Tianyu Wang, Sten Grahn, Zhihao Liu, Xi Vincent Wang and Lihui Wang.
> "Contract-Based Production Autonomy: A Governance Framework for LLM-Driven Intent-Based
> Manufacturing." *International Journal of Production Research*, accepted for publication (2026).

The manuscript itself is not distributed here. See [`CITATION.cff`](CITATION.cff) for machine-readable
citation metadata.

## Validation evidence

`data/validation/` holds the acceptance-check records of Supplementary Material S7 (fail-closed
deployment-boundary replay and verification completeness). These records, and the READMEs inside
them, were written in the development layout, where this repository root was `cbpa-case-study/`;
drop that prefix from the commands they give. The records of the other follow-up
studies (S12–S19) are in the source and data archive supplied with the article; their scripts are in
`scripts/` and expect those records under `data/validation/`. The study scripts take no options and
write into their fixed directories under `data/validation/`, replacing what is there, so run them in
a separate checkout.

## Layout

```
src/cbpa/            Five-layer implementation (L1-L5 + meta layer), physics models,
                     LLM clients, integration bridges, Streamlit UI
scripts/             Experiment runners, aggregation, baselines, figure generation
tests/               Unit, lifecycle, and end-to-end tests
docker/              BaSyx / Isaac Sim infrastructure for the integrated stack
data/                Archived experiment artefacts (see "Reproducing the paper")
docs/implementation.md  Detailed implementation and usage documentation
```

Start with [`docs/implementation.md`](docs/implementation.md) for the architecture walk-through,
scenario descriptions, configuration reference, and the full command list.

## Quick start

```bash
pip install -e ".[dev,export,figures]"      # Python 3.10+ (figures = matplotlib)

# Factory lifecycle - no LLM, no API key, no external services; runs to completion
python scripts/run_factory_experiment.py --no-llm

# Single-cell runs; the current runner stops both at phase 1 (see below)
python scripts/run_experiment.py --no-llm --shifts 3
python scripts/run_experiment.py --shifts 3          # LLM; Claude CLI auth: `claude login`

# Factory case, six-phase lifecycle
python scripts/run_factory_experiment.py

pytest -v
```

The `--integrated` flag routes Layer 4 through the Isaac Sim / BaSyx / OPC-UA bridges; start the
stack with `bash scripts/launch_integrated.sh`. A bridge whose service is unavailable runs in stub
mode, and execution then uses the analytical physics models. All archived `*_integrated` runs had
Isaac Sim and BaSyx in stub mode (see `bridges` in each `integrated_execution` audit record), so
their execution metrics come from the analytical models.

With no infrastructure and no API key you can run the test suite, the factory lifecycle in both
analytical and integrated mode, and every analysis script over the archived artefacts — that is,
every table and figure command below.

The current runner stops both single-cell runs at phase 1, before dispatch, by design:

- Without the LLM, the legacy scripted clarification answer adds a 3 kWh energy cap. The runner now
  encodes it, and no schedule can meet it: the model's fixed energy term alone is 92.75 kWh
  (Supplementary Material S12).
- With the LLM, clarification questions and hard-constraint disagreements between the LLM and the
  rule parser stay unresolved until confirmed (S12, S17). The command line supplies no reviewer;
  pass one through `CBPAExperiment(contract_reviewer=...)`.

The archived single-cell batches predate these checks. Release [`e21619c`](https://github.com/gunnerwang/Contract-Based-Production-Autonomy/tree/e21619c) (v1.1.0-r1) of
this repository runs the single-cell lifecycles as they were recorded: there the no-LLM run ends at
phase 5, where no deterministic anchor certifies `S3`, as in Table 6(a) (nominally feasible 1/1,
certified 0/1).

## Reproducing the paper

Two batches are the reference for the reported single-run numbers. Throughout the commands below:

```bash
S5=data/repeated/single_full_claude-sonnet-4-6_integrated_v5      # single cell, n=10, medoid run_01
F3=data/repeated/factory_claude-sonnet-4-6_analytical_v3          # factory,     n=10, medoid run_08
```

"Medoid" is the run closest to the per-phase medians of throughput, fatigue, and noise across the
batch; `scripts/paper_values.py` selects it and prints the table rows.

Table and figure numbers below follow the published article; the paper itself is not distributed
here. Every artefact the numbers refer to is already committed under `data/`, so the commands
regenerate what is in the repository rather than producing something new: figures sit next to their
run as PDF and PNG in a `figures/` directory, and tables are written as `.tex` and `.csv` alongside
the `.json` the scripts read. Table and figure numbers that appear in code comments and output file
names (for example `table3.json` or `fig11_stochastic_verification.pdf`) come from earlier drafts and
do not follow the published numbering.

### Tables

| Table | Content | Artefact / command |
|-------|---------|--------------------|
| 1–3 | Related work; layer interfaces and decision rights; enabling techniques | Conceptual; no run artefact |
| 4 | Implementation coverage | Assessed against `src/cbpa/`; no run artefact |
| 5(a) | Single-cell candidate KPIs, shift 1 | `python scripts/paper_values.py $S5 --shift 1` |
| 5(b) | Factory schedule comparison | `python scripts/paper_values.py $F3` |
| 5(c) | LLM vs. deterministic provenance | `python scripts/paper_values.py $S5 --shift 1` and `python scripts/paper_values.py $F3` — the `provenance:` lines |
| 6(a) | Candidate-pool ablation (no-LLM / hybrid / LLM-only) | `python scripts/summarize_ablation_acceptance.py` → `data/baselines/ablation_acceptance_summary.json` |
| 6(b) | Layer-2 baseline vs. constrained optimiser | `python scripts/planner_baselines.py data/baselines/planner --single-root $S5 --factory-root $F3 --single-nollm-root data/repeated/single_full_nollm_integrated --factory-nollm-root data/repeated/factory_nollm_analytical`; optimiser-anchor column from the `..._optanchor_...` batches |
| 7 | Contract-fidelity benchmark (30 intents) | `data/baselines/fidelity_llm_v2/results.json` (`python scripts/contract_fidelity_benchmark.py --llm`) |
| 8(a) | Factory repeated runs, n=10 | `python scripts/aggregate_repeated.py --root $F3` → `summary.json` |
| 8(b) | Additional disturbances and second model | `data/repeated/factory_supplydelay_claude-sonnet-4-6_analytical_v2`, `data/repeated/factory_supplydelay_only_claude-sonnet-4-6_analytical_v2`, `data/repeated/factory_gemini-3.8-flash_analytical_v2` |
| 8(c) | Single-cell repeated runs, n=10 | `python scripts/aggregate_repeated.py --root $S5` → `summary.json` |
| 9 | Sensitivity to twin mismatch and sensor noise | `python scripts/sensitivity_analysis.py data/sensitivity --single-root $S5 --factory-root $F3 --single-nollm-root data/repeated/single_full_nollm_integrated` |

`paper_values.py` and `summarize_ablation_acceptance.py` print to stdout or write JSON; the
booktabs versions are in the repository already:

| File | Holds |
|------|-------|
| `$S5/summary_table.tex`, `.csv` | Table 8(c) (single-cell repeated runs) |
| `$F3/summary_table.tex`, `.csv` | Table 8(a) (factory repeated runs) |
| `data/sensitivity/sensitivity_table.tex` | Table 9 |
| `<run>/table3.tex`, `table3.json` | Per-run single-cell phase metrics (Table 5(a) is the medoid run's) |
| `<run>/factory_table.json` | Per-run factory phase metrics (Table 5(b) is the medoid run's) |

In a multi-shift run directory, `llm_contribution.json` summarises the last shift. Table 5(c) reports
shift 1, so take the provenance counts from `paper_values.py`, which selects the shift explicitly.

### Figures

Figures 1 and 2 are conceptual diagrams. The data-derived figures of the article and its
Supplementary Material were rendered from the archived records by `scripts/render_paper_figures.py`.
The renderer reads only the two reference batches and `data/sensitivity/summary.json`, makes no model calls, and needs Times
New Roman:

```bash
python scripts/render_paper_figures.py /tmp/cbpa-figures
```

It reproduces the published files pixel for pixel and writes `figure_render_audit.json` with the
SHA256 of every input. The experiment scripts also write their own renderings next to each run; these
show the same records in the runners' layout, without three display corrections listed in the audit
(batch-medoid reference bars, S1 Monte Carlo feasibility shown as 99.5% rather than rounded to 100%,
and S3d labelled as an uncertified anchor).

| Figure | Shows | File written by the renderer | Per-run rendering and command |
|--------|-------|------------------------------|-------------------------------|
| 3 | Integrated execution-stack summary (lower part; the top-row photographs are not in this repository) | `fig_integrated_stack.pdf` | `python scripts/generate_integrated_figure.py --data $S5/run_01 --factory-data $F3/run_08 --out /tmp/cbpa-stack` (without `--out` it writes into `$S5/run_01/figures/`, whose archived copy is the single-case variant written during the run) |
| 4 | Stochastic verification of the cell schedules | `fig11_stochastic_verification.pdf` | `$S5/run_01/figures/fig11_stochastic_verification.pdf`: `python scripts/paper_values.py $S5 --export-stochastic $S5/run_01/stochastic_verification.json` then `python scripts/visualize_results.py --data-dir $S5/run_01` |
| 5 | Per-operator fatigue across factory schedules | `fig3_factory_operator_fatigue.pdf` | `$F3/run_08/figures/fig3_factory_operator_fatigue.pdf`: `python scripts/visualize_factory_results.py --data-dir $F3/run_08` |
| 6 | Firewall sensitivity to twin mismatch and noise | `fig_sensitivity.pdf` | `data/sensitivity/figures/sensitivity.pdf`: `python scripts/sensitivity_analysis.py` |
| S2 | Factory execution swimlane, medoid run | `fig4_factory_execution_swimlane.pdf` | Rendered from `$F3/run_08/execution_trace.json` |
| S8 | Multi-shift learning curve | `fig9_learning_curve.pdf` | `$S5/run_01/figures/fig9_learning_curve.pdf`: `python scripts/visualize_results.py --data-dir $S5/run_01` |
| S9 | Selected-schedule KPIs over the factory runs | `fig_repeated_factory.pdf` | `$F3/figures/variability_boxplots.pdf`: `python scripts/aggregate_repeated.py --root $F3` |
| S10 | Selected-schedule KPIs over the single-cell runs | `fig_repeated_single.pdf` | `$S5/figures/variability_boxplots.pdf`: `python scripts/aggregate_repeated.py --root $S5` |

The remaining supplementary figures are diagrams, interface screenshots and laboratory photographs.

### Supporting analyses

| Analysis | Artefact |
|----------|----------|
| H3 control (next-shift schedule under the unrevised contract) | `data/baselines/h3_control.json` (`scripts/h3_control.py`) |
| Structural-decision probe | `data/baselines/structural_probe.json` (`scripts/structural_probe.py`) |
| Fail-closed deployment-boundary replay | `data/validation/fail_closed/` (`scripts/replay_deployment_boundary.py`) |
| Verification completeness across lifecycles | `data/validation/verification_completeness/` (`scripts/validate_corrected_lifecycles.py`) |
| Longer-horizon learning (10 shifts) | `data/repeated/single_10shifts_claude-sonnet-4-6_integrated_v2` |
| Follow-up semantic, compilation, review, capacity and feedback studies (S12–S19) | Source and data archive supplied with the article |

## Archived batches

`data/results_*` holds single-run outputs, kept as examples of a complete exporter bundle. The
`_paper` suffix on `data/results_factory_integrated_paper` is a legacy label from an earlier
methodology and does not identify the source of the current performance tables. The tables above
identify their repeated-run sources.

`data/repeated/` holds 24 batches (189 runs). Batch directories carry the case, model, execution
mode, and a version suffix; earlier batches are kept alongside the reported ones because the paper's
Supplementary Material S7 discusses them.

| Batch | Runs | Role |
|-------|------|------|
| `single_full_claude-sonnet-4-6_integrated_v5` | 10 | Single-cell reference (medoid `run_01`) |
| `factory_claude-sonnet-4-6_analytical_v3` | 10 | Factory reference (medoid `run_08`) |
| `single_full_nollm_integrated`, `factory_nollm_analytical` | 1 each | No-LLM references (deterministic, one run by construction) |
| `*_llmonly_*` | 10 each | LLM-only condition (anchors withheld) |
| `*_optanchor_*` | 10 each | Certified optimiser schedule added to the anchor pool |
| `factory_gemini-3.8-flash_analytical_v2` | 10 | Cross-model check |
| `factory_supplydelay_*_v2` | 5 each | Additional disturbance combinations |
| `single_10shifts_*_v2` | 3 | Longer-horizon learning |
| `_v1_...`, `_v2_...` | 10 each | Earlier batches retained for the record (Supplementary Material S7) |
| `*_v2`, `*_v3`, `*_v4`, `pilot_*` (others) | varies | Earlier batches, superseded by the versions above |

Each `run_XX/` contains `run_meta.json` (model, sampling settings, Monte Carlo seed base, prompt hash,
git commit), `llm_calls.json` (per-call latency, exceptions, empty or schema-invalid outputs and the
deterministic fallbacks they triggered), `audit_trail.json`, `execution_trace.json`, the phase tables,
and `figures/`.

## Reproducibility scope

Within one release the deterministic paths (`--no-llm`, the anchors, the constraint checker, the
Monte Carlo certification at a fixed seed base) are repeatable; across releases, later corrections
change some outcomes, such as the single-cell phase-1 refusal above. The LLM paths are not
repeatable: candidates vary between calls, and the batches above are the record of that variation
rather than a fixed output. Two further
limits apply to the archived batches, both discussed in the paper's Supplementary Material S7 and S11:
full LLM responses were not retained for the historical batches, and historical factory sub-seeds
depended on Python string hashes and were therefore process-dependent. Those trajectories are
therefore not replayed exactly, and the corrected runners intentionally refuse some historical
continuations. Re-running the commands above produces new batches under the same protocol rather
than the archived numbers.

Each `run_meta.json` and batch `summary.json` records the `git_commit` of the working repository at
the time of the run. Those 17 commit ids belong to the development history and are not part of this
repository's history; they identify runs relative to each other rather than pointing at code that can
be checked out here. The S7 lifecycle records in `data/validation/verification_completeness/` were
produced by release `e21619c`; rerun `scripts/validate_corrected_lifecycles.py` there.

The archived artefacts are published as written by the experiment runners, with one modification:
absolute local paths in the `reference_path` field of `data/repeated/*/summary.json` were rewritten
as repository-relative paths.

## LLM providers

| Provider | Auth | Extra | CLI flag |
|----------|------|-------|---------|
| Claude (default) | `claude login` (CLI auth) | included | `--provider claude` |
| OpenAI-compatible | `OPENAI_API_KEY` (and `OPENAI_BASE_URL` for other endpoints) | `pip install -e ".[openai]"` | `--provider openai` |

Deterministic mode (`--no-llm`) needs neither. Copy `.env.example` to `.env` to
configure a key; the file is git-ignored.

## Dashboard

`streamlit_app.py` is the Streamlit entry point for browsing results, the audit trail and live phase
progress:

```bash
pip install -e ".[ui]"
streamlit run streamlit_app.py
```

`scripts/run_full_demo.sh --with-dashboard` starts it alongside a run.

## Citation

If you use this software, please cite both the paper and the software release. `CITATION.cff` at the
repository root carries both records.

## Funding

This work was supported by the Swedish Research Centre of Excellence in Production Research (XPRES)
and the collaborative research programme between KTH Royal Institute of Technology and the Research
Institutes of Sweden (RISE).

## License

MIT — see [`LICENSE`](LICENSE).
