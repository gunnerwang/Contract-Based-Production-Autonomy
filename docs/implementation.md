# CBPA Case Study

A runnable instantiation of the **Contract-Based Production Autonomy (CBPA)** framework on two scenarios:

1. **Single-cell** — an automotive collaborative assembly cell with two robots and one human operator, exercised through a 5-phase lifecycle (optionally 6 with macro-adaptation), repeated across three shifts.
2. **Factory-scale** — a two-cell factory with four robots, three operators (one inspection-only backup), three product variants, and AGV-mediated cross-cell routing, exercised through a six-phase lifecycle ending in shift-close legislator handover.

Both scenarios run on the same five-layer CBPA stack (L1–L5 + Meta layer) and can be executed analytically, through SimPy discrete-event simulation, or through the **integrated execution stack** (Isaac Sim digital twin + Eclipse BaSyx AAS + OPC-UA + Streamlit dashboard). A **blend-and-verify** planner combines LLM-generated candidates with calibrated deterministic anchors, and a **no-LLM ablation** path produces the comparison baseline used in the paper.

This document is the detailed reference: architecture, scenarios, configuration, the full command
list, and the output artefacts. For what the repository is, how to install it, and the mapping from
each paper table and figure to the command that produces it, see the [README](../README.md).

## Quick Start

```bash
# Install (Python 3.10+, from the repository root)
pip install -e ".[dev,export,figures]"

# --- Single-cell case (1 or 3 shifts) ---
# Deterministic baseline (no API key needed).  Ends at phase 5, where no
# deterministic anchor certifies S3 and the firewall holds the deployment;
# this is the no-LLM result reported in Table 9.
python scripts/run_experiment.py --no-llm --shifts 3

# LLM-empowered (requires Claude CLI auth)
claude login            # one-time setup
python scripts/run_experiment.py --shifts 3

# Full integrated execution stack (Isaac Sim + BaSyx + OPC-UA)
python scripts/run_experiment.py --integrated --shifts 3

# --- Factory case (six-phase lifecycle) ---
python scripts/run_factory_experiment.py --integrated

# --- LLM vs no-LLM ablation (writes the baseline used in Table 9) ---
python scripts/run_experiment.py --no-llm --integrated --shifts 3 \
    --output-dir data/results_nollm_integrated_3shifts
python scripts/run_factory_experiment.py --no-llm --integrated \
    --output-dir data/results_factory_nollm_integrated

# --- All-in-one shell launcher (matches the paper full demo) ---
bash scripts/run_full_demo.sh                            # single-cell, default
bash scripts/run_full_demo.sh --factory                  # factory case
bash scripts/run_full_demo.sh --integrated --with-dashboard

# Run tests
pytest tests/ -v
```

## Repeated Runs (LLM variability and reproducibility)

LLM-generated candidates vary between runs. `scripts/run_repeated.py` runs a
lifecycle *N* independent times (fresh LLM client per run) and records, per run,
the selected schedule and its KPIs for every planning phase, the firewall
verdicts, the full candidate pool with provenance, every LLM call (latency,
exceptions, empty / schema-invalid outputs) and the deterministic fallbacks
they triggered, together with the model, sampling settings, Monte-Carlo seed
base, prompt hash and git commit. `scripts/aggregate_repeated.py` then reports
mean / SD / 95 % CI per phase, strategy stability (same candidate, same
parameter vector within tolerance, KPI-equivalent), firewall pass rates by
provenance, and LLM call statistics, as `summary.json`, a CSV, a booktabs
LaTeX table and figures.

```bash
# 10 LLM runs of the factory lifecycle (Monte-Carlo seeds fixed → only LLM sampling varies)
python scripts/run_repeated.py --case factory --runs 10

# 10 runs of the 3-shift single-cell full demo, also varying the Monte-Carlo seeds
python scripts/run_repeated.py --case single --full-demo --runs 10 --vary-mc-seed

# deterministic reference: identical across runs by construction
python scripts/run_repeated.py --case factory --runs 3 --no-llm

# resume an interrupted batch / re-aggregate
python scripts/run_repeated.py --case factory --runs 10 --resume --output-root data/repeated/factory_claude-sonnet-4-6_analytical_v3
python scripts/aggregate_repeated.py --root data/repeated/factory_claude-sonnet-4-6_analytical_v3
```

Outputs land in `data/repeated/<case>_<model>_<execution>[_varyseed]/run_XX/`
(the usual exporter files plus `run_meta.json` and `llm_calls.json`) with the
aggregate `summary.*` and `figures/` in the batch root. The Monte-Carlo seed
base is exposed as `mc_seed_base` on both experiment classes (default 42, which
reproduces the single-run results in the paper). The Claude path uses the Agent
SDK, which does not expose a temperature parameter (provider-default sampling);
the OpenAI path uses `temperature=0.2`.

## Extended Experiments

Additional analyses reported in Sections 5.4-5.7; none of them needs a running Isaac Sim / BaSyx / OPC-UA stack.

```bash
# Sensitivity of certification to twin mismatch and sensor noise (Section 5.7)
python scripts/sensitivity_analysis.py                     # -> data/sensitivity/

# Layer-2 baseline: constrained optimiser vs hybrid planner vs anchors (Section 5.5)
python scripts/planner_baselines.py                        # -> data/baselines/planner/

# Layer-1 baselines on the 30-intent contract-fidelity benchmark (Section 5.5)
python scripts/contract_fidelity_benchmark.py              # template + rule/ontology (no LLM)
python scripts/contract_fidelity_benchmark.py --llm        # + CBPA LLM elicitor (60 calls)

# Cross-model check (Section 5.6): Gemini 3.8 Flash via the OpenAI-compatible endpoint
#   export OPENAI_API_KEY=<gemini key> OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
python scripts/run_repeated.py --case factory --runs 10 --provider openai --model gemini-3.8-flash \
    --output-root data/repeated/factory_gemini-3.8-flash_analytical_v2

# Longer-horizon learning test (Section 5.6): 3 runs x 10 shifts (about 55 min per run with Claude)
python scripts/run_repeated.py --case single --full-demo --shifts 10 --integrated --runs 3 \
    --output-root data/repeated/single_10shifts_claude-sonnet-4-6_integrated_v2

# H2 LLM-only condition (Section 5.5, Table 9): anchors withheld, LLM candidates only
python scripts/run_repeated.py --case factory --runs 10 --llm-only --output-root data/repeated/factory_claude-sonnet-4-6_llmonly_analytical
python scripts/run_repeated.py --case single --full-demo --shifts 3 --integrated --runs 10 --llm-only \
    --output-root data/repeated/single_full_claude-sonnet-4-6_llmonly_integrated

# Optimiser as an anchor generator (Section 5.5, Table 10): certified optimiser schedule added to the pools
python scripts/run_repeated.py --case factory --runs 10 --optimiser-anchor --output-root data/repeated/factory_claude-sonnet-4-6_optanchor_analytical
python scripts/run_repeated.py --case single --full-demo --shifts 3 --integrated --runs 10 --optimiser-anchor \
    --output-root data/repeated/single_full_claude-sonnet-4-6_optanchor_integrated

# H3 control (Section 5.4): the next-shift schedule and anchor pool evaluated under the unrevised contract
python scripts/h3_control.py --factory-root data/repeated/factory_claude-sonnet-4-6_analytical_v3

# Structural-decision probe (Section 5.5): does the LLM reassign/reroute unaided?
python scripts/structural_probe.py --provider claude --calls 5
python scripts/structural_probe.py --provider openai --model gemini-3.8-flash --calls 5

# Additional disturbance scenarios (Section 5.6)
python scripts/run_repeated.py --case factory --runs 5 --supply-delay --output-root data/repeated/factory_supplydelay_claude-sonnet-4-6_analytical_v2
python scripts/run_repeated.py --case factory --runs 5 --supply-delay --no-operator-absence --output-root data/repeated/factory_supplydelay_only_claude-sonnet-4-6_analytical_v2

# Paper values, Figure 5 data, Layer-2 baseline (Table 10) and sensitivity (Table 15) from the batch medoid runs
S5=data/repeated/single_full_claude-sonnet-4-6_integrated_v5; F3=data/repeated/factory_claude-sonnet-4-6_analytical_v3
python scripts/paper_values.py $S5 --export-stochastic $S5/run_01/stochastic_verification.json
python scripts/paper_values.py $F3
python scripts/planner_baselines.py data/baselines/planner --single-root $S5 --factory-root $F3 \
    --single-nollm-root data/repeated/single_full_nollm_integrated --factory-nollm-root data/repeated/factory_nollm_analytical
python scripts/sensitivity_analysis.py data/sensitivity --single-root $S5 --factory-root $F3 --single-nollm-root data/repeated/single_full_nollm_integrated
python scripts/visualize_results.py --data-dir $S5/run_01            # Figures 5 and 6
python scripts/visualize_factory_results.py --data-dir $F3/run_08     # Figure 7 (operator fatigue)
python scripts/generate_integrated_figure.py --data $S5/run_01 --factory-data $F3/run_08  # Figure 4

# Batches reported in the paper (all produced after the last prototype fix; earlier batches are archived alongside):
#   single-cell  : single_full_claude-sonnet-4-6_integrated_v5 (n=10, medoid run_01), single_10shifts_..._v2 (n=3)
#   factory      : factory_claude-sonnet-4-6_analytical_v3 (n=10, medoid run_08), factory_gemini-3.8-flash_analytical_v2 (n=10)
#   scenarios    : factory_supplydelay_..._v2, factory_supplydelay_only_..._v2 (n=5 each)
#   no-LLM refs  : single_full_nollm_integrated, factory_nollm_analytical
```

The baselines live in `src/cbpa/baselines/` (`TemplateElicitor`, `RuleBasedElicitor`,
`OptimiserPlanner`, `FactoryOptimiserPlanner`). The single-cell runner canonicalises
elicited contracts to the plant ontology, clamps limits to the mandatory floors, and
cross-checks the LLM's limits against a rule-based parse of the same intent
(`_canonicalise_contract` in `src/cbpa/runner/experiment.py`).

## Architecture

The implementation mirrors the paper's five-layer architecture plus a cross-cutting meta layer. The same layer modules are reused by both single-cell and factory experiments.

```
src/cbpa/
  layer1_outcome/                 # L1: Outcome elicitation
    elicitation_agent.py            - LLM-assisted intent → contract translation
                                      with propose-and-refine ambiguity resolution
    contract_validator.py           - Schema and ontology validation

  layer2_planning/                # L2: Blend-and-verify planning
    plan_generator.py               - Single-cell PlanGenerator (LLM + deterministic
                                      anchors, blended through Pareto)
    pareto.py                       - Multi-objective non-dominance + provenance
    vr_scorer.py                    - Value/Resource index scoring

  layer3_verification/            # L3: Verification firewall
    constraint_checker.py           - Deterministic + Monte Carlo (n=200) checks
    feasibility_cert.py             - Issues feasibility certificates
    guard_synthesizer.py            - Synthesises runtime guards (clamp / escalate / warn)

  layer4_execution/               # L4: Execution & monitoring
    simulation.py                   - SimPy single-cell simulation
    monitor.py                      - Contract monitoring with guard enforcement
    assumption_tracker.py           - Typed assumption drift attribution

  layer5_adaptation/              # L5: Closed-loop adaptation
    adaptation_engine.py            - Three-tier hierarchy (micro / meso / macro)
    learning_store.py               - Experience memory with L5→L2 (single-cell)
                                      and L5→L1 (factory) feedback paths

  llm/                            # LLM access layer
    client.py                       - Provider-agnostic client (Claude Agent SDK /
                                      OpenAI-compatible) with query_text / query_structured
    prompts.py                      - Prompt templates for every LLM-assisted layer
    schemas.py                      - Expected response schemas and validation

  meta_layer/                     # Cross-cutting governance
    governance.py                   - Working-mode orchestration (Legislator /
                                      Auditor / Partner)
    escalation_agent.py             - Human-in-the-loop escalation with LLM-generated
                                      contextual explanations
    audit_trail.py                  - Granular per-layer audit log

  models/                         # Shared data models
    contract.py                     - OutcomeContract with typed Assumptions
    schedule.py                     - Schedule + FactorySchedule (both with `source`
                                      field for LLM/deterministic provenance)
    metrics.py                      - SimulationMetrics, FactoryMetrics, ParetoFrontResult
                                      (with candidate_sources mapping)

  physics/                        # Domain physics models
    cell_evaluator.py               - Single-cell throughput / fatigue / noise / energy
    factory_evaluator.py            - Multi-cell physics with cross-cell coupling
    fatigue_model.py                - ISO-based operator fatigue
    noise_model.py                  - Acoustic noise (logarithmic sum across cells)
    energy_model.py                 - Energy consumption
    defect_model.py                 - Quality / defect rate
    agv_model.py                    - AGV transfer dynamics for the factory case

  runner/
    lifecycle.py                    - Shared CBPA lifecycle base class
    experiment.py                   - Single-cell experiment + multi-shift runner
    factory_experiment.py           - Factory experiment with FactoryPlanGenerator
                                      (blend-and-verify) and 6-phase lifecycle
    results_exporter.py             - Single-cell CSV / LaTeX / JSON / provenance export
    factory_results_exporter.py     - Factory tables and provenance export
    live_writer.py                  - Streams phase results to live_demo.json

  service/integration/            # Real-system bridges (controller-agnostic)
    orchestrator.py                 - Single-cell IntegrationOrchestrator
    factory_orchestrator.py         - Factory FactoryOrchestrator
    isaac_bridge.py                 - Isaac Sim digital-twin bridge
    basyx_bridge.py                 - Eclipse BaSyx AAS bridge
    opcua_bridge.py                 - OPC-UA server (publishes KPIs/contract status)
    event_bus.py                    - Inter-bridge messaging

  ui/                             # Streamlit dashboard
    pages/2_Results_Explorer.py     - Single-cell results viewer
    pages/3_Live_Monitor.py         - Live phase progress
    pages/4_Audit_Trail.py          - Audit trail explorer
    pages/5_Integration_Status.py   - Bridge readiness
    pages/6_Factory_Dashboard.py    - Factory case dashboard

  config/scenario.py              - CellConfig, ScheduleParams, FactoryScenarioConfig
```

## Single-Cell Case Study

Eight-hour shift on a collaborative assembly cell (2 robots + 1 human operator) running through 5 phases (or 6 with macro-adaptation):

| Phase | Layer | Working Mode | What Happens |
|-------|-------|-------------|-------------|
| **1. Contract & Deployment** | L1→L4 | Legislator | LLM elicits and refines contract C1 (5 ambiguities resolved). Layer 2 generates 9 candidate schedules (4 LLM + 5 deterministic anchors). Pareto-filtered, stochastically verified. S1 deployed with runtime guards armed. |
| **2. Disturbance & Monitoring** | L4 | Auditor | Demand surges +20%. Assumption tracker attributes drift. Guards enforce safety bounds. Shortfall predicted. |
| **3. Fast Replan** | L2→L3 | Auditor | 9 aggressive S2 candidates generated (LLM blend). LLM "Low-Noise Aggressive" wins Pareto front but is rejected: P(feasible)=0% on fatigue and noise. |
| **4. Escalation** | Meta | Partner | LLM generates contextual conflict explanation (specific margin violations). Manager picks Option C; contract C2 pre-authorised. |
| **5. Adaptation & Stabilization** | L2→L5 | Auditor | L5 learning bias applied. 9 balanced S3 candidates generated. S3 deployed; experience recorded. |
| **6. Macro-Adaptation** *(optional)* | L1→L5 | Legislator | R1 mechanical degradation injected. Micro/meso fail. L1 re-elicits C3, S4 deployed for degraded conditions. |

Three-shift runs (`--shifts 3`) inject demand spikes of +20%, +15%, and +25% across consecutive shifts and exercise the L5→L2 learning loop.

## Factory Case Study

Two-cell factory with cross-cell coordination, exercising the framework at the production-system level:

```
+----------------------+    AGV    +-------------------------+
|       Cell A         | <-------> |        Cell B           |
| (assembly)           |           | (test & pack)           |
| R1 pick&place + R2   |           | R3 testing + R4 pack    |
| operator H1          |           | operator H2             |
+----------------------+           +-------------------------+
                  |             |
                  v             v
              backup operator H3 (inspection-only)
              variants: V_A, V_B, V_C
```

Six-phase lifecycle:

| Phase | Mode | Trigger | What Happens |
|-------|------|---------|-------------|
| 1 | Legislator | Shift start | C1_factory elicited; FS1 deployed at 84.4 uph |
| 2 | Auditor | Disturbance | Demand +20% AND H2 absent at hour 3 |
| 3 | Auditor | Cross-cell replan | LLM proposes "V_C_R2_Bottleneck_Relief" (96.2 uph). Rejected: H1=0.900>0.4, H3=0.708>0.4, noise=86.5>82 dB |
| 4 | Partner | Escalation | Manager picks Option D; C2_factory pre-authorised |
| 5 | Auditor | Stabilisation | FS3 deployed at 75.8 uph, all constraints satisfied |
| 6 | Legislator | Shift close | L5 surfaces 4 lessons → L1 rewrites C3_factory → FS4 pre-staged at 81.7 uph for next shift (proactive recovery via L5→L1 feedback) |

## Blend-and-Verify Planning

Both `PlanGenerator` (single-cell) and `FactoryPlanGenerator` use the same blend-and-verify pattern:

1. The LLM proposes a set of candidate schedules with a textual rationale.
2. A set of calibrated deterministic anchors (named `*_det`) is appended as a safety net.
3. The combined pool is evaluated by the same physics/constraint checker.
4. Pareto selection picks the best schedule **regardless of source**.
5. The verification firewall (Layer 3) blocks any schedule that violates hard constraints, no matter where it came from.

Each candidate carries a `source` field (`"llm"` or `"deterministic"`) that propagates through the Pareto result and into the audit trail. After every run, `data/results_*/llm_contribution.json` summarises per-phase provenance — how many candidates came from each source, how many survived the Pareto front, and which won the final selection.

To produce the no-LLM baseline used in the paper's ablation table, simply pass `--no-llm`. The factory case Phases 1, 5, and 6 produce **identical** schedules with or without the LLM (calibrated anchors win); only Phase 3 differs (LLM explores ~11% harder, both blocked by L3).

## Integrated Execution Stack

When invoked with `--integrated`, Layer 4 deploys schedules through real industrial-grade bridges instead of analytical evaluation:

| Service | Default endpoint | Purpose |
|---------|------------------|---------|
| Isaac Sim digital twin | `http://localhost:8211` | Physics-based cell execution |
| Eclipse BaSyx AAS registry | `http://localhost:9082` | Digital-twin interoperability, contract & KPI sync |
| OPC-UA server | `opc.tcp://localhost:4840/cbpa/` | Live KPI / constraint / contract status to PLC/SCADA clients |
| Streamlit dashboard | `http://localhost:8519` | Real-time monitoring (`--with-dashboard`) |

CBPA treats each bridge as a black-box runtime service: it sets operating points and reads back execution evidence. The factory lifecycle uses analytical execution when the services are absent, so it runs in integrated mode either way. Single-cell integrated runs read their KPI observations from the bridges; start the stack with `bash scripts/launch_integrated.sh` before running them. The OPC-UA bridge auto-cleans stale port-4840 processes on start-up and registers an `atexit` shutdown hook to release the port cleanly.

Use `bash scripts/launch_integrated.sh` to start the Docker-based BaSyx + Isaac Sim infrastructure before running the experiment.

## Output Artefacts

Each experiment writes a complete bundle to `data/results_*/`:

| File | Description |
|------|-------------|
| `audit_trail.json` | Every layer invocation tagged with phase, working mode, and details |
| `execution_trace.json` | Phase-level summary mapped to active layers and contract state |
| `working_modes.json` | Working-mode lifecycle and transitions |
| `table3.json` / `factory_table.json` | Schedule metrics per phase (paper Tables 6 / 7; legacy `table3` naming) |
| `multi_shift_learning_curve.json` | V/R, violations, experience records per shift |
| **`llm_contribution.json`** | **Per-phase LLM vs deterministic provenance breakdown. In a multi-shift run this file describes the last shift; use `scripts/paper_values.py --shift N` for a specific shift (paper Table 8 reports shift 1)** |
| `live_demo.json` | Phase-by-phase live log consumed by the dashboard |
| `figures/` | PDF and PNG versions of the paper figures |

## Configuration

All scenario parameters live in `src/cbpa/config/scenario.py`. Key flags:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `demand_spike_pct` | 20.0 | Demand surge magnitude (%) |
| `enable_macro_phase` | False | Enable Phase 6 macro-adaptation demo (single-cell) |
| `n_shifts` | 1 | Number of shifts for the learning curve |
| `use_simulation` | False | Run SimPy discrete-event simulation for deployed schedules |
| `use_integrated` | False | Use Isaac Sim + BaSyx + OPC-UA execution stack |
| `enable_operator_absence` | True | (factory) Inject H2 absence at `operator_absence_hour` |
| `vr_value_weights` | [0.50, 0.30, 0.20] | V/R value weights (throughput, quality, flexibility) |
| `vr_resource_weights` | [0.35, 0.25, 0.20, 0.20] | V/R resource weights (energy, downtime, labor, cost) |

## LLM Providers

The case study supports two LLM providers; deterministic mode (`--no-llm`) needs neither.

| Provider | Auth | Install | CLI flag |
|----------|------|---------|----------|
| **Claude** (default) | CLI auth via `claude login` | Included | `--provider claude` |
| **OpenAI / GPT** | `OPENAI_API_KEY` env var | `pip install -e ".[openai]"` | `--provider openai` |

Both providers expose the same `query_text` / `query_structured` interface, so the framework is provider-agnostic. The blend-and-verify architecture means an LLM failure or hallucination cannot break the experiment — the deterministic anchors and the verification firewall together guarantee that a feasible, certified schedule is always selected when one exists.

## Scripts

**Experiments**

| Script | Purpose |
|--------|---------|
| `scripts/run_experiment.py` | Single-cell experiment (CLI) |
| `scripts/run_factory_experiment.py` | Factory experiment (CLI) |
| `scripts/run_repeated.py` | Repeat a lifecycle N times with a fresh LLM client per run |
| `scripts/run_full_demo.sh` | All-in-one shell launcher with preflight checks |
| `scripts/launch_integrated.sh` | Bring up Isaac Sim / BaSyx / OPC-UA infrastructure |

**Analysis and baselines**

| Script | Purpose |
|--------|---------|
| `scripts/aggregate_repeated.py` | Mean / SD / CI, strategy stability, firewall pass rates and LLM call statistics over a batch |
| `scripts/paper_values.py` | Select the batch medoid run and print the paper's single-run table rows |
| `scripts/planner_baselines.py` | Layer-2 baseline: hybrid planner vs. anchors vs. constrained optimiser |
| `scripts/contract_fidelity_benchmark.py` | Layer-1 baselines on the 30-intent contract-fidelity benchmark |
| `scripts/sensitivity_analysis.py` | Firewall sensitivity to twin mismatch and sensor noise |
| `scripts/summarize_ablation_acceptance.py` | Candidate-pool ablation summary over the no-LLM / hybrid / LLM-only batches |
| `scripts/h3_control.py` | H3 control: next-shift schedule evaluated under the unrevised contract |
| `scripts/structural_probe.py` | Probe whether the LLM reassigns or reroutes unaided |
| `scripts/replay_deployment_boundary.py` | Replay archived L3 decisions through the corrected runner boundary |
| `scripts/validate_corrected_lifecycles.py` | Controlled no-LLM lifecycle validation, independent of the historical batches |

**Figures, export and utilities**

| Script | Purpose |
|--------|---------|
| `scripts/visualize_results.py` | Single-cell result figures |
| `scripts/visualize_factory_results.py` | Factory result figures |
| `scripts/generate_integrated_figure.py` | Four-panel integrated stack figure |
| `scripts/export_table3.py` | Run the single-cell no-LLM lifecycle and export its phase-level schedule-metrics bundle (legacy `table3` naming; ends at phase 5 like any no-LLM cell run) |
| `scripts/calibrate.py` | Calibration check: verify the physics models against their reference targets |
| `scripts/opcua_monitor.py` | Live OPC-UA node monitor for a running experiment |
| `scripts/isaac_scene_setup.py` | Headless Isaac Sim production-cell scene setup and verification server |

## Tests

```bash
pytest tests/ -v                                 # Full suite
pytest tests/test_experiment_e2e.py              # End-to-end single-cell
pytest tests/test_comprehensive_lifecycle.py     # Full lifecycle, both cases
pytest tests/test_constraint_checker.py          # Verification firewall
pytest tests/test_verification_completeness.py   # Every deployment passes L3
pytest tests/test_deployment_boundary.py         # Fail-closed deployment boundary
pytest tests/test_runtime_guards.py              # Guard synthesis and enforcement
pytest tests/test_simulation.py                  # SimPy execution
```

## Reproducing the Paper

The mapping from every paper table and figure to the artefact and command that produces it is in the
[root README](../README.md#reproducing-the-paper). The reported values come from the medoid runs of
the repeated batches under `data/repeated/`, not from a single execution.

To produce fresh single runs of each configuration:

```bash
# Single-cell case, three shifts, integrated stack
python scripts/run_experiment.py --integrated --shifts 3 \
    --output-dir data/results_full_integrated_3shifts

# Factory case, six-phase lifecycle
python scripts/run_factory_experiment.py --integrated \
    --output-dir data/results_factory_integrated

# No-LLM ablation references
python scripts/run_experiment.py --no-llm --integrated --shifts 3 \
    --output-dir data/results_nollm_integrated_3shifts
python scripts/run_factory_experiment.py --no-llm --integrated \
    --output-dir data/results_factory_nollm_integrated
```

The `data/results_*` directories hold exactly these single-run outputs. To reproduce the batch
statistics instead, use `scripts/run_repeated.py` with the commands listed under *Extended
Experiments* above, then `scripts/aggregate_repeated.py`.

LLM-driven phases do not reproduce exactly: candidates vary between calls, full LLM responses were
not retained for the historical batches, and historical factory sub-seeds were process-dependent.
Re-running produces a new batch under the same protocol rather than the archived numbers. The
deterministic paths (`--no-llm`, the anchors, the constraint checker, and Monte Carlo certification
at a fixed seed base) do reproduce exactly.
