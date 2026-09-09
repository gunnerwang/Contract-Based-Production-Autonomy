# Deployment-boundary correction validation

The original repeated batches are unchanged. `replay_summary.json` applies the corrected runner boundary to 282 archived non-replan L3 decisions independently: 248 allowed, 20 nominal failures refused, and 14 further sampled failures refused. Later phases are included even if a corrected earlier phase would halt; these are not reconstructed lifecycle trajectories or new throughput results. Sample counts absent from an event use the documented historical 200-sample protocol.

Run from the repository root:

```bash
python scripts/replay_deployment_boundary.py
python -m pytest tests -q --disable-warnings
```

`test_results.txt` records 318 passing tests. Tests include whole factory lifecycles, rejected final certificates before dispatch/prestaging, cell refusal, candidate fallback to a certified alternative, probability/count checks, error-state and partial-export behavior, and cross-process sampling reproducibility. They do not validate a physical emergency stop or independent model calibration.

`historical_source/` preserves the pre-correction versions of every changed implementation module (not tests). `source_versions.json` records before/after SHA-256 values. For historical-code reconstruction in a separate copy, restore these files to their corresponding paths under ``; all other implementation sources are unchanged by this correction. The historical factory RNG uses Python's randomized string hash; the seed recorded in a run is insufficient to reconstruct its cell draws unless its process hash seed is also known. Restoring code alone cannot recover that unrecorded state.

The corrected factory uses stable sorted cell indices in sub-seeds and phase-consistent certification/reporting seeds. Original performance tables remain historical results. The V/R `+1` baseline was already in the analytical scorer; the manuscript equation was corrected, without changing that scoring function.

The follow-up completeness correction also rejects uncheckable constraints, missing/non-finite required readings, partial factory inputs, and report/contract mismatches. The archived-decision replay cannot establish completeness of the historical checks because it consumes their stored verdicts. `../verification_completeness/` contains four independent, controlled no-LLM lifecycle executions and their source hashes; they are distinct from historical LLM results.

The runtime guard follow-up synthesizes all supported encoded obligations, uses the verifier's comparison and metric semantics, and aborts on missing/non-finite readings or hard violations without rewriting measurements. The simulated cyber-risk level is explicitly supplied; absent input fails. Factory phase 2 checks an analytical snapshot, SimPy checks end-of-shift metrics, and integration checks every collected poll using raw sensor readings (missing values, model substitutes, and unavailable throughput/energy/defect/deadline observations cannot pass). Runtime failures propagate without fallback and are audited as `runtime_execution_blocked`; software abort is not physical stopping. `tests/test_runtime_guards.py` contains 63 focused regressions; `tests/test_runtime_guard_integration.py` adds 15 mocked sensor/integration regressions. The controlled lifecycle outputs and source hashes were refreshed after this correction.

The comprehensive follow-up adds 38 cross-path regressions for service failure states (including escalation completion), canonical audit paths, accepted-learning timing, repeated contract validation and preservation, mandatory upper-bound directions, zero/non-finite assumption observations, absence-planning semantics, grounded nullable attribution, and refusal after integrated execution errors. These corrections are included in the refreshed controlled-run source hashes.
