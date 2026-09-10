# Calyx — a testbed for validating learning rules on the Drosophila connectome

Testbed code where the learning rule is a swappable module on top of a real
connectome, and failure criteria are fixed before the run.

Project documents — the whitepaper, the H1–H3 experiment specification, and a note on the measurement
the result rests on — are published separately: <https://keweegen.github.io/calyx>.

Code and document text are Apache-2.0, © 2026 16th Research Lab MMC: use, modify,
distribute. Run artifacts are derived from FlyWire data (CC BY-NC 4.0) and
inherit the non-commercial-use restriction — that is the licensor's condition, not the project's.
The reference computed from Huang et al. data is CC BY 4.0. Details in `NOTICE`.

**Status: the validation ladder is stopped, the testbed with plasticity is not built.** Not a single
run with plasticity has been performed and none will be: the stopping rule fired
(whitepaper, section 6.5б). The repository is open until the first substantive result,
as section 6.4 of the whitepaper promised, so the negative result is published exactly
the same way a positive one would have been. V0 and V1a-S passed; V1a-D did not pass and is stopped;
V1b is closed due to an execution error, and V1b′, which replaced it, gave the pre-registered
negative result. V1c — the third knob — was run on September 9, 2026 and also
closed negative, but with an outcome different from the announced expectation.

## Validation ladder

| Stage | What it checks | Status |
|---|---|---|
| V0 | Simulator and config: determinism, no activity without input | **passed** September 7, 2026 |
| V1a-S | Substrate fidelity: subcircuit vs. the full Shiu et al. model | **passed** September 8, 2026 |
| V1a-D | Level-1 decoder validation | **not passed, branch stopped** September 8, 2026 |
| V1b | Subcircuit after step-1 substitutions (DAN synapse mask, graded APL, calibration) | **closed, NOT-TESTABLE** September 8, 2026: the code implemented the admissibility constraint not to the letter of the specification |
| V1b′ | Same, with admissibility to the letter | **closed, FAIL-CAL-MBON** September 9, 2026 |
| V1c | Third knob: KC→MBON weight scale | **closed, FAIL-CAL-MBON-FLOOR** September 9, 2026, specification v0.17, hash `636c49968a0866bd`. FAIL-CAL-MBON-CEIL was expected — the expectation was not confirmed |
| V2 | The rule as a specification on a reduced circuit in a rate-equivalent form | — |
| V3 | Translation of the rule into Brian 2 / NESTML | — |
| H1 | Step 1: olfactory conditioning on the connectome-complete subcircuit | — |

The ladder is ordered by dependencies, not linearly (specification, section 3, revision
v0.10): moving to V1b required passing V1a-S, while V1a-D remains a separate
registered result and does not block V1b, since V1b's criteria do not use the decoder.
V1a-D is not retroactively declared passed. Wording and
criteria are in the experiment specification, section 3.

### V1b′ result

The stage was pre-registered by specification v0.15 with the expected outcome
declared before the run. Of 447 grid points, 57 reached the Kenyon-cell sparse
regime, but **none passed the output-neuron response constraint**:
the 67 Hz ceiling was not violated anywhere, and the 2 Hz floor was not reached for any
of the six reference types at any point. The best rate for a single type across the whole grid
was 0.33 Hz.

Substantively: in the family with KC→MBON weights fixed per Shiu et al. and
zero MBON background, Kenyon-cell sparseness and the output-neuron operating range
are simultaneously unreachable. The claim applies to this family and to
this grid, not to the connectome. Reports — [`results/v1b_prime/report_v1b_prime.md`](results/v1b_prime/report_v1b_prime.md)
and [`results/v1b/report_v1b_closure.md`](results/v1b/report_v1b_closure.md).

### V1c result

The third and last allowed knob is a single global multiplier `s` on all
KC→MBON weights. The stage's design was recorded as formulas and hashed as stamp 0
(`results/v1c/stamp0_sha256.txt`) before any measurements that could have produced its numbers;
the pre-registration is version v0.17, hash `636c49968a0866bd`. The expected outcome was declared before
the run: FAIL-CAL-MBON-CEIL.

**Run on September 9, 2026: outcome FAIL-CAL-MBON-FLOOR, the expectation was not confirmed.**
285 points, 270 of them admissible by sparseness, the band is empty. Mechanism analysis and
inference bounds — whitepaper, sections 5.10 and 5.11; the full run report —
[`results/v1c/report_v1c.md`](results/v1c/report_v1c.md).

### V0 result

Sugar sensory neurons → motor neuron MN9, FlyWire v630, 6 frequencies × 30 repeats ×
1000 ms. Threshold between 20 and 60 Hz, monotonic growth with saturation — reproduces
Fig. 1D of the Shiu et al. paper.

| Input, Hz | 20 | 60 | 100 | 140 | 180 | 200 |
|---|---|---|---|---|---|---|
| MN9, Hz | 0.00 | 37.60 | 65.80 | 80.30 | 87.67 | 93.37 |

Summaries — `results/v0_sugar/rate.csv` and `mn9.json`.

## Installation

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Linux/macOS: .venv/bin/pip
python scripts/setup_model.py                   # the Shiu model at a fixed commit
python scripts/fetch_huang_data.py              # Huang data from Zenodo, with checksum verification
```

Large files are not stored in the repository — why, and where they come from, see
[DATA.md](DATA.md).

## Code-generation backend

By default `brian_preferences` sets `codegen.target = 'numpy'`. An explicit target
is needed because with auto-selection, Brian2 tries to compile
Cython in each worker, fails, and writes a warning through the logger; on 12 parallel
processes the loky pool stalls in `logging.emit`.

Failure doesn't necessarily mean there's no compiler: on Windows `setuptools`
often fails to find an installed Visual Studio and fails with
`Unable to find a compatible Visual Studio installation`. The
`msvc_run.bat` wrapper finds the installation via `vswhere`, calls `vcvars64.bat`, and sets
`DISTUTILS_USE_SDK=1`, after which compilation succeeds; the insert cache is placed in
`.cython_cache` next to the code.

Measurement of both backends — `bench_backend.py`, results in
`results/bench_backend/`. The backend is part of the run config, and the config is hashed,
so changing the backend requires re-running the ladder's stages with fixture refreezing.

## What's here

| File | Purpose |
|---|---|
| `v0_smoke.py` | Smoke test: the model comes up and computes |
| `v0_sugar.py` | Stage V0. The number of processes is computed from available memory: the full brain takes ~2 GB per worker, and `n_proc=-1` on 12 cores gives a `MemoryError` |
| `bench_mb_size.py` | Cost measurement on a mushroom-body-sized subcircuit |
| `diag_mem.py` | Memory diagnostics |
| `huang_design.py` | Parsing the design of the Huang archives: cell types, flies, sessions, CS+/CS− labeling |
| `huang_reference.py` | Computing the H1b reference from open data by the procedure of specification section 6а |
| `brian_preferences` | Brian2 backend = numpy |
| `results/huang_reference/` | The computed reference, design breakdown, archive checksums |

### H1b reference

`huang_reference.py` computes, from Huang et al. 2024 data, the value against which
the model effect is compared. For MBON-γ1pedc>α/β, the response shift to the reinforced
odor relative to control, 5 minutes after training:

**−34.27 Hz, 95% CI [−44.08, −24.46]**, n = 12 flies.

The procedure is fixed in the specification **before** the run, so it cannot be
tuned to the result. Extraction check: the same procedure picks out a significant
effect for exactly MBON-γ1pedc>α/β, MBON-γ2α′1, and MBON-α3 — the list matches
the one published in the source.

The reference constrains the subcircuit's output, not the weights directly: it is a neuron's spike
rate, not synaptic strength. The flies in the recordings are not the individual the connectome
was captured from.

## Licenses

Testbed code is **Apache License 2.0** ([LICENSE](LICENSE)). Attribution and third-party
components — [NOTICE](NOTICE).

Third-party data and code are not redistributed, so their licenses do not affect
this repository's license. A significant consequence: the Shiu et al. model under MIT
contains inside it FlyWire data under **CC BY-NC 4.0**, and if it were placed here,
the non-commercial-use restriction would spread to the whole testbed. The
Huang et al. data is under **CC BY 4.0**, with no such restriction. Details — [DATA.md](DATA.md).

The rights holder remains an open question: the author is listed as a natural person, but this may
be 16th Research Lab (see NOTICE and the note "Organizational form and licenses").

## References

- Dorkenwald et al. Neuronal wiring diagram of an adult brain. *Nature* 634, 124–138 (2024). <https://doi.org/10.1038/s41586-024-07558-y>
- Shiu et al. A Drosophila computational brain model reveals sensorimotor processing. *Nature* 634, 210–219 (2024). <https://doi.org/10.1038/s41586-024-07763-9>
- Huang et al. Dopamine-mediated interactions between short- and long-term memory dynamics. *Nature* 634, 1141–1149 (2024). <https://doi.org/10.1038/s41586-024-07819-w>
- Data for Huang et al.: Zenodo, <https://doi.org/10.5281/zenodo.10998457> (CC BY 4.0)
