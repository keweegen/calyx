# Run records

Every stage of the validation ladder left three kinds of file here: the pre-registration that was
frozen before the run, the artefacts the run produced, and the report that interprets them. This
index says which is which. **Nothing in this directory is edited after its stage closes.**

The records are in Russian, the language they were written in. The project moved to English on
10 September 2026, and the interpretation of every outcome is carried in English by the whitepaper
(sections 5.7–5.11) and the experiment specification. These files were left as they are because
their point is to be unchanged evidence: a pre-registration that gets rewritten — even into another
language — is no longer a pre-registration. The numbers in them are language-independent.

## What is frozen, and what the hash covers

A stage's pre-registration is the **byte copy of the specification stored next to its hash**, not
the live document in `whitepaper/`. The live document is deliberately extended after a run, so its
hash does not match and is not meant to.

| Stage | Pre-registration (byte copy) | Hash file | Outcome | Report |
|---|---|---|---|---|
| V1a | `v1a/experiment-spec-h1-h3.v0.9.frozen.md` | `v1a/spec_sha256.txt` | V1a-S passed, V1a-D not passed | `v1a/` artefacts |
| V1b | `v1b/experiment-spec-h1-h3.v0.14.frozen.md` | `v1b/spec_sha256.txt` | NOT-TESTABLE, closed on an execution error | `v1b/report_v1b_closure.md` |
| V1b′ | `v1b_prime/experiment-spec-h1-h3.v0.15.frozen.md` | `v1b_prime/spec_sha256.txt` | FAIL-CAL-MBON, as pre-registered | `v1b_prime/report_v1b_prime.md` |
| V1c | `v1c/experiment-spec-h1-h3.v0.17.frozen.md` | `v1c/spec_sha256.txt` | FAIL-CAL-MBON-FLOOR; the declared expectation FAIL-CAL-MBON-CEIL did not hold | `v1c/report_v1c.md` |

Stage V1c also carries `v1c/experiment-spec-h1-h3.v0.16.stamp0.md` with `v1c/stamp0_sha256.txt`:
the construction of the stage written as formulas and hashed **before** the measurements that could
supply its numbers. The difference between that stamp and the pre-registration is listed, item by
item, in the specification's "Provenance" section.

`.gitattributes` stores every frozen copy and hash file with `-text`, so a checkout cannot change a
line ending and invalidate a digest.

## Reports are generated, not written

`report_v1c.md`, `report_v1b_prime.md` and `report_v1b_closure.md` are produced by `v1c_report.py`,
`v1b_prime_report.py` and `v1b_diagnostics.py`. Those scripts keep their output strings in Russian
on purpose: the committed report has to be reproducible byte-for-byte from the artefacts, and
re-wording the generator would break that. Their comments and docstrings are in English.

One consequence worth knowing before editing `v1c_report.py`: it slices its own module docstring at
a Russian marker and writes the result into the report's section on what was declared before the
run. The docstring is data, not documentation.
