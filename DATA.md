# Third-party data and code: what isn't stored here and why

The repository does not redistribute any large third-party file. Each one has
a canonical source and a checksum; reproduction is via two scripts from
`scripts/`. There are three reasons: license cleanliness (see below), size, and the fact that a copy
without a DOI is worse than a link with a DOI.

## 1. The FlyWire connectome and the Shiu et al. model

| | |
|---|---|
| Source | `https://github.com/philshiu/Drosophila_brain_model` |
| Fixed commit | `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960` (2024-09-14) |
| Size | ~370 MB including git history |
| Code license | MIT, © 2023 Philip Shiu and Nico Spiller |
| License of the data inside | **CC BY-NC 4.0** (FlyWire), separate from MIT |

The connectome data is **inside** this repository (`2023_03_23_connectivity_630_final.parquet`,
86 MB for v630 and 100 MB for v783), contrary to its Readme, which suggests
downloading an external archive. Nothing additional needs to be downloaded.

Clone: `python scripts/setup_model.py` (clones at the fixed commit).

**Why we don't vendor it.** If the fork's code and data were placed here, the repository
would redistribute data under CC BY-NC 4.0, and the NC restriction would spread
to the whole repository. Leaving the fork outside keeps our code's license
independent of NC. The same consideration is in the note "Organizational form and licenses
of the testbed", section 2.

The fork is used **without code changes**: `git status` in the clone is clean. This is verified and
is an argument in section 5.6 of the whitepaper.

## 2. Voltage-recording data, Huang et al. 2024

| | |
|---|---|
| Source | Zenodo, DOI `10.5281/zenodo.10998457` |
| License | **CC BY 4.0** — no NC restriction |
| Size | 920 MB, five archives |
| Used | `Figure3.zip` and `Figure4.zip` (reference), `Figure1.zip` and `Figure2.zip` (format check and calibration) |

Download and verify: `python scripts/fetch_huang_data.py`.

SHA-256 checksums — `results/huang_reference/checksums.txt`; the script checks
against them and refuses to accept a file with a mismatch.

**A note on downloading.** Zenodo's file endpoint regularly responds 504 while
metadata works fine. The script downloads to a temporary file, checks the
exact size from the API, and only then renames it; resuming via `curl -C -` is
unsuitable for this source — on a 504 an HTML error page is written into the file, and
resuming appends data on top of it.

## 3. FlyWire v630 cell-type annotations and hemibrain metadata

| | |
|---|---|
| Source | `https://github.com/flyconnectome/flywire_annotations`, tag `v1.1.0`, commit `df6bb136f5b3d91c3992df4e8de2642329e2a384` |
| Files | `Supplemental_file1_annotations.tsv` (21.7 MB), `Supplemental_file4_hemibrain_meta.csv` (2.4 MB) |
| FlyWire annotation license | **CC BY-NC 4.0** (the source repository has no `LICENSE` file; we attribute it to the FlyWire terms [27], like the rest of the connectome data) |
| Hemibrain metadata license | CC BY 4.0 (our reading of the Janelia terms; the license is not declared in the file itself) |

Download and verify: `python scripts/fetch_flywire_annotations.py`.
SHA-256 checksums — `results/mb_subcircuit/checksums.txt`.

**Why tag `v1.1.0`, not `main`.** Starting with release `v2.0.0`, the `root_id`
field in these files refers to FlyWire materialization **783**, but the model [2] is built
on **630**; identifiers from different materializations do not match, and mixing
them is not allowed. `v1.1.0` is the last release where `root_id` refers to 630. Caveat:
this is the annotation version from the Schlegel et al. preprint; the version the
*Nature* paper [18] references is tag `v2.1.0` on 783. Matching 783 to 630 could
be done via `supervoxel_id` and CAVE, but that requires an access token and adds another
unverifiable step, so it is not done.

**Why not Codex.** `codex.flywire.ai/api/download` returns a login page:
the annotation export there is behind authorization. Repository [18] is open and, unlike
Codex, contains a consistent and verified set — as the README of the
repository itself also indicates.

**Hemibrain metadata — why.** FlyWire does not label mushroom-body compartments. In the
hemibrain metadata [9], the compartment is in the `instance` field
(`MBON11(y1pedc>a/B)_R`), and the "type → compartment" map is built from it by the
`build_mb_subcircuit.py` script. The map itself (63 lines) goes into git as
`results/mb_subcircuit/compartment_map.tsv`: this is the type nomenclature from [9] under
CC BY 4.0, not a FlyWire data export.

**What from the subcircuit does not go into git.** The whole of `data/mb_subcircuit/`:
the ID list with types (`neurons.csv`) and the subcircuit in the model's format
(`completeness.csv`, `connectivity.parquet`) — this is material derived from FlyWire data under
CC BY-NC 4.0. Only summaries without identifiers go into git:
`results/mb_subcircuit/type_counts.tsv`, `subcircuit_stats.json`, `config.json`,
`compartment_map.tsv`, `checksums.txt`.

## 4. What from run outputs goes into the repository

Goes into git — small summaries serving as fixtures for the validation ladder:

- `results/v0_sugar/rate.csv`, `rate_std.csv`, `mn9.json` — the result of stage V0;
- `results/huang_reference/*.txt` — the computed H1b reference, archive design breakdown,
  checksums;
- `results/mb_subcircuit/*` — type counts, compartment map, subcircuit config
  and its hash, annotation checksums (see section 3).

Does not go in — raw spikes (`*.parquet`) and logs: they are reproduced from code and config,
are large, and, by our reading of CC BY-NC 4.0, inherit the NC restriction from
FlyWire data as derived material. The scope of the Adapted Material concept for model outputs
has not been confirmed by a lawyer; the project takes the conservative reading (Appendix C of the whitepaper).
