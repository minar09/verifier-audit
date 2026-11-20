# Auditing Whole-Item Validity in Generated Visual Tests

This repository accompanies *Auditing Whole-Item Validity in Generated Visual Tests: An Executable Benchmark and a Study of Verifier Transfer* by Juyeon Park, Matiur Rahman Minar, Yunhyeong Kwon, and Unsang Park.

A generated question can show plausible pictures and still have no correct answer, multiple correct answers, or a key that points to the wrong option. This release provides the procedural benchmark and the reference procedure that checks rendered evidence for derived-option and change-detection tasks. It also provides adapters for public benchmarks and the aggregate inputs needed to regenerate the manuscript tables.

## Start here

Use Python 3.11 and run the commands from the repository root.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m t1.replay --split test_id --split test_ood --render --limit 3
```

This verifies the frozen manifest checksums, checks six example verdicts against the reference procedure, and renders the panels under `data/procgen_v3/test_id/images` and `data/procgen_v3/test_ood/images`. Each frozen row includes a question, proposed key, panel paths, and the reference label. Use the question, proposed key and panels as verifier inputs. The remaining fields describe how the item was built and scored.

## Procedural benchmark

Run `python -m t1.replay` to restore all frozen manifests. Add `--render` to regenerate the panels for the four procedural splits. Rendering the full suite takes appreciably longer and writes roughly 1 GB of data. The separate calibration manifest references items from procedural development and uses those render paths.

| Split | Items | Source groups |
|---|---:|---:|
| Train | 18,720 | 1,796 |
| Development | 1,889 | 180 |
| Held-out assets and wording | 2,505 | 239 |
| Held-out templates | 1,795 | 200 |

The frozen manifests preserve the exact experiment inputs. The row loader assigns distinct canonical IDs to variants that once shared an identifier. The scene archives reconstruct 11 overwritten answer keys from the frozen rows. These variants share rendered panels with their counterpart. The reference compares renders at 768 px with the bundled calibrated tolerance. The exported panels shown to models are 448 px.

To generate fresh examples using the same procedure and corruption operators, use a distinct output name.

```sh
python -m t1.procgen --out demo --seed 7 --train 0 --dev 0 --test-id 3 --test-ood 2 --workers 1
```

The oracle is defined for supported procedural transformations and object changes. It measures agreement with the reference under the stated renderer and tolerance.

## Public benchmark adaptations

Install the optional data-conversion dependencies. The downloader retrieves pinned files from the original Hugging Face repositories and verifies size and SHA-256 using `frozen/external_sources.json`. Review upstream terms before using the data. The three source downloads total about 832 MB.

```sh
python -m pip install -r requirements-external.txt
python -m t1.fetch muirbench dare mm_upd
python -m t1.external build-muir
python -m t1.benchmarks all
```

Output under `data/external_eval/` contains the adapted validity and native-answering records. Frozen first-pass membership for DARE and MM-UPD is in their respective `protocol_v1.json`. The Muir split manifest records membership in the exploratory first pass and the 901-pair confirmation. Curate and score derivatives under the respective original releases. [MuirBench](https://huggingface.co/datasets/MUIRBENCH/MUIRBENCH) · [DARE](https://huggingface.co/datasets/cambridgeltl/DARE) · [MM-UPD](https://huggingface.co/datasets/MM-UPD/MM-UPD)

## Regenerate manuscript tables

```sh
python -m t1.replay
python -m t1.tables
```

The builders write LaTeX table bodies and provenance receipts under `paper_ieeeaccess/tables/`. They use frozen result summaries, including the locked Muir-V2 confirmation and the full and held-out DARE and MM-UPD analyses. The pilot and reviewer inputs are reduced to the tallies used by the tables. To score new verdicts, supply a directory of tag-named JSON predictions mapping each canonical item ID to a verdict object such as `{"valid": true}`.

```sh
python -m t1.score path/to/predictions data/procgen_v3/test_id/instances.jsonl --methods experiment_name
```

The table workflow reproduces reported tabulation from stored aggregate inputs. End-to-end GPU retraining and re-inference require separate models, checkpoints and predictions. The optional `build_full_external_tables.py --from-predictions` recomputes the external summary from locally generated inputs and predictions in the layout specified by that script.

## Provenance and attribution

- `t1/` contains the study's generator, verifier reference, corruption operators, adapters and scoring tools.
- `frozen/` contains the compressed procedural manifests and scene specifications, exact source pins, and file checksums. It also contains the public benchmark split manifest and aggregate result snapshots.
- `itemgen_reference/` contains the necessary renderer modules and slot specification from the MIT-licensed ItemGen transfer package, with its original license notice. The 52 object PNGs are the exact files used from that package, generated with the Apache-2.0 FLUX.2-klein-4B checkpoint. Their checksums are in `itemgen_reference/assets/SHA256SUMS`. The related public ItemGen code has its own release history. [ItemGen](https://github.com/Lucypothesis/dementia_questions)

See `CITATION.cff` for the authors and paper title. The vendored ItemGen license is recorded separately from the companion study's code.
