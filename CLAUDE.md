# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

IntelliFold is an AlphaFold3-style foundation model for biomolecular structure prediction
(proteins, DNA, RNA, ligands, and their complexes), published by IntelliGen AI. It is an
**inference-only** open-source release: a PyTorch model plus a data pipeline, packaged as the
`intellifold` PyPI package with a single `intellifold predict` CLI command. There is no training
code, and no test suite in this repo.

## Common Commands

```bash
# Environment (conda brings in the bioinformatics binaries: mmseqs2, hmmer, kalign2)
conda env create -f environment.yaml && conda activate intellifold
pip install -e .                      # editable install for development

# Run inference (downloads weights + CCD to the cache dir on first run)
intellifold predict ./examples/5S8I_A.yaml --out_dir ./output --cache ./cache_data --model v2-flash

# Equivalent without installing (runs the standalone copy of the runner):
python run_intellifold.py ./examples/5S8I_A.yaml --out_dir ./output --cache ./cache_data

# Demo wrapper script (single-GPU + commented multi-GPU/accelerate variants):
bash predict.sh

# Full option list:
intellifold predict --help

# Multi-GPU via HuggingFace Accelerate:
accelerate launch --multi_gpu --num_processes 2 run_intellifold.py ./examples --out_dir ./output --cache ./cache_data
# or with a config file:
accelerate launch --config_file ./accelerator_single_machine.json run_intellifold.py ./examples ...

# Auto-generate MSAs (no precomputed .a3m needed) via the ColabFold mmseqs2 server:
intellifold predict input.yaml --out_dir ./output --use_msa_server --msa_pairing_strategy greedy

# Only run the data-processing stage (skip the model forward pass):
intellifold predict input.yaml --out_dir ./output --only_run_data_process

# Template search helper (HMMER-based; needs a pdb_seqres FASTA database):
python runner/run_templates_search.py --input_msa ./examples/msas/5s8i_A.a3m \
  --output_template ./output/5s8i_A_hmmsearch.a3m --seqres_database_path /path/to/pdb_seqres.fasta

# Pre-download all weights/CCD/databases manually (otherwise done lazily at runtime):
HF_ENDPOINT=huggingface.co cache_dir=./cache_data bash download_cache_data.sh
cache_dir=./cache_data bash download_template_data.sh
```

## Key Conventions

- **Model versions** (`--model`): `v1` (original paper, `intellifold_v0.1.0.pt`), `v2`
  (`intellifold_v2.pt`), `v2-flash` (`intellifold_v2_flash.pt`, the **default** — faster + more
  accurate). Each version has its own config builder in `intellifold/openfold/` — see the version
  dispatch in the runner's `main()`.
- **Cache directory** holds downloaded weights + `ccd_v2.pkl` + sequence databases. Resolved from
  `--cache`, defaulting to `$INTELLIFOLD_CACHE` or `~/.intellifold`. Downloads fall back from
  `huggingface.co` to `hf-mirror.com` automatically; override via `HF_ENDPOINT`.
- **Input format**: a YAML file (or a directory of them) listing `sequences` of typed entities
  (`protein`/`dna`/`rna` with `sequence`, `ligand` with `smiles` or `ccd`). See
  `docs/input_yaml_format.md`. `msa` is required for proteins unless `--use_msa_server` is passed;
  `msa: empty` forces single-sequence mode.
- **Output layout**: `<out_dir>/<input_stem>/predictions/<record_id>/` with one `.cif`/`.pdb` plus
  `_summary_confidences.json` and `_confidences.json` per (seed, diffusion sample). Failed targets
  are caught per-record and written to `<out_dir>/<input_stem>/errors/<id>.txt` rather than
  aborting the run. `processed/` holds intermediate tokenized/featurized data.
- **Optional CUDA kernels** (off by default, compiled on first use) — see `docs/kernels.md`:
  `export LAYERNORM_TYPE=fast_layernorm` and
  `export USE_DEEPSPEED_EVO_ATTENTION=true` (the latter needs `CUTLASS_PATH` pointing at a cloned
  CUTLASS v3.5.1).

## Architecture

Two layers: a **data pipeline** (`intellifold/data/`) and the **model** (`intellifold/openfold/`).

**Entry points are duplicated.** `runner/intellifold_inference.py` is the installed console script
(`intellifold` → `intellifold_cli`, a `click` group with the `predict` command). `run_intellifold.py`
at the repo root is a near-identical standalone copy used for `python run_intellifold.py` and
`accelerate launch`. **Changes to inference orchestration usually need to be made in both files.**
The click `predict` command packs all options into an `argparse.Namespace` and calls `main(args)`.

`main()` flow (all heavy steps gated on `accelerator.is_main_process`, with
`wait_for_everyone()` barriers between stages):
1. `download(...)` — fetch model + CCD into the cache.
2. `check_inputs(...)` — validate the YAML/dir of inputs.
3. `process_inputs(...)` in `intellifold/data/inference/data_tools.py` — the data pipeline:
   parse (`data/parse/`, schema in `parse/schema.py`) → MSA (`compute_msa` via the mmseqs2 server
   in `data/msa/`, or precomputed) → tokenize (`data/tokenize/`) → featurize (`data/feature/`) →
   optional templates (`data/template/`). Writes a `manifest.json` + `processed/` tree.
4. Build a dataloader (`data/module/inference.py:get_inference_dataloader`) over the manifest.
5. Select config + checkpoint by `--model`, instantiate `IntelliFold`, `load_state_dict`,
   `accelerator.prepare`.
6. Loop over batches; for each, loop over seeds → `predict_and_save`, which runs the forward pass,
   aggregates per-atom outputs (`aggregate_fn` in `openfold/utils/atom_token_conversion.py`),
   computes confidences (`openfold/model/confidences.py`), and writes CIF/PDB + JSON.

**Model** (`intellifold/openfold/model/model.py`, class `IntelliFold`) has three stages, mirroring
AlphaFold3: `BackboneTrunk` (`backbone.py`, the Pairformer + recycling trunk producing single/pair
reprs) → `DiffusionModule` (`diffusion.py`, EDM-style diffusion sampler that generates coordinates,
driven by `sample_diffusion`/`noise_schedule`) → `ConfidenceHead` (`heads.py`, pLDDT/PAE/pTM).
`CentreRandomAugmentation` provides the per-sample random rotation/translation. Supporting layers
(triangular attention/multiplication, outer product mean, primitives, embedders) live alongside.

**Provenance** (per the README): `intellifold/openfold/` is adapted from
[OpenFold](https://github.com/aqlaboratory/openfold); the inference data pipeline and MSA generation
are derived from [Boltz-1](https://github.com/jwohlwend/boltz) (hence `Boltz`-prefixed names like
`BoltzProcessedInput`, `boltz.py` in `crop/` and `tokenize/`); the template pipeline follows
[Protenix](https://github.com/bytedance/Protenix); fast layernorm kernels follow FastFold/OneFlow.
When editing these subtrees, match the upstream conventions already present.
