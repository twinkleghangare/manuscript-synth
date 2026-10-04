# Synthetic Indic Manuscript Generator

Pipeline that synthesizes historical manuscript folios (Devanagari, Modi, Sharada) with paired `.md` ground truth.

**Hugging Face dataset:** [twinklemghangareaids27/indic-manuscripts-synthetic](https://huggingface.co/datasets/twinklemghangareaids27/indic-manuscripts-synthetic) (configs: `devanagari`, `modi`, `sharada`)

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Place fonts in `fonts/` (e.g. Noto Sans Devanagari / Modi / Sharada) and text in `corpora/<script>.txt` (one passage per line).

## Usage
```bash
python generate.py                      # all scripts, 100 images each, 85/10/5 split
python generate.py --scripts modi --n 20
python generate.py --push               # upload to Hugging Face (huggingface-cli login first)
```
All parameters live in `config.yaml`.

## Structure
```
generate.py   entry point        src/backgrounds.py  paper / palm leaf
config.yaml   parameters         src/deform.py       warp, curl, folds
corpora/      text per script    src/layout.py       blocks and margins
fonts/        fonts              src/render.py       handwriting-style text
output/       generated data     src/ink.py          ink colour, bleed, fade
                                 src/annotate.py     .md ground truth
                                 src/export_hf.py    HF upload
```

## Dataset organization
`output/<script>/{train,validation,test}/Image_N.png|.md` plus `metadata.jsonl` per split.
