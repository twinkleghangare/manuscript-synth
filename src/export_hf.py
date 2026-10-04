"""Push the generated dataset to Hugging Face (3 configs: devanagari / modi / sharada)."""
import json
from pathlib import Path


def push(output_dir, repo_id, private=False):
    from datasets import Dataset, DatasetDict, Image
    out = Path(output_dir)
    for script_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        splits = {}
        for split in ("train", "validation", "test"):
            meta = script_dir / split / "metadata.jsonl"
            if not meta.exists():
                continue
            rows = [json.loads(l) for l in meta.read_text(encoding="utf-8").splitlines()]
            for r in rows:
                r["image"] = str(script_dir / split / r["file_name"])
            splits[split] = Dataset.from_list(rows).cast_column("image", Image())
        DatasetDict(splits).push_to_hub(repo_id, config_name=script_dir.name, private=private)
