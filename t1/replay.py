"""Restore frozen procedural manifests and optionally regenerate rendered panels."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json

from . import oracle
from .itemgen import DATA, TRACK, write_json
from .records import atomic_text, load_rows, sha256

FROZEN = TRACK / "frozen"


def restore(frozen, dest, expected):
    with gzip.open(frozen, "rb") as stream:
        payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Frozen payload checksum mismatch: {frozen.name}")
    if dest.exists() and sha256(dest) != expected:
        raise ValueError(f"Refusing to replace different local input: {dest}")
    if not dest.exists():
        atomic_text(dest, payload.decode("utf-8"))


def render_split(split, dest, rows, limit):
    # The stored scene specifications preserve every frozen item, including
    # the distinct answer keys that formerly shared legacy identifiers.
    seen = 0
    with gzip.open(FROZEN / f"{split}.specs.jsonl.gz", "rt", encoding="utf-8") as stream:
        for row, spec_line in zip(rows, stream, strict=True):
            record = json.loads(spec_line)
            if row["id"] != record["id"]:
                raise ValueError("Frozen specification does not match instance order")
            if limit is not None and seen >= limit:
                continue
            spec = record["spec"]
            if row["family"] == "A":
                label = oracle.label_choice(spec, row["intended"])
            else:
                label = oracle.label_change(spec, set(row["asked"]) if row["asked"] else None, row["required"])
            fields = ("valid", "defect", "correct_set")
            if any(label[k] != row["label"][k] for k in fields):
                raise ValueError(f"Replayed label differs for {row['id']}")
            # Canonical ID is used for metadata. Historical image paths are
            # preserved, since legacy rows differed only in their answer key.
            write_json(dest / "items" / f"{row['id']}.json", spec)
            image_dir = (DATA / row["images"][0]["path"]).parent
            generated = oracle.save_panel_images(spec, image_dir)
            if [p["path"] for p in generated] != [str(DATA / p["path"]) for p in row["images"]]:
                raise ValueError(f"Replayed panel paths differ for {row['id']}")
            seen += 1
    return seen


def main():
    index = json.loads((FROZEN / "procedural_splits.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=index, action="append", help="Select splits. Defaults to all.")
    parser.add_argument("--render", action="store_true", help="Also verify labels and render the instance panels")
    parser.add_argument("--limit", type=int, help="Render only the first N items of each selected split")
    args = parser.parse_args()
    if args.limit is not None and (not args.render or args.limit <= 0):
        parser.error("--limit requires --render and a positive number")
    splits = args.split or list(index)
    for split in splits:
        expected = index[split]
        output = DATA / "procgen_v3" / split
        instances = output / "instances.jsonl"
        restore(FROZEN / f"{split}.instances.jsonl.gz", instances, expected["instances_sha256"])
        rows = load_rows(instances)
        if len(rows) != expected["rows"] or len({r["base_id"] for r in rows}) != expected["groups"]:
            raise ValueError(f"Frozen population differs for {split}")
        print(f"{split}: {len(rows):,} rows, {expected['groups']:,} groups")
        if args.render and "specs_sha256" in expected:
            with gzip.open(FROZEN / f"{split}.specs.jsonl.gz", "rb") as stream:
                if hashlib.sha256(stream.read()).hexdigest() != expected["specs_sha256"]:
                    raise ValueError(f"Frozen scene checksum differs for {split}")
            print(f"{split}: verified and rendered {render_split(split, output, rows, args.limit):,} rows")


if __name__ == "__main__":
    main()
