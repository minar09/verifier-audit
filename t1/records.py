"""Canonical identities and atomic artifacts; frozen source files remain unchanged."""
from __future__ import annotations

import collections
import hashlib
import json
import os
import math
from pathlib import Path


def canonical_rows(rows):
    counts = collections.Counter(r['id'] for r in rows)
    out = []
    for r in rows:
        if counts[r['id']] > 1:
            # Include the input key, never the gold label. Keep the asset filename.
            payload = {k: r[k] for k in ('question', 'key', 'images')}
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
            r = dict(r, legacy_id=r['id'], id=r['id'] + '__input_' + digest)
        out.append(r)
    ids = [r['id'] for r in out]
    if len(ids) != len(set(ids)):
        raise ValueError('Identical duplicate inputs or canonical ID collision')
    return out


def load_rows(path):
    return canonical_rows([json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()])


def observation_for(mapping, row):
    return mapping[row['id']] if row['id'] in mapping else mapping[row.get('legacy_id',row['id'])]


def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def atomic_json(path, value):
    def finite(x):
        if isinstance(x,float) and not math.isfinite(x):return None
        if isinstance(x,dict):return {k:finite(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [finite(v) for v in x]
        return x
    atomic_text(path, json.dumps(finite(value), ensure_ascii=False, indent=1,allow_nan=False) + '\n')


def check_coverage(rows, preds):
    expected = {r['id'] for r in rows}
    if len(rows) != len(expected):
        raise ValueError('Duplicate IDs: canonicalize before evaluation')
    if set(preds) != expected:
        raise ValueError(f'Prediction coverage mismatch: missing={len(expected-set(preds))}, extra={len(set(preds)-expected)}')


def manifest_path(path):
    path = Path(path)
    return path.with_name('_manifest_' + path.stem + '.json')


def write_manifest(path, rows, provenance=None):
    check_coverage(rows, json.loads(Path(path).read_text()))
    value = {'version': 1, 'n': len(rows), 'rows_sha256': fingerprint(rows),
             'predictions_sha256': sha256(path), 'provenance': provenance or {}}
    raw = Path(path).with_suffix('.raw.jsonl')
    if raw.exists():
        value['raw_sha256'] = sha256(raw)
    atomic_json(manifest_path(path), value)


def complete(path, rows):
    path = Path(path)
    try:
        check_coverage(rows, json.loads(path.read_text()))
        m = json.loads(manifest_path(path).read_text())
        ok = m['rows_sha256'] == fingerprint(rows) and m['predictions_sha256'] == sha256(path)
        if 'raw_sha256' in m:
            ok = ok and m['raw_sha256'] == sha256(path.with_suffix('.raw.jsonl'))
        return ok
    except (OSError, ValueError, KeyError, TypeError):
        return False


def checkpoint_complete(path, supervision=None, seed=None, steps=1100):
    path = Path(path)
    try:
        c = json.loads((path/'config.json').read_text())
        log = [json.loads(l) for l in (path/'train_log.jsonl').read_text().splitlines()]
        return (c['steps'] == steps and log[-1]['step'] == steps
                and c['data'] == 'procgen_v3'
                and (supervision is None or c['supervision'] == supervision)
                and (seed is None or c['seed'] == seed)
                and all((path/k/'adapter_model.safetensors').is_file() for k in ('best', 'last')))
    except (OSError, ValueError, KeyError, IndexError):
        return False
