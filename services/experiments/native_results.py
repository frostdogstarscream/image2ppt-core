"""Tabulate fresh native-edit benchmarks from their download and desktop Office receipts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

UNKNOWN = 'UNKNOWN'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def results(folder):
    comparison = read(folder/'comparison.json')
    delivery = folder/'delivery'
    summary = read(delivery/'summary.json') if (delivery/'summary.json').is_file() else {}
    versions = dict(comparison['versions'])
    independently_probed = folder/'agent-before-fallback'
    trial = independently_probed/'result.pptx'
    probe = independently_probed/'office/report.json'
    if 'agent_candidate' not in versions and trial.is_file() and probe.is_file():
        scorefile = delivery/'agent-scorecard.json'
        with trial.open('rb') as f:
            sha = hashlib.file_digest(f, 'sha256').hexdigest()
        if not scorefile.is_file() or read(scorefile)['pptx_sha256'] != sha:
            raise ValueError('Pre-fallback candidate differs from the scored Agent deck')
        versions['agent_candidate'] = {'sha256':sha,'actions':read(probe)}
    for version, record in versions.items():
        sha = record.get('sha256')
        score = {}
        for path in (delivery/'baseline-scorecard.json', delivery/'agent-scorecard.json'):
            if path.is_file():
                candidate = read(path)
                if sha and candidate['pptx_sha256'] == sha:
                    score = candidate
                    break
        agent = {'participated':False} if version=='host' else summary.get('agent', {})
        usage = agent.get('usage', {})
        off = record.get('actions')
        for idx, source in enumerate(comparison['sources'], 1):
            p = next((r for r in score.get('pages', []) if r['page']==idx), {})
            sc = p.get('scorecard', {})
            edits = [r for r in off if r['page']==idx] if isinstance(off, list) else []
            passed = sum(r.get('status')=='EDIT_OBSERVED' and r.get('saved_value') is True
                         and r.get('selection')!='multiple selection' for r in edits)
            fields = [r for r in edits if r['action']=='text']
            hits = sum(bool(r.get('object')) for r in fields)
            catalog = comparison.get('sample_catalog', [])
            sample = catalog[idx-1] if len(catalog) >= idx else {}
            yield dict(split=sample.get('split') or UNKNOWN, family=sample.get('family') or UNKNOWN, job=comparison['job_id'], profile=folder.parent.name, version=version, page=idx,
                       upload_sha256=source, pptx_sha256=sha or UNKNOWN, state=comparison['state'],
                       office=score.get('office_status', UNKNOWN), observed_edits=passed if edits else UNKNOWN,
                       tested_edits=len(edits) if edits else UNKNOWN, native_fields=p.get('native_field_count', UNKNOWN),
                       ocr_native_hits=len(p['ocr_native_hits']) if 'ocr_native_hits' in p else UNKNOWN,
                       frozen_text_cases=len(fields) if fields else UNKNOWN, frozen_text_hits=hits if fields else UNKNOWN,
                       independent_field_recall=UNKNOWN, editable_objects=p.get('editable_count', UNKNOWN),
                       plate_content=sc.get('raster_plate_content_fraction', UNKNOWN),
                       ssim=sc.get('fidelity', {}).get('ssim', UNKNOWN),
                       delta_e_p95=sc.get('fidelity', {}).get('delta_e2000_p95', UNKNOWN),
                       agent_participated=agent.get('participated', UNKNOWN),
                       model_messages=usage.get('reported_model_messages', 0 if version=='host' else UNKNOWN),
                       tokens_in=usage.get('tokens', {}).get('input', 0 if version=='host' else UNKNOWN),
                       tokens_out=usage.get('tokens', {}).get('output', 0 if version=='host' else UNKNOWN),
                       independent_review=summary.get('independent_go_review', UNKNOWN) if version=='selected' else UNKNOWN,
                       human_wps=UNKNOWN)


def write_table(files, target):
    rows = [row for path in sorted(files) for row in results(path.parent)]
    if not rows:
        raise ValueError('No completed comparison reports were found')
    with target.open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument('csv',type=Path)
    args=parser.parse_args()
    print(write_table(args.results.rglob('comparison.json'),args.csv))
