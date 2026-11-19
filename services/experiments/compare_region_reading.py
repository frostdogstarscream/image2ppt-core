"""Compare three fresh-upload deliveries against the frozen (evaluation-only) edit cases."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from native_results import read
from holdout.check_inputs import check

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'product'))
from product.materialize import digest, write
from product.scorecard import inspect_pptx, _native_fields, _overlap


def native_reading(fields, target):
    """Match by position before inspecting text, so a misspelling remains observable."""
    candidates = []
    for field in fields:
        box = field['bbox']
        smaller = min(box[2] * box[3], target[2] * target[3])
        overlap = _overlap(box, target)
        if smaller > 0 and overlap >= .55 * smaller:
            candidates.append((overlap / (box[2]*box[3] + target[2]*target[3] - overlap), field))
    candidates.sort(key=lambda entry: entry[0], reverse=True)
    if not candidates:
        return {'status':'MISSING'}
    if len(candidates) > 1 and candidates[1][0] >= .9 * candidates[0][0]:
        return {'status':'AMBIGUOUS'}
    return {'status':'FOUND', 'name':candidates[0][1]['name'], 'text':candidates[0][1]['text']}


def record_arm(root, pages, cases, expected_profile):
    results = {}
    receipts = sorted(root.rglob('comparison.json'))
    for path in receipts:
        comparison = read(path)
        deck = path.parent / 'delivery/result.pptx'
        sha = digest(deck)
        selected = comparison['versions']['selected']
        summary = read(path.parent/'delivery/summary.json')
        actual_profile = summary.get('text_profile', {'region_text':False, 'page_reading':False})
        if actual_profile != expected_profile:
            raise ValueError(f'Unexpected runtime text profile in {path}: {actual_profile}')
        if sha != selected.get('sha256'):
            raise ValueError(f'Final deck differs from its fresh-upload receipt: {path}')
        actions = selected.get('actions')
        if not isinstance(actions, list):
            raise ValueError(f'Missing actual selected-deck Office edit results: {path}')
        for slide, source in enumerate(comparison['sources'], 1):
            if source not in pages or source in results:
                raise ValueError(f'Unexpected or duplicated source in {path}')
            original_page = pages[source]['page']
            page_cases = [c for c in cases if c[0] == original_page]
            checks = [row for row in actions if row['page'] == slide]
            if [(c[1],c[3]) for c in page_cases] != [(a['target'],a['action']) for a in checks]:
                raise ValueError(f'Edit cases differ from the frozen manifest: {path}')
            fields = _native_fields(inspect_pptx(deck, tuple(pages[source]['size']), slide)['objects'])
            items = []
            for case, observed in zip(page_cases, checks):
                item = {'target':case[1], 'action':case[3], 'bbox':case[2], 'office':observed['status'],
                        'edit_passed':observed['status']=='EDIT_OBSERVED' and observed.get('saved_value') is True
                                      and observed.get('selection')!='multiple selection'}
                if case[3] == 'text':
                    reading = native_reading(fields, case[2])
                    item['field'] = reading
                    if reading['status'] == 'FOUND':
                        item['matches_reference'] = ''.join(reading['text'].split()) == ''.join(case[1].split())
                items.append(item)
            results[source] = {'job':comparison['job_id'], 'template_family':pages[source].get('family'),
                               'pptx_sha256':sha, 'cases':items}
    if set(results) != set(pages):
        raise ValueError(f'{root} contains {len(results)} of {len(pages)} frozen pages')
    return results


def compare(from_arm, to_arm):
    counts = Counter(dict.fromkeys(('newly_passed_edits', 'referenced_text_cases', 'wrong_to_correct',
                                    'correct_to_wrong', 'native_field_lost_or_ambiguous',
                                    'native_field_recovered', 'unknown', 'unchanged_correctness'), 0))
    regressions = []
    gains = []
    transitions = []
    for source, page in from_arm.items():
        for before, after in zip(page['cases'], to_arm[source]['cases']):
            identity = {'source':source, 'template_family':page.get('template_family'),
                        'target':before['target'], 'action':before['action']}
            if before['edit_passed'] and not after['edit_passed']:
                regressions.append(identity)
            if not before['edit_passed'] and after['edit_passed']:
                counts['newly_passed_edits'] += 1
                gains.append(identity)
            if before['action'] != 'text':
                continue
            old, new = before.get('matches_reference'), after.get('matches_reference')
            counts['referenced_text_cases'] += 1
            if old is False and new is True:
                transition = 'wrong_to_correct'
            elif old is True and new is False:
                transition = 'correct_to_wrong'
            elif old is not None and new is None:
                transition = 'native_field_lost_or_ambiguous'
            elif old is None and new is not None:
                transition = 'native_field_recovered'
            elif old is None or new is None:
                transition = 'unknown'
            else:
                transition = 'unchanged_correctness'
            counts[transition] += 1
            transitions.append({**identity,'before':before['field'],'after':after['field'],'transition':transition})
    return {'counts':dict(counts), 'lost_edits':regressions, 'newly_passed_edits':gains,
            'text_transitions':transitions,
            'unreferenced_text_accuracy':'UNKNOWN'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('control','region','combined','output'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--project-root',required=True,type=Path)
    parser.add_argument('--input-root',required=True,type=Path)
    args=parser.parse_args()
    check(args.project_root,args.input_root)
    manifest=read(HERE/'holdout/manifest.json')
    pages={p['sha256']:p for p in manifest['pages']}
    cases=read(HERE/'holdout/cases.json')
    profiles = {'control': {'region_text':False, 'page_reading':False},
                'region': {'region_text':True, 'page_reading':False},
                'combined': {'region_text':True, 'page_reading':True}}
    arms={name:record_arm(getattr(args,name),pages,cases,profile) for name,profile in profiles.items()}
    write(args.output, {'arms':arms, 'control_to_region':compare(arms['control'],arms['region']),
                        'region_to_combined':compare(arms['region'],arms['combined']),
                        'control_to_combined':compare(arms['control'],arms['combined'])})


if __name__ == '__main__':
    main()
