"""Local vetoes on shadow recommendations. No generation or stage mutation."""
HARD_DEFECTS={'content_error','missing_text','wrong_number','ghosting','neighbor_damage','incomplete_object','clipping'}


def shadow_select(bundle,defects,comparison):
    from .parallel_quality_arbitrator import validate_review
    validate_review(defects,bundle);validate_review(comparison,bundle)
    if defects.status!='COMPLETE' or comparison.recommendation=='UNKNOWN':
        return {'status':'REVIEW_UNKNOWN','selected_candidate_id':None}
    if comparison.recommendation=='REJECT':return {'status':'REJECTED','selected_candidate_id':None}
    preferred=comparison.preferred_candidate_id
    candidate=next((r for r in bundle['candidates'] if r['candidate_id']==preferred),None)
    if not candidate or candidate['edit_probes']!='PASS' or candidate['residual']!='PASS':
        return {'status':'REJECTED','selected_candidate_id':None,'reason':'HARD_GATE'}
    findings=[f for f in [*defects.defects,*comparison.defects] if f.candidate_id==preferred]
    if any(f.severity in ('MODERATE','MAJOR') or (f.type in HARD_DEFECTS|{'other'} and f.severity!='NONE') for f in findings):
        return {'status':'REJECTED','selected_candidate_id':None,'reason':'UNACCEPTABLE_DEFECT'}
    return {'status':'SHADOW_RECOMMENDATION','selected_candidate_id':None,
            'recommended_candidate_id':preferred,'final_page_qa':'UNKNOWN','publish_eligible':False}
