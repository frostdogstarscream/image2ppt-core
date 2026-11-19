from compare_region_reading import compare, native_reading


def test_position_finds_incorrect_text_without_using_the_answer():
    target=[10,20,100,25]
    label={'name':'label','bbox':target,'text':'会议材科理'}
    outside={'name':'elsewhere','bbox':[200,20,100,25],'text':'会议材料整理'}
    assert native_reading([label,outside],target)['text'] == '会议材科理'
    assert native_reading([label,dict(label,name='copy')],target)['status'] == 'AMBIGUOUS'
    assert native_reading([outside],target)['status'] == 'MISSING'


def test_new_field_and_corrected_field_are_counted_separately_from_ability_to_edit():
    def arm(match, native):
        row={'target':'会议材料整理', 'action':'text','field':{'status':native},'edit_passed':True}
        if match is not None:
            row['matches_reference']=match
        return {'page':{'cases':[row]}}
    good=arm(True,'FOUND')
    bad=arm(False,'FOUND')
    missing=arm(None,'MISSING')
    assert compare(bad,good)['counts']['wrong_to_correct']==1
    assert compare(good,bad)['counts']['correct_to_wrong']==1
    assert compare(missing,good)['counts']['native_field_recovered']==1
    assert compare(good,missing)['counts']['native_field_lost_or_ambiguous']==1
    assert compare(bad,good)['lost_edits']==[]
