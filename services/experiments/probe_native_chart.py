"""Edit one verified bar-chart value through PowerPoint's embedded workbook."""
import argparse
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
import win32com.client

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'services/product'))
from product.materialize import digest, write
from product.resource_lease import lease


def probe(source, output, page, name, origin):
    output.mkdir(parents=True,exist_ok=False)
    copy=output/'edit.pptx'
    shutil.copyfile(source,copy)
    report={'source_sha256':digest(source),'origin':origin,'slide':page,'chart':name,'status':'UNKNOWN'}
    with lease('office'):
        app=win32com.client.DispatchEx('PowerPoint.Application')
        deck=wb=None
        try:
            deck=app.Presentations.Open(str(copy.resolve()),False,False,False)
            slide=deck.Slides(page)
            shapes=[s for s in slide.Shapes if s.HasChart and (not name or s.Name==name)]
            if len(shapes)!=1:
                report['status']='NO_UNIQUE_NATIVE_CHART'
                return report
            shape=shapes[0]
            report['chart']=shape.Name
            slide.Export(str((output/'before.png').resolve()),'PNG',1280,720)
            chart=shape.Chart
            chart.ChartData.Activate()
            wb=chart.ChartData.Workbook
            sheet=wb.Worksheets(1)
            old=sheet.UsedRange.Value
            # The fixed compiler writes the category in column A and series in B.
            if len(old)<3 or len(old[0])<2 or not isinstance(old[1][1],(float,int)):
                report['status']='NO_VERIFIED_SERIES_CELL'
                return report
            expected=[list(row) for row in old]
            expected[1][1]=float(old[1][1])+max(1,abs(float(old[1][1]))*.20)
            sheet.Cells(2,2).Value=expected[1][1]
            wb.Close(True)
            wb=None
            chart.Refresh()
            deck.Save()
            deck.Close()
            deck=app.Presentations.Open(str(copy.resolve()),True,False,False)
            slide=deck.Slides(page)
            restored=slide.Shapes(report['chart']).Chart
            restored.ChartData.Activate()
            wb=restored.ChartData.Workbook
            actual=[list(row) for row in wb.Worksheets(1).UsedRange.Value]
            wb.Close(False)
            wb=None
            series=list(restored.SeriesCollection(1).Values)
            slide.Export(str((output/'after.png').resolve()),'PNG',1280,720)
            before=np.asarray(Image.open(output/'before.png').convert('RGB'),dtype='int16')
            after=np.asarray(Image.open(output/'after.png').convert('RGB'),dtype='int16')
            changed=np.abs(before-after).max(axis=2)>12
            report.update(cell='B2',before=old[1][1],after=expected[1][1],saved_reopened=True,
                          workbook_matches=expected==actual,series_matches=series==[r[1] for r in expected[1:]],
                          render_changed=bool(changed.any()))
            report['status']='PASS' if all(report[k] for k in ('workbook_matches','series_matches','render_changed')) else 'REVIEW_REQUIRED'
        except Exception as exc:
            report.update(status='REVIEW_REQUIRED',error_type=type(exc).__name__)
        finally:
            if wb is not None:wb.Close(False)
            if deck is not None:deck.Close()
            if app.Presentations.Count==0:app.Quit()
            write(output/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('pptx',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--page',type=int,default=1)
    p.add_argument('--name')
    p.add_argument('--origin',choices=['synthetic','fresh-upload'],required=True)
    a=p.parse_args()
    print(probe(a.pptx,a.output,a.page,a.name,a.origin)['status'])

