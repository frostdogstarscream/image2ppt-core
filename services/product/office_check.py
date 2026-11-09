"""Optional Windows smoke: edit a copy in actual PowerPoint, save and reopen."""
import argparse
import json
from pathlib import Path
import shutil
import win32com.client
from product.materialize import digest,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('pptx');parser.add_argument('output');args=parser.parse_args()
    source=Path(args.pptx).resolve();original=digest(source)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    copied=output/'editable-copy.pptx';shutil.copyfile(source,copied)
    app=win32com.client.Dispatch('PowerPoint.Application')
    deck=None
    try:
        deck=app.Presentations.Open(str(copied),ReadOnly=False,Untitled=False,WithWindow=False)
        deck.Slides(1).Export(str(output/'before.png'),'PNG',960,540)
        target=None
        for shape in deck.Slides(1).Shapes:
            if shape.HasTextFrame and shape.TextFrame.HasText and shape.TextFrame.TextRange.Text=='02':
                target=shape;break
        if target is None:raise ValueError('fixture editable number missing')
        target.TextFrame.TextRange.Text='99'
        deck.Save();deck.Close();deck=None
        deck=app.Presentations.Open(str(copied),ReadOnly=True,Untitled=False,WithWindow=False)
        assert any(s.HasTextFrame and s.TextFrame.HasText and s.TextFrame.TextRange.Text=='99' for s in deck.Slides(1).Shapes)
        deck.Slides(1).Export(str(output/'after.png'),'PNG',960,540)
        assert digest(source)==original
        result={'status':'PASS_APP_AUTOMATED','powerpoint_version':app.Version,'opened_saved_reopened':True,
                'native_number_edit_persisted':True,'original_unchanged':True,'human_acceptance':False}
        write(output/'report.json',result);print(json.dumps(result))
    finally:
        if deck is not None:deck.Close()
        # Never quit a pre-existing user application or close its other documents.

if __name__=='__main__':main()
