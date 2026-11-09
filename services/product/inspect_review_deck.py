"""Read/export the real review deck in PowerPoint, then test editing a copy."""
import argparse
from pathlib import Path
import shutil
import win32com.client
from product.materialize import write,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('pptx');parser.add_argument('output');args=parser.parse_args()
    source=Path(args.pptx).resolve();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    original=digest(source)
    app=win32com.client.Dispatch('PowerPoint.Application');deck=None
    try:
        deck=app.Presentations.Open(str(source),ReadOnly=True,Untitled=False,WithWindow=False)
        counts=[]
        for index in range(1,deck.Slides.Count+1):
            slide=deck.Slides(index)
            counts.append(sum(1 for s in slide.Shapes if s.HasTextFrame and s.TextFrame.HasText))
            slide.Export(str(output/f'page-{index:02d}.png'),'PNG',1672,941)
        assert len(counts)==12 and all(counts),'Missing page or native text'
        deck.Close();deck=None
        copy=output/'edit-check-copy.pptx';shutil.copyfile(source,copy)
        deck=app.Presentations.Open(str(copy),ReadOnly=False,Untitled=False,WithWindow=False)
        target=next(s for s in deck.Slides(2).Shapes if s.HasTextFrame and s.TextFrame.HasText and s.TextFrame.TextRange.Text.strip()=='02')
        target.TextFrame.TextRange.Text='99';deck.Save();deck.Close();deck=None
        deck=app.Presentations.Open(str(copy),ReadOnly=True,Untitled=False,WithWindow=False)
        assert any(s.HasTextFrame and s.TextFrame.HasText and s.TextFrame.TextRange.Text=='99' for s in deck.Slides(2).Shapes)
        assert digest(source)==original
        write(output/'report.json',{'status':'PASS_APP_AUTOMATED','slides':12,'powerpoint_version':app.Version,
            'native_text_objects_per_slide':counts,'edited_copy_saved_reopened':True,'original_unchanged':True,
            'human_acceptance':False})
        print('POWERPOINT_12_PAGES_OPENED_EXPORTED_EDIT_COPY_VERIFIED')
    finally:
        if deck is not None:deck.Close()


if __name__=='__main__':main()
