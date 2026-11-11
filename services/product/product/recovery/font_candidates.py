"""Installed font comparisons on source-bound multiline glyph samples."""
import numpy as np

from ..parallel_font_match import match
from ..materialize import write


def compare(graph, object_ids, source, fonts, directory):
    from ..parallel_text_candidates import peer_protection
    from .gradient_backing import fit_gradient
    styles={s.id:s for s in graph.text.styles};samples={};result={}
    for text in graph.text.texts:
        if text.id not in object_ids or text.spans:continue
        style=styles[text.style]
        try:
            protected=peer_protection(text,graph,source,np.zeros(source.shape[:2],bool))
            x,y,w,h=map(round,text.bbox)
            backing,_=fit_gradient(source,[x-3,y-3,w+6,h+6],protected)
            lines=text.text.split('\n');pitch=text.line_spacing_px or h/max(1,len(lines))
            for index,content in enumerate(lines):
                top=y+round(index*pitch);bottom=min(y+h,y+round((index+1)*pitch))
                crop=source[top:bottom,x:x+w].astype(float);bg=backing[top:bottom,x:x+w].astype(float)
                signal=np.abs(crop-bg).max(2);ink=signal>40
                if ink.sum()<8:continue
                yy,xx=np.nonzero(ink)
                strong=ink&(signal>=np.percentile(signal[ink],80))
                values=crop[strong];quant=values.astype('uint8')//8
                colors,counts=np.unique(quant,axis=0,return_counts=True)
                foreground=np.median(values[(quant==colors[counts.argmax()]).all(1)],axis=0)
                samples.setdefault(text.style,[]).append({'id':text.id+'-'+str(index),'text':content,
                    'source':crop,'background':bg,'foreground':foreground,'ink_left':int(xx.min()),
                    'ink_top':int(yy.min()),'ink_width':int(xx.max()-xx.min()+1)})
        except (ValueError,KeyError):continue
    for identifier,rows in samples.items():
        selection=match(sorted(rows,key=lambda row:-len(row['text']))[:4],fonts,styles[identifier])
        if not selection:continue
        current=styles[identifier]
        prior=match(sorted(rows,key=lambda row:-len(row['text']))[:4],
                    {'fonts':[f for f in fonts['fonts'] if f['family'] in (current.ea,current.latin)]},current)
        candidates=[]
        installed={(f['family'],f['bold']) for f in fonts['fonts']}
        for ranked in selection['ranked_candidates']:
            latin=current.latin if (current.latin,ranked['bold']) in installed else ranked['family']
            proposal={'ea':ranked['family'],'latin':latin,'bold':ranked['bold'],'size_px':ranked['size_px']}
            if proposal not in candidates:candidates.append(proposal)
        result[identifier]={'original':current.model_dump(),'original_family_comparison':prior,
                            'comparison':selection,'proposals':candidates}
    write(directory/'font-comparison.json',{'source_sha256':graph.source_sha256,'styles':result,
          'content_changed':False,'actual_powerpoint_verification_required':True})
    return result
