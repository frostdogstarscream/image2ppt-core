"""Source-only experimental requests, independent from the sealed API baseline."""
import base64,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'services/vision/src'))
from image2ppt_vision.storage import write_json
from image2ppt_vision.typography import font_status

PROMPT='''You restore a screenshot as editable PowerPoint, not redesign it. Image 1 is the original; image 2 is an OpenCV candidate contact sheet. OCR and CV are fallible evidence. Read the original carefully, correct all Chinese OCR errors. Return ONLY a JSON object with schema="ElementPlan/2", source_sha256, image_size, background_color, objects, text_decisions, limitations.
All boxes [x,y,width,height] use ORIGINAL pixel coordinates. objects must enumerate EVERY visible independent non-text element: panels/card backgrounds, title ribbons, separator lines, arrows, circular badges, logos, icon glyphs, decorative elements. Preserve nested boundaries and layering. Do not treat a whole page or a whole diagram as one raster object. A card background MUST be separate from its text and icon. Objects have id, kind(native_shape or raster_asset), bbox, z(integer), label, candidate_id(optional), module(optional). Native shapes also have geometry(rect/roundRect/ellipse/line/chevron/pentagon/rightArrow/leftArrow/trapezoid/parallelogram/hexagon/diamond/triangle), fill(6hex or none), line(optional {color:6hex,width_pt:number}), corner_radius_px(optional), fill_gradient(optional {angle_deg:number,stops:[{offset:0,color:6hex},{offset:100000,color:6hex}]}). Use editable native shapes for simple cards, circles, rectangles, lines, arrows and ribbons, accurately match fill/outline, including subtle light-blue panels. Use independently cropped raster_asset only for genuinely complex icons, photos, logos, textures. Each raster_asset also has mask(rect or ellipse), contains_text_ids:[OCR ids], and rationale. Prefer a tight raster icon including its full circle rather than many cropped fragments. No giant residual/background bitmap is allowed. No raster_asset may cover >18% of the page. A screenshot UI must be broken into panels/text/icon atoms. A transparent mask for an ellipse retains interior white glyphs. Do not use raster_asset containing text unless explicitly declared contains_text_ids and text_decisions marks those retained_in_asset. Never overlay native text on original raster text. Use z for background then panels then icons then native text. Every item should be independently movable; module may group a card's members.
text_decisions must cover EVERY OCR id exactly once with {id,text,action:native or retained_in_asset,bbox,color:6hex,bold:boolean,font_size_px:number,font_family:string,font_latin:string,font_confidence:number,font_evidence:string,z:integer,module:optional}. Correct text ONLY by reading source pixels; if unclear keep OCR and record in limitations. Match source typography, bbox, color, boldness and relative size. Every native text MUST specify font_family (East Asian), font_latin, font_confidence (0..1) and font_evidence (brief visual evidence). Use available_families supplied with the request; fonts are visual candidates, not verified metadata. Reuse consistent typography within visually matching roles. Font_size_px is source-pixel em size, usually 1.0 to 1.2 times OCR ink bbox height for one Chinese line. Preserve original wording and numbers, don't summarize. Optional runs:[{text,color,bold}] allow mixed colors within one line, must concatenate to text. Ensure explicit retained ownership for logo text and screenshot text inside raster assets. Do not copy OCR blindly. Missing decorative background can be recorded as a limitation but do not omit main diagram components. Aim for a complete object inventory, typically 50-130 non-text objects for these dense diagrams. CV candidate ids are hints, not mandatory selections. You may specify corrected boxes from original image. Do not return code or markdown.'''

def img(path):
    mime='image/jpeg' if path.suffix=='.jpg' else 'image/png'
    return {'type':'image_url','image_url':{'url':'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()}}

def main():
    root=ROOT/'outputs/supervised-elements-20261019'
    write_json(root/'settings.json',{'base_url':'https://moma.cmecloud.cn/v1','timeout_seconds':390})
    for page in ('02','03','05'):
        folder=root/'inventory-v2'/page;inv=json.loads((folder/'inventory.json').read_text(encoding='utf8'))
        evidence={k:inv[k] for k in ('source_sha256','image_size','texts')}
        evidence['available_families']=font_status()['families']
        evidence['candidates']=[{k:c[k] for k in ('id','bbox','geometry_hint','ocr_inside','parent_candidate')} for c in inv['candidates']]
        job=root/('plan-kimi-'+page);job.mkdir(exist_ok=False)
        write_json(job/'request.json',{'model':'kimi/kimi-k2.6','messages':[{'role':'system','content':PROMPT},{'role':'user','content':[{'type':'text','text':json.dumps(evidence,ensure_ascii=False)},img(folder/'source.png'),img(folder/'candidate-contact.jpg')]}],'max_tokens':26000})
    print('Prepared 3 source-only planner jobs; no credentials stored.')

if __name__=='__main__':main()
