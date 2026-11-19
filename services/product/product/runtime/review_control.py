"""Image-only negative controls for the independent Go reviewer's edit judgments."""
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from ..materialize import digest, write
from ..parallel_store import key


class ControlAnswer(BaseModel):
    defects: dict[str, str]


def verify_reviewer(reviewer, folder: Path, budget):
    import json
    folder.mkdir(exist_ok=True, parents=True)
    policy = key([reviewer.model, digest(__file__)])
    receipt = folder/(policy+'.json')
    if receipt.is_file():
        return json.loads(receipt.read_text(encoding='utf8'))
    if not budget.reserve('model_calls', {'tool':'review-controls','key':policy}):
        return {'passed':False,'status':'UNKNOWN'}
    kinds = ['clean','wrong_unit','old_number','flattened_card']
    random.SystemRandom().shuffle(kinds)
    expected = {}
    image = Image.new('RGB',(960,400),'white')
    draw = ImageDraw.Draw(image)
    try:
        font=ImageFont.truetype('arial.ttf',26)
    except OSError:
        font=ImageFont.load_default(size=26)
    for i,kind in enumerate(kinds):
        label=chr(65+i)
        expected[label]=kind
        y=95*i
        draw.text((5,y+28),label,fill='black',font=font)
        for x in (72,510):
            old=x==72
            if kind=='flattened_card':
                left=x+10
                if not old:
                    draw.text((x+148,y+19),'Sales',fill='black',font=font)
                draw.rounded_rectangle((left,y+10,left+124,y+68),10,fill='#bcdaf4')
                if old:draw.text((left+16,y+19),'Sales',fill='black',font=font)
            else:
                value='26 kg' if old else ('38 km' if kind=='wrong_unit' else '38 kg')
                draw.text((x+12,y+20),value,fill='black',font=font)
                if kind=='old_number' and not old:
                    draw.text((x+16,y+45),'26 kg',fill='#566370',font=font)
    img=folder/(policy+'.png')
    image.save(img)
    verifier_session=reviewer.session
    result=reviewer.ask('Classify each lettered row of these PowerPoint edit controls. Left is before, right is after. '
        'Intended operation: replace 26 with 38 keeping the unit, or move the blue card together with its label. '
        'Return defects mapping each displayed letter to clean, wrong_unit, old_number or flattened_card.',
        {},[img],ControlAnswer)
    observed=ControlAnswer.model_validate(result).defects
    record={'status':'TESTED','synthetic':True,'model':reviewer.model,'session':verifier_session,
            'usage':getattr(reviewer,'last_usage',None),'image_sha256':digest(img),'expected':expected,'observed':observed,'passed':observed==expected}
    write(receipt,record)
    return record

