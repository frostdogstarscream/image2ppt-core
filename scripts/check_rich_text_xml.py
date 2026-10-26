"""Check final exported runs, including explicit false and inherited defaults."""
import json
import sys
from zipfile import ZipFile
import xml.etree.ElementTree as ET

ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
with ZipFile(sys.argv[1]) as z:
    slide=ET.fromstring(z.read('ppt/slides/slide1.xml'))
shape=next(s for s in slide.findall('.//p:sp',ns)
           if s.find('p:nvSpPr/p:cNvPr',ns).get('name')==sys.argv[2])
expected=[('红色 ','FF0000',True),('Mixed ','008000',False),('正常 ','172B4D',False),('继承 Default','172B4D',True)]
actual=[]
for run in shape.findall('.//a:r',ns):
    props=run.find('a:rPr',ns)
    color=props.find('a:solidFill/a:srgbClr',ns)
    actual.append((run.find('a:t',ns).text,color.get('val') if color is not None else None,props.get('b')=='1'))
assert actual==expected, {'expected':expected,'actual':actual}
print(json.dumps({'passed':True,'final_xml_runs':actual},ensure_ascii=False))
