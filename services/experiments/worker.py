"""Offline single-request worker. Stdin/stdout transport also supports OpenSSH."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time

MARKER = 'IMAGE2PPT_EXPERIMENT='


def png(image):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def infer(request, root, device):
    from PIL import Image
    import numpy as np
    op = request['op']
    raw = base64.b64decode(request['png'], validate=True)
    image = Image.open(io.BytesIO(raw))
    result = {'input_sha256': hashlib.sha256(raw).hexdigest(), 'op': op, 'regions': []}
    start = time.perf_counter()
    folder = root / 'weights' / op
    if (folder / 'weights.json').exists():
        result['weights'] = json.loads((folder / 'weights.json').read_text())
    if op in ('pp-l', 'v3'):
        from paddleocr import LayoutDetection
        import paddle
        name = 'PP-DocLayout-L' if op == 'pp-l' else 'PP-DocLayoutV3'
        model = LayoutDetection(model_name=name, device=device, enable_mkldnn=False, **({'model_dir':str(folder)} if op=='pp-l' else {}))
        ready = time.perf_counter()
        data = next(iter(model.predict(np.asarray(image.convert('RGB'))[..., ::-1].copy(), batch_size=1))).json
        for row in data.get('res', data)['boxes']:
            result['regions'].append({'label':row['label'], 'score':float(row['score']), 'xyxy':list(map(float,row['coordinate']))})
        if device.startswith('gpu'):
            result['peak_device_bytes'] = paddle.device.cuda.max_memory_allocated()
    elif op in ('omni','doclayout'):
        import torch
        if op == 'omni':
            from ultralytics import YOLO
            model = YOLO(folder / 'icon_detect/model.pt')
        else:
            from doclayout_yolo import YOLOv10
            model = YOLOv10(str(folder / 'doclayout_yolo_docstructbench_imgsz1024.pt'))
        ready = time.perf_counter()
        for box in request.get('windows') or [[0,0,*image.size]]:
            x,y,w,h = box
            crop = image.crop((x,y,x+w,y+h)).convert('RGB')
            pred = model.predict(crop, imgsz=1280 if op=='omni' else 1024, conf=.05 if op=='omni' else .2, device=device,verbose=False)[0]
            for b in pred.boxes:
                l,t,r,bt = b.xyxy[0].tolist()
                result['regions'].append({'label':'icon' if op=='omni' else pred.names[int(b.cls.item())],
                    'score':float(b.conf.item()),'window':box, 'xyxy':[x+l,y+t,x+r,y+bt]})
        result['peak_device_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    elif op == 'vitmatte':
        import torch
        import cv2
        from transformers import VitMatteImageProcessor, VitMatteForImageMatting
        from pymatting import estimate_foreground_ml
        mask = Image.open(io.BytesIO(base64.b64decode(request['mask']))).convert('L')
        if mask.size != image.size:
            raise ValueError('coarse mask size mismatch')
        binary = np.asarray(mask) > 32
        radius = max(1,min(4,round(min(image.size)/50)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
        hard = cv2.erode(binary.astype('uint8'),kernel)
        if not hard.any() and binary.any():
            dist=cv2.distanceTransform(binary.astype('uint8'),cv2.DIST_L2,3)
            hard = binary & (dist==cv2.dilate(dist,kernel))
        soft = cv2.dilate(binary.astype('uint8'),kernel,iterations=2)
        tri = Image.fromarray(np.where(hard,255,np.where(soft,128,0)).astype('uint8'))
        proc = VitMatteImageProcessor.from_pretrained(folder,local_files_only=True)
        model = VitMatteForImageMatting.from_pretrained(folder,local_files_only=True).to(device).eval()
        ready = time.perf_counter()
        with torch.inference_mode():
            alpha = model(**proc(images=image.convert('RGB'),trimaps=tri,return_tensors='pt').to(device)).alphas[0,0,:image.height,:image.width].cpu().numpy().copy()
        alpha[~soft.astype(bool)] = 0
        alpha[hard.astype(bool)] = 1
        fg = estimate_foreground_ml(np.asarray(image.convert('RGB')) / 255., alpha)
        result['rgba'] = png(Image.fromarray(np.round(np.dstack((fg,alpha)).clip(0,1)*255).astype('uint8')))
        result['peak_device_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    elif op == 'local-layerd':
        import subprocess
        app = Path(os.environ.get('IMAGE2PPT_REMOTE_APP', 'F:/i2oc/app'))
        cfg = json.loads((app/'.runtime/parallel-services.json').read_text(encoding='utf-8-sig'))['layerd']
        token = Path(cfg['token_file'])
        if not token.is_absolute():
            token = app/token
        ready = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp)/'source.png', Path(tmp)/'layers'
            source.write_bytes(raw)
            subprocess.run([sys.executable,str(app/'services/layerd/client.py'),'--input',str(source),
                            '--output',str(out),'--url',cfg['url'],'--token-file',str(token)],
                           check=True, stdout=subprocess.DEVNULL,timeout=1200)
            meta=json.loads((out/'manifest.json').read_text(encoding='utf8'))
            if meta['source_sha256'] != result['input_sha256']:
                raise ValueError('local LayerD source mismatch')
            result['model_binding']=meta.get('model_binding')
            result['layers']=[{'z':r['z'], 'png':base64.b64encode((out/r['file']).read_bytes()).decode(),
                               'sha256':r['sha256']} for r in meta['layers']]
    elif op == 'vtracer':
        import vtracer
        from importlib.metadata import version
        result['library_version'] = version('vtracer')
        ready = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            src,svg = Path(tmp)/'asset.png', Path(tmp)/'asset.svg'
            src.write_bytes(raw)
            vtracer.convert_image_to_svg_py(str(src),str(svg),colormode='color',mode='spline',filter_speckle=2)
            result['svg'] = svg.read_text(encoding='utf8')
    else:
        raise ValueError('unsupported experiment worker')
    result.update(load_seconds=ready-start,inference_seconds=time.perf_counter()-ready)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    if (args.root/'deps').exists():
        sys.path.insert(0,str(args.root/'deps'))
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK','True')
    try:
        request = json.load(sys.stdin)
        if request['op'] in ('vitmatte','doclayout') and (args.root/'compat').exists():
            sys.path.insert(0,str(args.root/'compat'))
        response = infer(request, args.root, args.device)
    except Exception as exc:
        response = {'error':type(exc).__name__, 'detail':str(exc)[:500]}
    print(MARKER+json.dumps(response,ensure_ascii=True))

