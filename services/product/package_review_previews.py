"""Build a portable page gallery and ZIP from an existing review delivery."""
import argparse
import html
from pathlib import Path
import zipfile
from PIL import Image,ImageOps,ImageDraw


def main():
    parser=argparse.ArgumentParser();parser.add_argument('delivery');args=parser.parse_args()
    root=Path(args.delivery).resolve()
    images=[root/f'page-{i:02d}.png' for i in range(1,13)]
    assert all(p.is_file() for p in images)
    contact=Image.new('RGB',(1500,1260),'#e7edf3');draw=ImageDraw.Draw(contact)
    for i,p in enumerate(images):
        x=(i%3)*500;y=(i//3)*315
        draw.text((x+12,y+10),f'{i+1:02d}',fill='#163253')
        contact.paste(ImageOps.contain(Image.open(p).convert('RGB'),(480,270)),(x+10,y+35))
    contact.save(root/'overview.png')
    links=''.join(f'<a href="#page-{i:02d}">{i:02d}</a>' for i in range(1,13))
    pages=''.join(f'<section id="page-{i:02d}"><h2>第 {i} 页</h2><a href="{p.name}" target="_blank"><img loading="lazy" src="{p.name}" alt="第 {i} 页整页预览"></a></section>' for i,p in enumerate(images,1))
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>12 页 PPT 人工复核稿</title><style>body{margin:0;background:#e9eef4;color:#163253;font:16px system-ui,sans-serif}header{padding:24px;max-width:1400px;margin:auto}h1{font-size:26px}nav{position:sticky;top:0;background:#ffffffed;padding:12px;display:flex;gap:12px;justify-content:center;z-index:2;flex-wrap:wrap}a{color:#0868c6}nav a{padding:6px 14px;background:#edf5ff;border-radius:5px;text-decoration:none}main{max-width:1500px;margin:auto;padding:16px}section{scroll-margin-top:76px;margin-bottom:32px}h2{font-size:18px}img{display:block;width:100%;height:auto;background:white;box-shadow:0 3px 12px #17304a22}p{line-height:1.7}</style><header><h1>12 页 PPT · 人工复核稿</h1><p>按原始页序排列。点击图片可打开完整分辨率预览。此稿尚未人工验收；纯斜体差异按约定忽略。</p><a href="Image2PPT-12pages-human-review.pptx">打开可编辑 PPTX</a></header><nav>'''+links+'</nav><main>'+pages+'</main></html>'
    (root/'index.html').write_text(document,encoding='utf-8')
    archive=root.parent/'Image2PPT-12pages-human-review.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(root.iterdir()):
            if path.is_file():zipped.write(path,path.name)
    print(str(archive))


if __name__=='__main__':main()
