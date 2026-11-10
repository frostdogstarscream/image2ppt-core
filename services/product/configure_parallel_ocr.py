"""Explicit install-time verification of existing local PP-OCRv4, no downloads."""
import argparse
from pathlib import Path
from product.materialize import digest,write
EXPECTED={
 'det.onnx':'d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9',
 'cls.onnx':'e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c',
 'rec.onnx':'48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b'}

def configure(directory,output):
    directory=Path(directory).resolve();records={}
    for name,sha in EXPECTED.items():
        file=directory/name
        if digest(file)!=sha:raise ValueError('unexpected local PP-OCRv4 model: '+name)
        records[name]={'path':str(file),'sha256':sha,'read_only_consumer':True}
    write(output,{'schema':'LocalOCRModels/1','engine':'PP-OCRv4','models':records,'runtime_download':False})
    print('Verified three offline PP-OCRv4 files; no model files modified.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('output');a=p.parse_args();configure(a.directory,a.output)
