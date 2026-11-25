"""Private SAM responses remain distinguishable from service and network failures."""
import json
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import numpy as np
import pytest
from PIL import Image

from product.materialize import digest, write
from product.runtime import sam_evidence


def inputs():
    return (SimpleNamespace(image_size=[20, 16], source_sha256='f'*64),
            SimpleNamespace(id='icon', bbox=[2,2,10,10],model_dump=lambda: {'id':'icon','bbox':[2,2,10,10]}))


@pytest.mark.parametrize('failure,status', [
    (URLError(ConnectionRefusedError()),'SERVICE_OFFLINE'),
    (HTTPError('','503','',{},None),'REQUEST_FAILED'), (URLError(TimeoutError()),'REQUEST_TIMEOUT')])
def test_sam_transport_receipt_without_secret(tmp_path,monkeypatch,failure,status):
    def infer(*args):
        raise failure
    monkeypatch.setattr(sam_evidence,'infer_sam',infer)
    graph,asset=inputs()
    _,_,out=sam_evidence.prepare_mask('image.png',graph,asset,tmp_path)
    assert out['status']==status and out['via']=='inference'
    monkeypatch.setattr(sam_evidence,'infer_sam',lambda *args: pytest.fail('offline request repeated'))
    assert sam_evidence.prepare_mask('image.png',graph,asset,tmp_path,offline=True)[2]['via']=='batch_offline'


@pytest.mark.parametrize('pixel,status', [(None,'sam_empty_mask'), ((14,17),'sam_no_owned_candidate'),
                                          ((5,5),'MASK_READY')])
def test_sam_bound_receipt_reuse_with_checked_mask(tmp_path,monkeypatch,pixel,status):
    graph,asset=inputs()
    calls=[]
    def infer(source,plan,out,version,ids):
        calls.append(ids)
        out.mkdir(parents=True,exist_ok=True)
        masks=np.zeros((1,16,20),dtype='uint8')
        if pixel:
            masks[0,pixel[0],pixel[1]]=1
        np.savez_compressed(out/'icon.npz', masks=masks, scores=[.8])
        write(out/'icon.json',{'source_sha256':graph.source_sha256,'plan_sha256':digest(plan),'version':version,
              'status':'SUCCEEDED','image_size':graph.image_size,'checkpoint_sha256':'a'*64,'code_revision':'abc',
              'mask_file':'icon.npz','mask_sha256':digest(out/'icon.npz')})
        return out
    monkeypatch.setattr(sam_evidence,'infer_sam',infer)
    plan,out,new=sam_evidence.prepare_mask('image.png',graph,asset,tmp_path)
    assert new == {'status':status,'via':'inference'}
    if pixel is None:
        # Frozen r10 calls the shared loader with its original default.
        assert sam_evidence.load_sam(out,asset,graph.source_sha256,digest(plan),
                                     graph.image_size,'sam3',individual=True)[1] == 'sam_no_owned_candidate'
    assert sam_evidence.prepare_mask('image.png',graph,asset,tmp_path,offline=True)[2] == {'status':status,'via':'cache'}
    assert calls == [['icon']]
    (out/'icon.npz').write_bytes(b'incomplete')
    assert sam_evidence.prepare_mask('image.png',graph,asset,tmp_path)[2] == new
    assert calls == [['icon'],['icon']]


def test_batch_records_unavailable_sam_and_keeps_verified_icons(tmp_path, monkeypatch):
    from test_r1205_agent_asset_stage import scene
    from product import matting
    from product.runtime.batch_backbone import _chain_stage_from_spec
    graph, layers, baseline = scene(tmp_path)
    calls = []
    def offline(*args):
        calls.append(args[-1])
        raise URLError(ConnectionRefusedError())
    monkeypatch.setattr(sam_evidence, 'infer_sam', offline)
    monkeypatch.setattr(matting, 'available', lambda: False)
    output, ids, failures = _chain_stage_from_spec(initial_graph=graph, initial_baseline=baseline,
        fonts={'fonts':[]},source_path=str(layers/'source.png'), layers_dir=str(layers), deferred_path=None,
        page_cache_dir=tmp_path, directory=tmp_path, tag='asset-batch', trials_spec=[], frame_overrides={},
        skip_ids=set(), include_assets=True, include_containers=False,
        wanted_filter={'first','second'}, sam_enabled=True)
    assert not failures and set(ids) == {'first','second'} and calls == [['first']]
    selected = json.loads((output/'scene.json').read_text('utf8'))['selected']
    receipts = sorted((tmp_path/'asset-batch').glob('asset-*/strategy-comparison.json'))
    assert len(receipts) == 2
    for path,via in zip(receipts, ('inference','batch_offline')):
        result = json.loads(path.read_text('utf8'))
        item = selected[result['object_id']]
        assert result['staged_strategy'] == item['protection_strategy'] == 'verified-backing'
        assert item['payload']['seed_method']
        sam = next(a for a in result['attempts'] if a['strategy']=='sam')
        assert sam['via'] == via and sam['status'] == 'SERVICE_OFFLINE'
        assert (path.parent/result['ranked'][0]['contrast_preview']).is_file()
    assert (np.asarray(Image.open(output/'first.png'))[...,3] == 0).any()
