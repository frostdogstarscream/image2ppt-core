"""Package local r11 evidence without claiming independent or human approval."""
import argparse,json,shutil,csv
from pathlib import Path
import numpy as np
from PIL import Image
from product.materialize import digest,write


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def package(run,previous,shadow,conflicts,output):
    run,previous,shadow,conflicts,output=map(lambda p:Path(p).resolve(),(run,previous,shadow,conflicts,output))
    output.mkdir(parents=True,exist_ok=False);manifest=load(run/'run-manifest.json');baseline=load(run/'baseline/baseline.json')
    shutil.copyfile(run/'baseline/result.pptx',output/'r10-pages02-05-baseline.pptx')
    summary={'schema':'R11DeliveryEvidence/1','mode':manifest['mode'],'published':False,
             'go_review_final_artifact':'UNKNOWN','human_acceptance':False,'wps_acceptance':'NOT_TESTED',
             'text_recall':None,'native_text_edit_probe_gain':manifest['page05_edit_probe_gain'],
             'automatic_verified_gain':0,'pages':[],'earlier_shadow_review':load(shadow/'shadow-summary.json'),
             'cloud_conflicts':load(conflicts/'asset-candidates.json')}
    for trial in manifest['trials']:
        if 'artifacts' not in trial:continue
        page=trial['source_page'];folder=Path(trial['ir']).parent.parent
        text_folder='text-office-expanded' if (folder/'text-office-expanded/report.json').exists() else 'text-office'
        copy=lambda source,name:shutil.copyfile(source,output/name)
        copy(trial['artifacts']['pptx'],f'r11-page{page:02d}-review-required.pptx')
        copy(folder/'office/page-01-before.png',f'page{page:02d}-candidate.png')
        old_page=(2,5).index(page)+1
        copy(previous/f'after-office/page-{old_page:02d}-before.png',f'page{page:02d}-baseline.png')
        before=np.array(Image.open(output/f'page{page:02d}-baseline.png').convert('RGB'))
        after=np.array(Image.open(output/f'page{page:02d}-candidate.png').convert('RGB'))
        masks=np.zeros(before.shape[:2],bool)
        scene=load(Path(trial['ir']).parent/'scene.json')
        for identifier in trial['object_ids']:
            record=scene['selected'][identifier];x,y,w,h=map(int,record['bbox'])
            masks[max(0,y-4):min(masks.shape[0],y+h+4),max(0,x-4):min(masks.shape[1],x+w+4)]=True
            for action,subfolder in [('delete','office'),('move-scale','office'),('replace-short',text_folder),('replace-equal',text_folder),('replace-long',text_folder)]:
                for ext in ('pptx','png'):
                    file=folder/subfolder/f'01-{identifier}-{action}.{ext}'
                    if file.exists():copy(file,f'page{page:02d}-{identifier}-{action}-TEST-COPY.{ext}')
        outside=int(((before!=after).any(2)&~masks).sum())
        old_scene=load(run/f'baseline/page-{page:02d}/after/scene.json')
        stable=[]
        for identifier,record in old_scene['selected'].items():
            ref=record.get('asset_reference')
            if ref:
                current=Path(trial['ir']).parent/ref['path']
                stable.append({'id':identifier,'sha256':ref['sha256'],'preserved':digest(current)==ref['sha256']})
        summary['pages'].append({'source_page':page,'candidate_id':trial['candidate_id'],'status':trial['status'],
            'actual_inventory':trial['actual_inventory'],'outside_candidate_window_changed_pixels':outside,
            'stable_assets':stable,'text_probe':load(folder/text_folder/'report.json'),
            'residuals':load(folder/'residuals.json')})
    copy_reports={'frozen-baseline.json':run/'baseline/baseline.json','run-manifest.json':run/'run-manifest.json',
                  'shadow-summary-earlier-trial.json':shadow/'shadow-summary.json','cloud-candidates.json':conflicts/'asset-candidates.json'}
    for name,file in copy_reports.items():shutil.copyfile(file,output/name)
    write(output/'evidence-summary.json',summary)
    text=f'''# r11 第一批开发与真实证据

本包是候选试验，不是已发布 r11 版本。默认管线仍为 r10 parallel_review。

第 5 页新增 **{manifest['page05_edit_probe_gain']} 个通过真实编辑探针的中文 TextBox**：“算网资源支撑”。正常候选 PPTX 保持源文；标有 TEST-COPY 的文件是有意替换、删除或移动过的测试副本。

| 项目 | 结果 |
| --- | --- |
| 第 2 页新增已测原生文字 | 0 |
| 第 5 页新增已测原生文字 | {manifest['page05_edit_probe_gain']} |
| 短文本/等长替换、保存重开 | PASS |
| 移动/缩放、删除、保存重开 | PASS |
| 标题源 Owner 删除残影 | 6,406 像素范围内确认残影 0，未知 0 |
| 四倍长文本 | 检出溢出，不支持 |
| 独立 Go 最终验收 | UNKNOWN，自动核验新增 0 |
| cloud_server | 两个候选均 BLOCKED；冲突 1,238 / 1,761 像素 |
| 人工/WPS 验收 | 未执行 |
| 文本召回率 | UNKNOWN，未建立独立参考分母 |

第 5 页其他内容保留原图及既有稳定 PNG。source-plate 不计可编辑对象。本轮不是整页完整重建。

放大查看删除图时仍可见极淡的局部背景轮廓；自动残影 PASS 只覆盖既定源 Owner 和容差，不能解释为所有低对比度像素完全消失。这部分仍需独立视觉确认。

Go 首轮把替换测试图当成最终成品而拒绝；修正图像角色后的新请求没有有效结果，留存 UNKNOWN，未自动重发。随附的 Go 收据属于较早试验，不能沿用为本包新 PPTX 的验收批准。

联合背景、混合卡片、完整云图、12 页回归和新上传验证尚未完成。全部候选和模型建议均没有发布权限；parallel_auto 继续关闭。

详细实现与复跑命令见工作树 services/product/PARALLEL_R11.md；机器可读依据见 evidence-summary.json。原始历史文件未覆盖。
'''
    (output/'结果说明.md').write_text(text,encoding='utf-8-sig')
    write(output/'file-hashes.json',{p.name:digest(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({'output':str(output),'pages':[{'page':p['source_page'],'outside_changed_pixels':p['outside_candidate_window_changed_pixels'],
          'stable_preserved':all(a['preserved'] for a in p['stable_assets'])} for p in summary['pages']]},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','previous','shadow','conflicts','output'):p.add_argument(name)
    a=p.parse_args();package(a.run,a.previous,a.shadow,a.conflicts,a.output)
