"""Package honest first-round diagnostics, never promotes unknown QA."""
import argparse,json,shutil,csv
from pathlib import Path
from collections import Counter
from product.materialize import write,digest


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def package(output,baseline):
 out=Path(output).resolve();baseline=load(baseline);comparison=load(out/'comparison.json');reviews=load(out/'independent-review.json');office=load(out/'after-office/report.json');delivery=out/'delivery';delivery.mkdir(exist_ok=False)
 for label,title in [('before','修复前_3e97fac自动证据重放_第2和5页.pptx'),('after','修复后_区域事务验证候选_第2和5页.pptx')]:
  a=load(out/(label+'-artifacts.json'));shutil.copyfile(a['pptx'],delivery/title)
 for i,page in enumerate((2,5),1):shutil.copyfile(out/'after-office'/f'page-{i:02d}-before.png',delivery/f'第{page:02d}页_实际PowerPoint预览.png')
 pages=[];failures=[]
 for row,qa in zip(comparison['pages'],reviews['pages']):
  report=load(Path(row['after']['ir']).parent/'transactions.json');staged=[r for r in report if r['state']=='STAGED'];failed=[r for r in report if r['state']=='FALLBACK']
  # Provider unavailability is unknown final quality, not proof of pixel failure.
  failed_qa=set(i for f in qa['review'].get('findings',[]) for i in f.get('node_ids',[]))
  residual={r['id']:r['residual_check_status'] for r in qa['residuals']}
  review_pass=qa['review'].get('passed') is True
  accepted=[r['id'] for r in staged if review_pass and r['id'] not in failed_qa and residual.get(r['id'])=='PASS' and office['status']=='PASS_APP_AUTOMATED']
  quality=[]
  for r in staged:
   quality.append({'id':r['id'],'candidate_state':'STAGED','final_state':'VERIFIED_AUTOMATED' if r['id'] in accepted else 'FALLBACK',
       'reason':None if r['id'] in accepted else 'QA_UNAVAILABLE' if 'quality' in qa['review'] else 'QUALITY_GATE_NOT_PASSED',
       'office_edit':'PASS','residual_check':residual.get(r['id'],'UNKNOWN'),'human_acceptance':False})
  row_summary={'source_page':row['page'],'proposed':row['proposed'],'before_candidates':row['before']['candidate_counts'],
       'after_candidates':row['after']['candidate_counts'],'new_candidates':row['added_candidates'],'removed_candidates':row['removed_candidates'],
       'extraction_fallback_count':len(failed),'fallback_reason_counts':dict(Counter(r['reason'] for r in failed)),
       'automatic_verified_objects':accepted,'new_automatic_verified_objects':[i for i in accepted if i in row['added_candidates']],
       'go':qa['review'],'object_quality':quality,'recall':{'text':None,'asset':None,'surface':None},
       'recall_note':'Independent semantic reference denominators not established; PNG node count is not complete icon count.'}
  pages.append(row_summary);failures.extend({'source_page':row['page'],**r} for r in failed)
 summary={'schema':'RegionFixRound/1','status':'REVIEW_REQUIRED','acceptance':'FIRST_ROUND_EDITABILITY_GAIN_NOT_ACHIEVED',
       'baseline_totals':baseline['totals'],'pages':pages,'office_actions':len(office['records']),
       'outside_changed_pixels':sum(r['outside_pixels'] for r in office['records']),
       'tests':{'product_related':131,'core':39},'parallel_auto_qualified':False,'human_acceptance':False}
 write(out/'final-quality.json',summary);write(delivery/'失败对象统计.json',{'extraction_fallback':failures,'final_quality':pages})
 with (delivery/'原始12页诊断.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.writer(f);writer.writerow(['页码','文字提议','素材提议','背景提议','原生文字','原生Shape','透明PNG','整页截图'])
  for row in baseline['pages']:
   actual=row['actual_pptx'];writer.writerow([row['page'],*[row['proposed'][k] for k in ('text','asset','surface')],*[actual[k] for k in ('native_text','native_shapes','transparent_png','full_raster')]])
 text='''# r10 区域事务第一轮验证：第 2、5 页

本轮已实施共享字体提交状态解耦和局部 Surface 依赖门禁，但真实样本新增独立对象为 0，低编辑率尚未解决。两页独立 Go 复核均为 RemoteProtocolError / UNKNOWN，不计自动编辑化成功。交付 PPTX 是实际编译的验证候选，不是发布通过的产品结果。

## 基线与对比口径

原始 12 页交付：11 个文字框、0 个原生 Shape、44 个透明 PNG；第 5、7、9、11 页为整页图片。冻结 732 个原始文件及副本，哈希复核全部未改变。原任务未记录生成时精确 Git SHA，因此不声称其必然由指定提交生成。

A/B 对照直接加载 Git 提交 3e97fac 的 parallel_rebuild.py 与修复后实现，输入使用同一份既有自动规划、字体、LayerD、稳定资产和已完成 SAM 结果。没有手写 SlideIR，没有重跑 OCR 或人工改规划。重放包含既有 SAM 候选，所以不能把相对历史交付额外出现的 PNG 归功于本次代码修改。

| 源页 | 3e97fac 重放：文字 / Shape / 透明 PNG | 修复后候选 | 新增候选 | 提取 FALLBACK |
|---|---|---|---|---|
| 2 | 3 / 0 / 20 | 3 / 0 / 20 | 0 | 76 |
| 5 | 0 / 0 / 7 | 0 / 0 / 7 | 0 | 79 |

PNG 数包括规则线条等素材节点，不代表完整逻辑图标数量；中央云服务器不在独立 PNG 中。

## 修改与验证映射

| 修改 | 代码 | 验证 |
|---|---|---|
| 共享字体不再连带撤回 | parallel_rebuild.py、parallel_repair.py | 30 个同样式文字仅 1 个失败，其余 29 个保留；明确 part_of 的不可分割组合仍整体回退 |
| 全页 all_foreground 改为局部依赖 | parallel_regions.py、parallel_region_gate.py、parallel_rebuild.py | 无关失败不阻塞卡片；卡片内未解决文字或未知像素仍阻止替换；bbox 相交只作保守依赖，不作归属证明 |
| 区域记录及源像素证据持久化 | parallel_pipeline.py、parallel_rebuild.py | region-transactions、owner masks/backing、generation 文件清单保留；候选状态与最终 QA 分离 |
| 实际删除后的残影检查 | parallel_residual_gate.py | 30 个暂存对象在已证明 owner 范围内残影检查通过；不替代完整语义复核 |
| 诊断及复现 | diagnose_region_baseline.py、run_region_regression.py、review_region_regression.py | 原始 12 页统计、同输入 A/B、实际 PPTX 和失败对象清单均保存 |

## 实际检查结果

- 产品/规划器/相关服务测试：131 通过；固定核心测试：39 通过。
- 核心测试中一项旧 Skill 文案断言已更新为现行自动产品要求；本轮未修改 Skill 或 AGENTS.md。
- PowerPoint：30 个暂存对象分别移动/缩放、删除、保存重开，共 60 次操作；事务外像素变化为 0。
- 同一 PowerPoint 渲染器下，修复前后两页像素变化均为 0；27 个独立 PNG 及 source-plate 的字节哈希均未改变。
- Go：两次独立请求均传输失败，保留未知调用记录，没有重发相同请求。自动核验成功数不因 Office 通过而增加。
- 残影指标只检查有源像素归属证据的范围；未经证明的语义完整性仍不能宣称通过。人工/WPS 验收未执行，parallel_auto 保持关闭。

## 尚未解决的原因

第 2 页：21 个文字因源字形归属不确定、4 个文字因内容/归属未确定而回退；32 个 Surface 的局部前景仍未闭合。素材 19 个回退。

第 5 页：39 个文字均因源字形归属不确定而回退。解除全页门禁后，7 个 Surface 进入后续检查，其中 6 个属于当前不支持的复杂 Surface，1 个几何/颜色证据不足；其余 23 个局部前景未闭合。素材 10 个回退。

cloud_server 的稳定资产与 SAM 候选都因 source component belongs to protected peer 被拒绝。原云图继续保留在 source-plate 中，不能独立移动或删除；这不是完整透明 PNG 提取成功。

下一步应定位文字抗锯齿/亮边的归属证据，以及云图与标签/底座的真实边界，随后处理圆角/渐变底板；不能通过改低阈值或删除保护来绕过。第一轮先保留该负结果，不扩展到其他页面。
'''
 (delivery/'结果说明.md').write_text(text,encoding='utf-8-sig')
 for name in ('code.diff','all-tests.xml','focused-tests.xml','core-tests.txt','nonregression.json','final-quality.json'):
  shutil.copyfile(out/name,delivery/name)
 write(delivery/'文件哈希.json',{p.name:digest(p) for p in delivery.iterdir() if p.is_file()})
 print(json.dumps({'delivery':str(delivery),'pages':[{'page':r['source_page'],'fallback':r['extraction_fallback_count'],'automatic_verified':len(r['automatic_verified_objects'])} for r in pages]},ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('baseline');a=p.parse_args();package(a.output,a.baseline)
