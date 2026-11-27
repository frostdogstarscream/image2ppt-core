"""Verify already-rendered pilot artifacts and write a bounded experiment report."""
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/supervised-elements-20261019'
BUILDS = {
    '02': '2026-10-19T10-29-18-825Z-5530ba9d',
    '03': '2026-10-19T10-14-24-508Z-eb2299a2',
    '05': '2026-10-19T10-14-25-338Z-53450617',
}
DECK = RUN / 'deck/2026-10-19T10-35-32-134Z-f914eece'
NS = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def objects(path, slide=1):
    with ZipFile(path) as package:
        assert package.testzip() is None
        tree = ET.fromstring(package.read(f'ppt/slides/slide{slide}.xml'))
    result = {}
    for element in tree.find('p:cSld/p:spTree', NS):
        identity = element.find('.//p:cNvPr', NS)
        if identity is not None and identity.get('name'):
            result[identity.get('name')] = element
    return result


def equal_pixels(first, second):
    a = Image.open(first).convert('RGB')
    b = Image.open(second).convert('RGB')
    return a.size == b.size and ImageChops.difference(a, b).getbbox() is None


def main():
    rows = []
    for index, (page, stamp) in enumerate(BUILDS.items(), 1):
        base = RUN / f'calibrated-{page}'
        build = base / 'outputs' / stamp
        ir = read(base / 'slideir.json')
        assert hashlib.sha256((base / ir['source_image']).read_bytes()).hexdigest() == ir['source_sha256']
        assert equal_pixels(build / 'preview.png', DECK / f'preview/page-{index:03}.png')
        quality = read(build / 'element-quality/quality.json')
        source_w, source_h = ir['canvas']['source_px']
        largest = max(node['bbox'][2] * node['bbox'][3] / (source_w * source_h)
                      for node in ir['nodes'] if node['type'] == 'image')
        row = {'page': page, 'native_shapes': quality['pptx']['native_shape_objects'],
               'native_texts': quality['pptx']['native_text_objects'],
               'raster_images': quality['pptx']['pictures'],
               'largest_raster_page_fraction': largest,
               'whole_page_mae': quality['whole_page_mean_abs_rgb_error'],
               'foreground_mae': quality['foreground_abs_rgb_error'],
               'regions_within_10': quality['regions_within_10'],
               'regions_total': quality['regions_total'],
               'reviewer_verdict': read(RUN / f'review-qwen-final-{page}/review.json')['verdict'],
               'human_acceptance': False}
        emitted = objects(DECK / 'deck.pptx', index)
        assert set(emitted) == {n['id'] for n in ir['nodes']}
        rows.append(row)

    probe_results = []
    for case in read(RUN / 'edit-probes.json')['cases']:
        edited = objects(case['pptx'])
        baseline_path = RUN / f"calibrated-{case['page']}" / 'outputs' / BUILDS[case['page']]
        baseline = objects(baseline_path / 'result.pptx')
        node_id = case['node_id']
        result = {'page': case['page'], 'action': case['action'], 'node_id': node_id}
        if case['action'] == 'delete':
            assert node_id in baseline and node_id not in edited
        elif case['action'] == 'replace':
            actual = ''.join(node.text or '' for node in edited[node_id].findall('.//a:t', NS))
            assert actual == case['value']
        elif case['action'] == 'move':
            before = baseline[node_id].find('.//a:xfrm/a:off', NS)
            after = edited[node_id].find('.//a:xfrm/a:off', NS)
            ir = read(RUN / f"calibrated-{case['page']}" / 'slideir.json')
            expected = 24 / ir['canvas']['source_px'][0] * ir['canvas']['target_px'][0] * 9525
            delta = int(after.get('x')) - int(before.get('x'))
            assert abs(delta - expected) <= 2  # Integer EMU rounding only.
            assert after.get('y') == before.get('y')
            result.update(actual_delta_emu=delta, expected_delta_emu=expected)
        assert not equal_pixels(baseline_path / 'preview.png', case['preview'])
        result['actual_pptx_xml_passed'] = True
        result['actual_preview_changed'] = True
        probe_results.append(result)

    synthetic = ROOT / 'runs/synthetic-2026-10-19T10-32-14-433Z-d653fb3f'
    synthetic_same = equal_pixels(
        synthetic / 'outputs/2026-10-19T10-32-15-253Z-e0ddccf8/preview.png',
        synthetic / 'outputs/2026-10-19T10-34-14-435Z-8a7b330a/preview.png')
    assert synthetic_same
    transports = Counter(read(p)['status'] for p in RUN.rglob('transport.json'))
    summary = {'schema': 'SupervisedElementPilot/1', 'pages': rows,
               'source_hashes_verified': True, 'composed_previews_equal_reviewed_previews': True,
               'synthetic_default_vs_batched_pixels_equal': synthetic_same,
               'edit_probes': probe_results, 'transport_status_counts': dict(transports),
               'quality_acceptance': 'FAIL', 'powerpoint_checked': False, 'wps_checked': False,
               'human_acceptance': False}
    (RUN / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    table = '\n'.join(f"| {r['page']} | {r['native_shapes']} | {r['native_texts']} | {r['raster_images']} | {r['whole_page_mae']:.3f} | {r['foreground_mae']:.3f} | {r['regions_within_10']}/{r['regions_total']} | FAIL |" for r in rows)
    report = f'''# 02、03、05 页元素拆解与监督复检实验结果

实验结论：元素拆解链路已产出实际可编辑对象，忠实还原质量仍未达标。三页均保持 FAIL / needs_review，human_acceptance=false。本实验属于多次调用、模型修复和本地校准的组合流程，不计入此前的单请求独立 API 模型基线。

## 交付与对照

- [三页试验 PPTX（未达标）](元素拆解与监督试验_02_03_05_未达标.pptx)
- [原图在左、实际 PPTX 回渲染在右](source-vs-actual.jpg)
- [机器可读验证摘要](summary.json)
- [九项编辑试验路径](edit-probes.json)

## 实际文件统计

| 源页 | 原生形状 | 原生文字对象 | 局部栅格图片 | 全页 RGB MAE | 前景 RGB MAE | 局部 MAE≤10 | 质量结果 |
|---|---:|---:|---:|---:|---:|---:|---|
{table}

以上数量来自导出的 PPTX XML，并核对合并后对象清单。图标内部仍是栅格像素，不称为原生可编辑图标。没有整页残差背景图；背景采用原生底色，复杂纹理因此存在遗漏。扁平对象尚未建立语义分组。MAE 是 0—255 RGB 通道绝对误差均值，不是相似度百分比；前景指标突出有内容区域，避免白底掩盖误差。局部框可能重叠，因此局部通过数不能换算成页面准确率。

## 方法与模型证据

文字由现有 OCR 引擎识别，OpenCV 负责原始边缘、排除文字后的边缘、颜色区域和轮廓候选。Kimi-K2.6 接收原图、OCR 和候选，提出对象与文字计划；长请求超时后拆为对象规划与文字决策。固定编译器生成原生对象及独立局部图片，本地像素测量校准可可靠拟合的字体参数和纯色底板。Qwen3-VL-Plus 独立读取原图、实际回渲染和实际对象清单，输出复检意见。

保留了 20 次请求的传输记录：15 次收到响应、5 次传输失败（超时）。收到响应不代表 JSON 合法或内容正确，也不意味着上游未对超时请求计费。最终三份复检 JSON 均可解析且给出 FAIL；原始响应与全部失败尝试保留，未覆盖旧基线。模型供应方仅按接口返回名称记录，不由此证明路由背后的模型权重身份。

## 尚未解决的问题

- 02 页：中央盾形框和底座被错误简化，标题条端部、箭头方向/样式、渐变和图标遮挡仍有偏差。
- 03 页：中国移动图形被错误替换为圆形轮廓，部分栏标题、反馈箭头及底纹不准确。
- 05 页：面板标题渐变被变成深色块，UI 内的线条、文字布局及小字识别仍存在缺失或偏差。
- 字体测量不能验证文字语义。模型提出的 OCR 修改仍是候选，没有逐字人工确认；不能报告文字已经全部正确。
- 监督模型存在可核对的误报。例如最终 02 页报告称“中国移动 / China Mobile”缺失，但实际 XML 与回渲染均存在；又错误地声称校徽与文档图标相同。因此复检意见没有自动写回，必须先定位并验证证据。

## 技术验证及范围

vision 服务测试 357 项通过；项目协议测试 36 项通过。正常合成示例及 check_demo 的删除/修改/移动验证通过。三页实际文件分别做删除原生形状、移动独立图片、替换原生文字，共 9 项：检查最终 XML 和回渲染确实变化，图片实际移动 175005 EMU，与源图横移 24 px 的换算一致。测试是通过 IR 扰动后重新编译，不是 PowerPoint/WPS 内人工操作，也没有为全部节点证明无残影。

02 页完整对象集合出现 Artifact Tool 2.8.99 导出停滞，本轮采用显式分批导出再合并原生对象的恢复路径；默认编译路径不变。合成示例默认导出与分批导出的实际预览逐像素相同。最终合并三页预览与各自被复检的单页预览逐像素相同。来源 SHA256、PPTX CRC、对象清单及原有包校验通过；未做 PowerPoint/WPS 打开验收。

## 后续收敛点

优先补充基于原图轮廓的几何拟合与渐变参数恢复，约束模型不能用错误的预设形状代替复杂边界；对不确定文字保留疑点与来源裁剪。监督意见应经过对象定位、成对裁剪、实际 XML 与像素差异复核后才进入修复队列，并用新渲染检查是否真的降低误差。先在这三页达到约定的局部还原门槛，再扩展批量处理。
'''
    (RUN / 'RESULTS.md').write_text(report, encoding='utf-8')
    print(json.dumps({'report': str(RUN / 'RESULTS.md'), 'edit_probes_passed': len(probe_results),
                      'synthetic_pixels_equal': synthetic_same, 'pages': rows}, ensure_ascii=False))


if __name__ == '__main__':
    main()
