# 2026-11-26  卡片换色、照片标题与提前下载

本轮在第 08 页开发样本上重放一次真实上传的 Office 输入包。区域重建、校读和编译都走产品代码。评测程序在文件生成后才读取预设编辑用例。页内三项操作在旧包中通过一项，本轮通过三项。整页重新上传、陌生模板通过率、Go 和人工/WPS 验收均为 UNKNOWN。

| 实测项目 | 结果 |
| --- | --- |
| 卡片底色 | 保存重开后颜色保留。照片和标题检查区的变化比例均为 0。 |
| 替换照片标题 | 初始标题命中冻结参考。替换、保存、关闭、重开后新字保留。再删掉文字，标题条上没有可辨认的旧字。 |
| 换色后删照片 | 建筑照片消失。测量范围全部露出新底色，标题条、相邻照片和卡片外区域没有可测变化。 |
| 视觉复核 | 独立 MoMA 会话查看原图及五个编辑状态，三项均通过。复核范围是中间卡片。 |
| 内容可信度 | 三个实图照片标题进入复读。一个与原 OCR 一致，两个更改作为待确认文字交付。 |
| 下载与超时 | 宿主测试模拟后期超时和后期版本失去交付资格，检查能取回已核验的同一份 PPTX。真实上传期间提前下载的表现为 UNKNOWN。 |

首次卡片修补恢复了原生标题条的三个部件，但在文字笔画位置误挖了蓝色标题底层，真实 Office 三项全部失败。最终实现只带回与已验证的文字底色相符的部分。原文件的编辑能力由重新打开后的编辑与删除检查判定。

徐州第 10 页另用真实 Office 回执做了两层对照。旧报告的两个短读数都只在隐藏文字后的整页里读到，单独提取的底图里没有匹配。剩余的图形仍记为底图问题。这项对照复用已绑定原文件的 OCR 证据。

## 改动文件

- `product/app.py`、`product/runtime/host_baseline.py`、`product/runtime/runner.py` 提前公布通过 Office 的完整文件，保留哈希、评分和超时原因。
- `product/recovery/layered_card.py`、`product/recovery/marked_surfaces.py`、`product/parallel_rebuild.py` 保留标题条、照片及文字背后的已清理底色。
- `product/runtime/text_reading.py`、`product/runtime/page_reading.py` 优先复读有实际照片标题条证据的文字。
- `product/runtime/plate_audit.py` 区分底图读数与隐藏文字后整页的读数。后一种读数保留待查，不用来擦除对应底图。
- `tests/test_card_caption.py`、`tests/test_page_reading.py`、`tests/test_partial_review_copy.py` 覆盖新规则。`OPENCODE_RUNTIME.md` 说明下载时机和结果含义。以上路径相对于 `services/product/`。

## 验证与产物

运行 `./image2ppt.ps1 test` 共 39 项通过。将 `services/product` 和 `ppt-rebuild/src` 加入 PYTHONPATH 后，使用 `python -m pytest -q` 运行 test_background_plate_audit、test_page_reading、test_partial_review_copy、test_host_baseline、test_card_caption、test_card_edit_preservation、test_marked_execution 七个测试文件，共 70 项通过。

实图结果保存在独立实验目录 `nt26-checked-card`。4090 的 `caption-office.zip` 含实际 PPTX、保存重开预览和 Office 回执。`independent-visual.json` 记录独立视觉复核。PPTX 的 SHA-256 为 `851bc99945effa92abb82bfeecf47528e4f1d37db1639b8da681fc3e1f4d7b5d`。
