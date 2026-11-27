# 源图边界细化与中性灰矩形补全

这是最小 ModulePlan 之后的本地确定性后处理，不修改模型原始响应，不计入独立模型能力评分。本阶段只增加有源像素依据的原生矩形，不移动、改色或删除既有对象，不进行自动修复循环。

## 边界与归属

`DerivedModuleReview/1` 保留原模型标签、锚点和自报置信度。仅对未归属对象或包含溢出触发的边界，向外搜索原边界 24 个源像素内的直线证据。相邻像素最大 RGB 差至少 6、支持比例至少 40% 才产生派生边界；2 像素以内的偏差不继续吸附其他边缘。所有边界搜索仍受原提案限制，不随重复处理扩大窗口。

重新计算对象归属、像素支持、模块重叠、包含问题和锚点冲突。全页背景单独记录 page scope；不强迫所有对象进入某个模块。边缘支持只是局部启发式证据，不等于语义边界正确，模块仍可能重叠或过宽。

## 图形补全

在原图中寻找低色差（RGB 通道差不超过 10）、亮度在 205—238 的连通块。只取宽 45—300、高 3—10 源像素、宽高比至少 8、矩形占有率至少 90% 的候选。

候选必须由一个 panel/workflow 模块包含；与源文字框、当前文字框、已有图片或尺寸相近的形状相交时拒绝。颜色由连通块像素中位数取得。新增对象使用独立 shape evidence、唯一对象 ID 和新的 z 值，原有 IR 节点逐字段保持不变。

这只能补全一类浅灰正文示意线。更淡、带渐变、彩色或有连接线干扰的对象可能漏检；复杂背景、模型虚构图形、真实文字重叠等问题仍需其他有证据的处理。

## 命令

```powershell
python -X utf8 scripts/source_graphics.py apply --folder <ModulePlan页面> --output <新页面>
node scripts/compose_text_ledger.mjs <新页面> --output <新导出目录> --batch-size 80
python -X utf8 scripts/source_graphics.py verify --folder <新页面> --pptx <实际PPTX>
python -X utf8 scripts/review_source_graphics.py --folder <新页面> --before-folder <原页面> --before-deck <原导出目录> --after-deck <新导出目录> --output <新审阅目录>
```

验证重新从保留的源图和父版 IR 计算候选及派生模块，核对台账与字体，并核对实际 PPTX 新增原生矩形的 geometry、fill 和坐标。回渲染必须有固定编译器的文件绑定记录；每条线的区域差异需要改善，区域外必须逐像素不变。通过状态仅为计算检查通过，PowerPoint/WPS 编辑与人工验收独立记录。
