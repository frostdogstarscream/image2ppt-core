# 混合可编辑重建（实验功能）

正文用原生文本，复杂内容用 RGBA 图片层；输出仍通过原有 SlideIR 0.1 和固定 Artifact Tool 编译器。图片中的文字不能修改，不计入原生文本数。此功能只新增本地 CLI，不改变 HTTP 请求协议或自动触发在线服务。

## 使用顺序

以下命令在仓库根目录运行；`python` 指已安装本服务依赖的 Python。所有输出目录必须是新目录。分析文件和计划仍需与原图绑定。

```powershell
# 独立配置 VISION_LAYER_API_KEY；不要写进命令参数、计划或报告。
# 可选 VISION_LAYER_BASE_URL（默认 https://maas-api.antdigital.com/v1）
# 可选 VISION_LAYER_MODEL（默认 ming-image-0.1-design-layer）
python -m image2ppt_vision.cli decompose --provider ming --input 03.png --output layers-03-candidate

# 或导入已有的来源绑定图层清单及其相对路径素材
python -m image2ppt_vision.cli decompose --provider import --input 03.png --manifest reviewed/03.json --output layers-03-import

python -m image2ppt_vision.cli reconstruct --input 03.png --analysis analysis/03.json --plan plans/03.json --layers reviewed/03.json --output rebuilt-03
python -m image2ppt_vision.cli convert --input-dir source --analysis-dir analysis --plan-dir plans --layers-dir reviewed --compiler-root . --output hybrid-deck
```

在线提供方一次调用 `/images/edits`，使用 multipart 的 `model`、`prompt`、`image` 字段。请求超时 600 秒，不自动重试。图片下载是独立的无认证 GET；不保存返回的签名 URL、请求凭据或任意服务端错误正文。原始图片字节保存在 `raw/`，标准化 RGBA PNG 保存在 `assets/`，响应以哈希留痕。无凭据也会生成失败状态记录，不会回退调用其他模型。

Ming 接口形状依据 Cover Studio 的公开实现核对；尚未凭真实账号验证模型权限或效果。服务返回的图片顺序只作为候选顺序，不能自动证明背景正确或文字已去除。

## 清单与审核

`LayerManifest 1.0` 独立于 ReconstructionPlan。字段如下：

| 字段 | 含义 |
|---|---|
| schema/version | `LayerManifest` / `1.0` |
| source_sha256/image_size | 原图 SHA256 与像素画布 `[width,height]` |
| provider/model_id | `import` 或 `ming` 与实际模型/来源标识 |
| reviewed | 图层顺序、内容归属和残留文字已经过审核；模型默认 false |
| layers | 每层 `id,path,sha256,bbox,z,module,retained_text_ids` |
| native_text | 每条原生文字的 `text_id,z,module`；文字本身取自 OCR 和原有显式修订 |
| native_shapes | 每个原生形状的 `shape_id,z,module`；shape_id 必须来自原有计划 |
| repairs | 每项 `bbox,kind,text_ids,shape_ids,reviewed`；kind 为 `text_cleanup`、`shape_cleanup` 或 `background_completion` |

`bbox` 使用原图像素 `[x,y,width,height]`。图片层必须像素对齐、尺寸精确匹配 bbox、PNG RGBA；可用透明整页图，也可用带原坐标的裁剪图。路径必须位于清单所在目录内。每条 OCR 文字只能属于一个保留图片或一个原生文本对象。所有 z 唯一；同模块对象在 z 顺序上必须连续，才能形成真实 PPT 分组。背景一般 module=null。

在线候选不填写文字归属，且 reviewed=false，不能直接用于重建。审核者需要在**新的清单修订**中完成以下步骤：

1. 核对图层合成与原图，确定 z 和模块；检查背景是否包含旧主体残影。
2. 对保留像素文字填写 retained_text_ids；对原生文字填写 native_text。删除或清理原来的像素文字，保存新素材并更新哈希。
3. 为已清理文字记录覆盖其完整 OCR 框的 text_cleanup，以及 text_ids。检查所有层而非只检查底图，确认没有残留副本。
4. 为生成式补全区域记录 background_completion，逐项审核并设置 reviewed；最后设置清单 reviewed=true。

`review/image-layers-only.png` 只合成图片层，故意不叠原生文字，便于查看旧字。`review/repair-*.png` 展示清理/补全局部。内容正确性不能仅凭这些布尔值或结构校验得到证明；整页仍为 REVIEW_REQUIRED，human_acceptance=false。

原生字体从原图估计。指定原生文字若无法可靠拟合会失败，不会静默降为图片。简单形状通过 native_shapes 显式选择，必须有覆盖原形状的已审核 shape_cleanup，并通过原图像素校验；未经确认的形状不会叠加到图片上。复杂形状仍保留图片。

## 验证与当前边界

本地修复使用完整文字框判定模块。越界文字放到页面层，跨边界保留文字以完整图片对象呈现；底层对应区域透明挖除。清理越界造成超过现有 12 色阶背景归一化容差的结构改动时停止文字提取，保留像素并记录原因。

`--layers-dir` 下每个原图必须有同名 JSON，目录清单不允许缺页或多页。清单和所有素材被复制并校验到新输出，编译与复验不调用模型。

```powershell
python -m pytest services/vision/tests -q --basetemp outputs/new-test-directory
./image2ppt.ps1 test
```

三页真实试点通过后才扩到 14 页。实际文件回读只证明导出/渲染链路；PowerPoint/WPS 开启、编辑文字、移动模块与人工核对仍需单独记录。当前没有 MaaS 凭据的环境只能完成本地修复试点和离线/模拟提供方测试，不能宣称真实拆层验收。
