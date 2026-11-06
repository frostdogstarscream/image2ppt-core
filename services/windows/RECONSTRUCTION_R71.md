# Windows r7.1 重建与验收

本入口继续使用 `reconstruct-r7` 和 `qa-r7` 命令名。r7.1 是已有 r7 的增量修复，
不调用模型 API，不重新生成识别结果。完整生产运行仍要求冻结的 12 页 batch；
关系模块支持其他画布，并不意味着这个业务入口已扩展为任意页数。

本轮正式 4090 运行已完成 12 页生成与最终整稿验收，目录为
`F:/code/image2ppt/outputs/r71-v4/runs/r71-01`。封面候选通过；38 项编辑探针为
17 PASS、18 REVIEW_REQUIRED、3 FAIL。失败均为第 3、4、10 页长文本压力替换的真实保护区碰撞，
因此自动验收为 FAIL，不能称为 r7.1 全部验收通过。本机完整重建测试为 295 PASS、1 SKIP，
协议测试为本轮 39 PASS。历史 `RECONSTRUCTION_R7.md` 继续保存 r7 当时的事实。

## 输入与运行目录

输入接受 `TypographyBatch/1` 或 `R7Batch/1`。r7.1 默认选用真正的 r7 batch；r6 用于
来源证据和已知负例。仅提供 PPTX 不足以恢复完整来源链。

- TypographyBatch 输入需要该格式约定的字体清单、每页 IR/PPTX、源图、素材、文字掩码与行清单。
- R7Batch 输入需要其声明的运行 manifest 和哈希绑定的父 batch，直到可验证的 r6 证据祖先。
- 缺文件、父链循环、源图／字体／素材／掩码变化均拒绝复用，不从相邻目录猜测来源。
- 每次新尝试使用全新目录。resume 只允许同一目录及相同输入、代码、字体、运行环境和关系配置。

在已安装 `image2ppt-local.cmd` 及运行时的 Windows 项目根目录执行。将下列占位路径替换成
实际输入及尚不存在的新目录；先冻结本轮代码，再开始 pilot，代码变化后改用新目录。

```bat
REM 只生成对象归属诊断，不构造修复候选
image2ppt-local.cmd rebuild reconstruct-r7 --input <r7-batch.json> --output <new-r71-run> --mode shadow

REM 局部验证：封面唯一容器候选及第2/8页已修复对象NO_OP
image2ppt-local.cmd rebuild reconstruct-r7 --input <r7-batch.json> --output <new-r71-run> --mode gated --pages 01 02 08

REM 冻结条件不变时完成其余页，重新合成12页并执行整稿QA
image2ppt-local.cmd rebuild reconstruct-r7 --input <r7-batch.json> --output <new-r71-run> --mode gated --resume <new-r71-run>

REM 原环境重新检查，或在独立新目录使用允许的验证代码更新
image2ppt-local.cmd rebuild qa-r7 --run <new-r71-run>
image2ppt-local.cmd rebuild qa-r7 --run <new-r71-run> --output <new-r71-qa-run>
```

`shadow` 当前输出注册表及冲突统计。`gated --pages` 输出局部场景，不能称为完整交付。
完整批次的串联回归可把第一次新运行的 `batch.json` 作为第二次输入，使用另一个新运行目录。
`--pages` 不生成完整 batch；局部幂等验证使用同一个已完成的正式 batch，在两个独立新目录重跑，
比较父场景与两轮的场景指纹、对象／证据／素材集合及同环境实际渲染。它证明同父重复幂等，
不宣称已执行两个完整批次串联，也不使用同目录缓存恢复代替重跑。

## 自定义关系清单

可增加 `--relations <catalog.json>`。默认清单绑定徐州已审核源图哈希；不能将其坐标直接用于
另一套图片。自定义清单采用 `RelationCatalog/1`，每个页面单独声明预期对象，坐标为源图像素
`xyxy`。`source_sha256` 如提供必须匹配真实源图；生成的关系规格始终绑定实际源图哈希。

```json
{
  "schema": "RelationCatalog/1",
  "coverage_scope": "declared_relations_only",
  "pages": {
    "01": {
      "elements": [
        {
          "id": "red-icon",
          "kind": "ICON_PRESENCE",
          "repairable": false,
          "bbox_xyxy": [18, 8, 52, 32],
          "segmentation": "color_distance",
          "foreground_rgb": [180, 20, 20],
          "color_tolerance": 10
        }
      ]
    }
  }
}
```

上例说明格式，来自合成测试，不能作为业务图像的审核证据。配置页面 ID 与入口使用的两位页号
一致。所有区域必须在真实画布内。源像素掩码在检查候选前冻结，保存文件哈希、原点和尺寸。

当前处理器区分标签、箭头、装饰、图标存在性和卡片外边界。标签须声明 `text_target_id`
及 `text_bbox_xyxy`，以绑定文字贡献；箭头方向支持 right、left、up、down。
未支持的关系、无独立 CLEAN_PLATE / TEXT_OVER_SURFACE 证据、未定义范围均为待复核。
明确不适用时使用 `applicable:false` 及 `not_applicable_reason`，不能以它代替未覆盖。

默认扩展检查包含第 3 页六图标、第 4 页四层卡片及三处向上箭头、第 10 页三主卡片；
这些对象均 `repairable:false`。卡片外边界检查不代表圆角、填充与子对象遮挡已完整通过。
已有第 2、8 页对象通过时不会新增候选或事务。

## 状态、封面和事务

FAIL 优先于 REVIEW_REQUIRED，适用项全部通过才 PASS；全部有理由地不适用才 NOT_APPLICABLE。
UNKNOWN 读入后保持待复核，非法状态及必需探针缺失为失败。未知底板不能掩盖确定的越界或保护区损伤。
长文本容量限制单列 `UNSUPPORTED_REFLOW`，不能吞掉权限错误、渲染错误或其他对象碰撞。

封面只尝试 `paragraph-l003` 的 `[142,169,453,212] → [142,157,453,236]` 容器变化，
其余文字和排版属性冻结。必须通过实际几何、视觉不变及短替换检查；候选失败就恢复完整父场景，
不会继续扩大容器或放宽掩码。安全容器通过也不代表支持任意长文本自然重排。

修复授权、候选、验收和提交绑定同一实际事务。共享区域成员和依赖闭包必须共同进入事务；
源图、父场景、对象集合、掩码及证据变化会拒绝提交。验收收据不能跨事务复用。

## QA 复用限制与输出

不带 `--output` 的 QA 要求当前代码与冻结环境一致。新目录 QA 当前只允许
`ppt-rebuild/src/qa_status.py`、`ppt-rebuild/src/r7_edit_checks.py` 的验证代码变化；
其他生成依赖或环境改变要求新的重建运行。不能用新目录参数绕过编译器、字体或运行时指纹。

实际编辑探针在最终交付 PPTX 的副本上按 presentation 页面关系定位目标，检查未编辑包部件
和其他页面；原文件保持只读。每页预期探针在基线读取时冻结，不会因目标丢失而跳过检查。

新交付目录保存正常／去字 PPTX、每页 source/r6/r7/r71/deleted 图、场景、注册表、关系规格、
编辑探针、逐页报告及 HTML 对照。`hard-failures.json`、`review-required.json`、
`object-coverage.json` 与 `qa.json` 使用同一汇总结果；`delivery.json` 保存文件 SHA-256
和独立的自动验收状态。保留的父页面不等于缺陷已修复。

PowerPoint、WPS、人工审阅单独记录。原生文字、Shape、独立 PNG 和真正 PPT 分组也应分开统计；
独立 PNG 可移动不代表内部路径可编辑。

## 实际 Office 应用检查

在具备相应应用的 Windows 主机上，使用新交付文件和两个不同的新目录：

```powershell
& .\scripts\check_r71_office.ps1 -InputPptx '<r71-delivery.pptx>' -OutputDirectory '<new-powerpoint-check>' -Application PowerPoint
& .\scripts\check_r71_office.ps1 -InputPptx '<r71-delivery.pptx>' -OutputDirectory '<new-wps-check>' -Application WPS
```

脚本拒绝已有输出目录，先复制输入，再以无窗口、未命名方式打开诊断副本。默认操作为第 1 页
`paragraph-l003` 的逐段短替换、第 3 页正文逐段替换、吊车图标组移动和原生卡片改色，保存后
重新打开检查；导出第 1、2、3、8 页的编辑前／保存后图像。目标名、页号、替换串、移动距离、
颜色、输出文件名和导出尺寸均可通过参数指定。该有界检查要求目标每段只有一个源 run，
保留段落分隔和格式；遇到其他结构会记录问题，不自动合并 runs。

报告保存脚本 SHA-256、输入前后 SHA-256、打开／保存／重开页数、页面顺序、实际请求及检查结果。
保存文件的 OOXML 重新核对页面顺序、对象成员／类型、非目标文本和几何，以及目标段落／run 数。
Office 序列化发生未预期结构变化时保留 REVIEW_REQUIRED，不能只因文件成功保存就通过。
这份检查不代替像素门禁、字体保真全检或人工视觉验收。

应用发现依据实际 COM 注册。请求的 WPS 没有注册类时写入 NOT_RUN 和原因，退出码为 2；
未通过或异常退出码为 1；满足该脚本范围才返回 PASS_APP_AUTOMATED 和退出码 0。
不修改 Trust Center、Protected View 或全局提示设置。只关闭脚本自己打开的文稿；存在预先运行
的应用进程、原有文稿、剩余文稿或无法确认的保护视图时不退出应用。不使用 taskkill。

本轮实际 PowerPoint 16.0 检查已完成，报告为
`D:/Code/image2PPT/outputs/r71-review/powerpoint-checks/run01/report.json`，结果 PASS_APP_AUTOMATED。
打开、保存、重开均 12 页；封面及第 3 页编辑、非目标语义与页序检查通过，第 2、8 页前后导出图
SHA-256 相同。输入成品前后哈希均为 `616379bd143d11a475f681a908bd7e3a1281161e342217247f971053b64b94ef`。
运行前已有的两个文稿保留，应用未退出。WPS 独立报告位于同级 `wps-checks/run01/report.json`，
因未注册 KWPP/WPP 类而记录 NOT_RUN。应用通过不消除整稿长文本压力检查的三项 FAIL。

工程检查运行完整重建测试和 `./image2ppt.ps1 test`。编译器或 OOXML 适配器有变化时，追加
合成 demo、`scripts/check_demo.mjs` 与实际文件预览。执行命令、日志和产物哈希随新运行保留。
