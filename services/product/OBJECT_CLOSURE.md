# 自动对象闭环（影子路线）

本路线在 `25ebee1` 上新增内部 `LogicalObjects/2`、`ObjectRepairs/1`、`ObjectReview/1` 协议，公共 VisualPage、SlideIR 及固定编译器不变。`LogicalObjects/1` 通过显式适配器读取；原始模型输出和失败记录保留。`auto` 可通过既有上传接口的 `object_extraction=auto` 查询参数选择。环境默认仍为 `off`；真实回归未达门槛时不得设为默认。

## 数据与提交边界

1. MoMA 规划对象、部件、图层、负例和完整旧 owner 集合。部件框是 Alpha 连通组件的种子，不是裁切刀。跨框组件保留整体或进入歧义；不膨胀笔画，不按颜色/距离直接合并语义。
2. Alpha、单层组合、各张 SAM mask 独立形成最多六个候选。保留 RGBA、mask 和装配记录。SAM 仅补充源图层现有像素；经过稳定登记的 Alpha 不被重分割覆盖。
3. 原生文字框内的源图文字颜色/抗锯齿证据用于保护。争议区域尝试本地 RapidOCR，禁止下载模型或运行全文 OCR；缺少 OCR 或原生字形证据时保持不确定并回退。白色有效内容及软 Alpha 不按背景删除。
4. 只有源图的精确不透明旧 crop 允许排除可由既有原生背景解释的 padding。真实透明旧图的白色像素仍有 owner。可暂存独立背景恢复：非对象像素逐字节保留，底层背景来自已有 LayerD 栈，或经几何与层序证明的既有原生页面／卡片内部；原生背景恢复和 LayerD 逆合成分别保留候选，前景软 Alpha 不变。白色有效内容可作为独立源图层装配假设，须单独通过白色归属复核。Go 必须额外检查实际删除图标后的放大预览，明确确认没有残留笔画、文字、邻居或色块才可登记。未知值不默认通过。恢复节点和图标作为一个场景事务提交，背景不随图标移动。新增节点只重排序号，保留既有非目标节点的相对层序、文字属性和位置。
5. 所有替换先进入独立场景。固定编译器使用已有逐页导出适配入口；PowerPoint 在副本中逐对象移动、缩放、删除、保存重开，检查其他对象属性及区域外像素。Go 复核实际 PowerPoint 整页与透明 PNG。像素、完整对象、替换、成品证据全部绑定后，才原子更新稳定指针。
6. 失败时尝试下一个完整候选，或保留完整父场景。编译父场景也失败时，不交付损坏文件。视觉失败或复核服务不可用时仍可交付安全 PPT 与预览，状态为 `REVIEW_REQUIRED`。不得将候选数或自动测试数称为提取成功率。

## 增量、预算与后台

普通背景恢复逐字节保留非对象像素。另有独立的整块旧 crop 归属假设：仅在整个旧矩形均有原生背景几何证据、无竞争图片或文字证据时生成，明确记录 `non_owned_pixels_unchanged=false` 和未确认像素数；除背景门禁外，Go 必须基于原图对每个旧矩形返回 `native_owner_scope=PASS`。缺失、未知或冲突均拒绝提交。私有复核图以红框标示旧 owner 边界，框外上下文中的标签和邻居必须保留，不能误判为残留。这些框不会进入交付 PPT。

上传队列将 `REVIEW_REQUIRED` 保留为终态并允许下载清单内、哈希正确的安全成品；页面明确显示待人工复核，不冒充自动验收通过。私有 12 页回归入口支持绑定历史最终场景的增量复用；相同模型、计划与实际预览的整页复核保留预留及原始回复，避免重复收费调用。

`object_registry.py` 将像素缓存和场景归属分开。缓存绑定源图、图层、部件/排除区域、算法和 SAM 模型/源码；改名及普通说明文案不使 Alpha 重提取。资产按内容哈希存储，文字修正通过私有资产引用复制 PNG 字节，禁止重新 crop。保护区、owner、层序改变仍须重验场景。

`object_scheduler.py` 将两次定向规划修正、两组 SAM 提示的预算绑定到源图与稳定对象 ID；重复定义不重新预留调用。保留中断预留记录，不默认为未收费。全页文字修正轮次不再决定对象阶段是否停用。

`services/segmentation/service.py` 提供认证的 `/v1/jobs` 提交、查询和绑定结果下载。GPU 任务串行；模型路径只能由服务器配置。缓存键不包含对象名/说明/旧 owner，保留原始 SAM 每张 mask。`single_frame.py --resume` 验证已有结果绑定再复用。

部署时先准备隔离 Python、固定上游源码与 SHA 校验后的 SAM 3 权重。`Install-PrivateService.ps1` 注册当前用户上下文的独立计划任务，只监听回环地址；`Start-PrivateTunnel.ps1` 建立本机回环 SSH 隧道。产品通过 `IMAGE2PPT_SAM_URL`、`IMAGE2PPT_SAM_TOKEN_FILE` 或项目 `.runtime/sam-client.json` 连接。MoMA/Go 仍使用既有内存密钥入口，不访问其他应用凭据库。服务令牌仅为本项目私有接口认证，不是模型供应商密钥。

## 可复现验证

```powershell
$env:PYTHONPATH='services/product;services/planner/src;services/vision/src'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest services/product/tests services/planner/tests services/segmentation/test_private_service.py -q
./image2ppt.ps1 test
```

`object_closure_run.py freeze/replay` 冻结私有真实材料并进行影子回归；`object_closure_model_run.py` 用同一后台阶段规划冻结材料，密钥通过 getpass 或标准环境注入。真实图、候选、回归参考、模型回复及详细问题只放忽略目录，不加入交付清单或 Git。源图手工描绘的语义参考必须标注边界不确定性，不能充当像素级 Alpha 真值。合成测试及 PowerPoint 自动编辑通过均不等于人工视觉验收。
