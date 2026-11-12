# OpenCode 自主工具运行验收

本轮状态为 `REVIEW_REQUIRED`。OpenCode Go 已接入主 Agent 和语义规划工具，实际模型为 `qwen3.8-flash`。最终视觉判断仍由当前 OpenCode Session 承担。各次运行均未调用独立 Go 复核工具。

## 已验证

- OpenCode 2.1.4 能通过 MCP ImageContent 读取图片。随机数字和颜色传输测试通过，测试数据为合成图。
- 原始第 2 页的一段正文经 Agent 自主选取、背景拟合、Mask 补全、字体修正和 Office 反馈修复，完成区域提交。移动、缩放、删除、短文本替换、等长替换与保存重开通过。
- 新第 6 页采用基础工具目录，Agent 自行更换目标和 Mask 策略。首个进入 Office 的候选出现等长替换溢出，随后依据实测 frame 反馈生成新候选，全部区域 Office 检查通过，事务范围外变化为 0。
- 测试结果为产品测试 325 项通过，固定核心测试 39 项通过，开发 Skill 同步检查通过。

## 仍待完成

第 2 页运行在继续扩展区域时耗尽预算。新第 6 页运行在第 32 轮工具调用结束后达到 Agent 步数上限，尚未提交已验证候选、执行最终整页 QA、自检和归档。其余真实任务也保持待验收，完整无人干预交付尚未通过。

开发者对照第 6 页原图及 Office 预览，观察到重建的页脚标题呈黑色且字重偏大，原图为蓝色。区域机械检查通过后的视觉修正仍有必要。保存的候选文件用于诊断。

Docker 沙箱的真实隔离与资源攻击测试、完整故障注入、双任务和多页组合验收仍未完成。当前服务宿主重启后需要重新注入模型凭据。

## 记录与产物

运行 ID、输入哈希、候选状态、Office 调用、模型消息、Token、事件耗时和产物哈希保存在同目录 `opencode-runtime.json`。完整本地收据位于 `outputs/opencode-go-final-closure/runtime-audit.json`，服务端原始证据保留在独立任务目录。

第 6 页最后一轮产生 4 个候选，执行 2 次 Office 区域检查，记录 32 条模型消息，输入 114603 Token、输出 7188 Token，工具事件跨度约 1289 秒。不同轮次采用不同图片、代码和预算，目前没有可比的提速结论。服务商报告的零成本缺少独立金额依据。

原始第 2 页区域检查文件保存在 `outputs/opencode-go-acceptance/original-page2/`。第 6 页已验证候选及预览保存在 `outputs/opencode-go-final-closure/new-page6/`。两者最终 Session QA 均为 `UNKNOWN`，自动发布保持关闭。
