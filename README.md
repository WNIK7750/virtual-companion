<div align="center">

# Virtual Companion

**让不同的角色，都能自然地与你交流。**

A virtual companion built around natural conversation, character identity, and memory.

[快速开始](#快速开始) · [项目结构](#项目结构) · [开发文档](docs/README.md) · [参与贡献](CONTRIBUTING.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

> **项目目前处于开发阶段。**

## 关于项目

Virtual Companion 探索可以持续相处的虚拟伴侣：能够接话、表达判断、记住交流，也保留自己的角色特点。

项目将共通交流能力与角色设定分开。Agent 负责上下文、记忆、工具和回复交付；具体性格、背景、关系与表达方式来自角色预设，让不同角色拥有不同的交流体验。

## 关注的方向

- **自然交流**：根据情境参与对话，支持观点、分歧与双向互动。
- **角色与记忆**：将角色资料与有来源、可更正的记忆结合，保留人物差异。
- **消息体验**：语义分片流式输出、多气泡呈现，以及安静接收。
- **信息收集**：结合本地记忆、精确与语义查询、公开资料检索。
- **可扩展架构**：业务与宿主分离，为语音、角色形象和多端演进留出接口。

## 快速开始

当前源码启动说明适用于 **Windows / PowerShell**。需要 Git、Python **3.13**、uv **0.12.9** 和 Node.js **24**。

```powershell
git clone https://github.com/WNIK7750/virtual-companion.git
cd virtual-companion
uv sync --locked
npm --prefix apps/desktop ci
npm --prefix apps/desktop run start:fake
```

这会启动**离线演示**，使用合成回复，不需要 API Key，也不调用真实模型。演示数据保存在本机 `.local/` 中。

安装、运行和模型接入说明见 [开发指南](docs/development.md)。请勿将 API Key、聊天记录或个人记忆提交到仓库。

## 项目结构

```text
apps/
  backend/src/virtual_companion/  # 对话、记忆、工具与数据适配
  desktop/                       # Electron 宿主与聊天界面
packages/contracts/              # 跨模块数据契约
configs/                         # 无密钥配置与演示资源
tests/                           # 离线回归测试
scripts/                         # 开发检查入口
docs/                            # 架构与开发说明
```

架构与主要职责见 [架构说明](docs/architecture.md)。

## 参与贡献

欢迎提交问题、使用反馈和 Pull Request。请先阅读 [贡献指南](CONTRIBUTING.md)；安全问题请使用 [安全反馈渠道](SECURITY.md)。

采用 `main` 加短期功能分支的工作方式，具体约定见 [Git 与版本管理](docs/git-workflow.md)。

## 致谢与许可

研究和开发参考了 [pi](https://github.com/earendil-works/pi)、[Cyrene-Agent](https://github.com/Playa-0v0/Cyrene-Agent)、[LingChat](https://github.com/SlimeBoyOwO/LingChat)、[Herta](https://github.com/PersonaCLI/Herta) 和 [RAGFlow](https://github.com/infiniflow/ragflow) 等项目。

本项目自有代码使用 [MIT License](LICENSE)。实际适配代码、依赖及独立服务保留各自许可，详见 [第三方声明](THIRD_PARTY_NOTICES.md)。模型权重、声音和角色素材不随本仓库分发。
