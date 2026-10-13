# 开发指南

## 环境

当前桌面入口的安装步骤适用于 Windows，使用 PowerShell 7、Git、Python 3.13、uv 0.12.9 和 Node.js 24。业务设计面向多端，平台能力集中在宿主适配层。

依赖版本由 `pyproject.toml`、`uv.lock` 和 `apps/desktop/package-lock.json` 固定。uv 安装说明见 [官方文档](https://docs.astral.sh/uv/getting-started/installation/)；确认 `uv --version` 与项目要求一致。

在 Git checkout 根目录执行：

```powershell
uv sync --locked
npm --prefix apps/desktop ci
npm --prefix apps/desktop run start:fake
```

演示使用合成回复，验证界面与本地消息流程，不代表真实模型效果。Python 子进程使用仓库内 `.venv`；首次安装依赖需要网络，演示本身不调用模型服务。

## 数据与真实模型

默认本地数据位于 `.local/`。运行数据、个人角色资料、记忆数据库和凭据不属于源码，不应提交。`.gitignore` 只是第一道保护，提交前仍需检查实际 diff。

真实模型运行与离线演示分开。代码中已有受管 provider 配置、凭据引用、计数器和请求预算接口；真实运行需要为所选数据目录配置完整依赖，不能只把 API Key 写进源码。首次公开版本的快速开始仅提供离线演示入口，不附带维护者的模型配置、凭据、权重或账本。

`configs/text-baseline-runtime.json` 是无私人路径的本地模型占位配置，其摘要由 `text/profile.py` 绑定。它不会下载模型，也不能作为真实模型制品清单；显式配置本地运行时需要实际文件及正确哈希。不要关闭校验来强行启动。

公共搜索与网页读取各自保留外部来源。外部服务的额度、可用性和条款由服务方决定，模型 API Key 不等于无限搜索授权。可选 SearXNG 是独立的 AGPL 服务，本仓库不分发其容器或代码。

## 验证

```powershell
pwsh -File scripts/check.ps1
```

也可分别执行：

```powershell
uv run --locked ruff check apps/backend/src tests scripts
uv run --locked --extra provider python -m pytest tests -q
npm --prefix apps/desktop test
npm --prefix apps/desktop run lint
npm --prefix apps/desktop run build
```

公开测试覆盖随仓库分发的产品行为，默认使用合成材料、fake 或本地回环。私人实验流水、供应商实模批次、个人工作区治理检查与已退役前身没有随仓库发布，也不属于这里的测试命令。

完整回归需要 `provider` extra 中的计数器库；安装库不会启用真实模型或要求凭据。检查脚本会按锁文件准备它。

`npm run build` 检查并暂存桌面源码，不生成安装包。真实窗口回归可以单独运行 `npm --prefix apps/desktop run e2e`，应先关闭其他本项目实例，避免本机端口冲突。默认测试不会替你开启真实模型或搜索服务。
