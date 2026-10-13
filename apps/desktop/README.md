# Desktop

Electron 宿主与聊天界面。main 管理窗口和后端进程，preload 暴露限定接口，renderer 不直接拥有数据库或凭据。

从仓库根目录执行：

```powershell
npm --prefix apps/desktop ci
npm --prefix apps/desktop run start:fake
```

Python 环境须先按 [开发指南](../../docs/development.md) 准备。`start:fake` 使用合成回复，不调用真实模型。

`test` 运行公开的桌面回归，`lint` 检查语法，`build` 检查并暂存源码；这些命令不生成安装包。
