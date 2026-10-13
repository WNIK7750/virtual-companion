# Backend

Python 后端负责对话编排、模型与工具适配、上下文、记忆和持久化。

安装与运行见根目录的 [开发指南](../../docs/development.md)。使用 uv 管理源码环境；此项目不是通过 `pip install .` 安装的 Python 包。

测试从仓库根目录执行 `uv run --locked --extra provider python -m pytest tests -q`。
