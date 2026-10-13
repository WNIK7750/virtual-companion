"""Safe, user-facing admission failures; never include model response bodies."""


class ContextLimitError(ValueError):
    def __init__(self):
        super().__init__(
            "当前对话加上这条消息超出本地模型容量。请缩短输入或新建对话；原历史未删改。"
        )


class ContextCheckError(RuntimeError):
    def __init__(self):
        super().__init__("暂时无法检查模型容量，本条消息尚未发送。请检查模型状态后重试。")
