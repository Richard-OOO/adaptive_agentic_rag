"""agent 包的初始化，尽量避免在导入时触发对顶级包的绝对导入。

之前的实现使用 `from agent.graph import graph` 会在不同的导入路径下导致
ModuleNotFoundError("No module named 'agent'")。这里保持空的导出列表以避免
副作用导入；需要时可显式从 `src.agent.graph` 导入所需符号。
"""

__all__ = []
