"""可视化子包。

:mod:`bridgezoo.render.pygame_render` 提供施工过程（逐阶段拼装 + 两次张拉）与成桥
状态的 pygame 动画，以及线形/索力/股数的实时展示。可由环境的 ``render_mode='human'``
调用，也可独立用于回放已训练策略。
"""

__all__ = ["pygame_render"]
