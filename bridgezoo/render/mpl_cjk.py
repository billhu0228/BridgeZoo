"""matplotlib 中文字体配置工具。

matplotlib 默认 sans-serif 字体没有中文字形，画中文会显示成方框/缺字，且负号
``−`` 也会渲染异常。调用 :func:`use_cjk_font` 在画图前选用一款已安装的中文字体并
修正负号显示。

用法::

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bridgezoo.render.mpl_cjk import use_cjk_font
    use_cjk_font()      # 必须在创建 figure 前调用
    ...
"""

from __future__ import annotations

# 常见中文字体优先级（Windows / macOS / Linux 通吃）
_CANDIDATES = [
    "Microsoft YaHei",      # Windows 微软雅黑
    "SimHei",               # Windows 黑体
    "Microsoft JhengHei",   # Windows 繁体
    "SimSun",               # Windows 宋体
    "Noto Sans CJK SC",     # Linux/通用
    "Source Han Sans SC",   # 思源黑体
    "PingFang SC",          # macOS
    "Heiti SC",             # macOS
    "Arial Unicode MS",
    "WenQuanYi Zen Hei",    # Linux
]


def use_cjk_font(extra: list[str] | None = None) -> str | None:
    """选用一款已安装的中文字体并修正负号显示。

    Parameters
    ----------
    extra : list[str], optional
        额外的优先候选字体名（排在内置候选之前）。

    Returns
    -------
    str | None
        实际选中的字体名；若系统没有任何候选字体则返回 None（此时中文可能仍缺字，
        建议改用英文标签或安装中文字体）。
    """
    import matplotlib
    from matplotlib import font_manager

    candidates = (extra or []) + _CANDIDATES
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)

    if chosen:
        # 把选中的字体放在 sans-serif 列表最前
        current = matplotlib.rcParams.get("font.sans-serif", [])
        matplotlib.rcParams["font.sans-serif"] = [chosen] + [c for c in current if c != chosen]
        matplotlib.rcParams["font.family"] = "sans-serif"

    matplotlib.rcParams["axes.unicode_minus"] = False  # 用 ASCII 减号，避免负号缺字
    return chosen
