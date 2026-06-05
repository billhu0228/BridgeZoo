"""导出桥梁模型几何为 DXF，便于人工核对节点/单元布置。

参考已归档的 ``archive/fem_legacy/oneshot_opensees_ref.py`` 中 ``FEM.generate_dxf``。用法::

    python -m tools.export_dxf --n 6 --out model.dxf
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="导出模型 DXF")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--anchor-height", type=float, default=20.0)
    parser.add_argument("--out", type=str, default="model.dxf")
    args = parser.parse_args()

    # TODO: 由 BridgeGeometry 组装成桥模型并导出 DXF(args.out)。
    raise NotImplementedError("TODO: tools.export_dxf.main")


if __name__ == "__main__":
    main()
