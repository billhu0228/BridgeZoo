from contextlib import contextmanager
import os

from ezdxf.math import Vec2
from openseespy.opensees import *


@contextmanager
def suppress_openseespy_output(enable):
    """
    上下文管理器，用于屏蔽 OpenSeesPy 的输出（stdout 和 stderr）。
    参数:
        enable (bool): 如果为 True，则屏蔽输出；如果为 False，则不屏蔽输出。
    """
    if enable:
        with open(os.devnull, 'w') as devnull:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                # 将标准输出和错误重定向到 /dev/null
                sys.stdout = devnull
                sys.stderr = devnull
                yield
            finally:
                # 恢复原来的标准输出和错误
                sys.stdout = original_stdout
                sys.stderr = original_stderr
    else:
        # 如果不屏蔽，直接执行代码块
        yield


def test2d_beam(x, y):
    with suppress_openseespy_output(True):
        wipe()
        model('basic', '-ndm', 2, '-ndf', 3)
        node(1, x, y)
        node(2, x, 0)
        node(3, x - 30, 0)
        node(4, x + 30, 0)
        # node(4, 0, y)
        fix(1, 1, 1, 1)  # 塔顶
        # fix(2, 1, 0, 1)  # 塔顶
        fix(3, 1, 1, 1)  # 承台
        fix(4, 1, 1, 1)  # 承台

        geomTransf('Linear', 1)
        element('CatenaryCable', 1, 1, 2, w3, E, A, 0.995 * y, alfa, cambiodetemp, rho, errorTol, NSubSteps, 0)
        element('elasticBeamColumn', 2, 3, 2, A, E, 1, 1)
        element('elasticBeamColumn', 3, 2, 4, A, E, 1, 1)
        timeSeries('Linear', 1)
        pattern('Plain', 2, 1)
        eleLoad('-ele', 1, '-type', '-beamUniform', 0., -1e-3)
        eleLoad('-ele', 2, 3, '-type', '-beamUniform', -1000.)
        system('FullGeneral')
        constraints('Plain')
        numberer('Plain')
        test('NormDispIncr', 1.0e-5, 100, 0)
        integrator('LoadControl', 1. / NSteps)
        algorithm("Newton")
        analysis("Static")
        analyze(NSteps)
        nd = nodeDisp(2)
        printModel()
    print(nd)


def test2d(x, y):
    with suppress_openseespy_output(True):
        wipe()
        model('basic', '-ndm', 2, '-ndf', 3)
        node(1, 0, 90)
        node(2, x, y)
        node(3, 90, 0)
        fix(1, 1, 1, 1)  # 塔顶
        fix(2, 0, 1, 0)  # 塔顶
        fix(3, 1, 1, 1)  # 承台
        element('CatenaryCable', 1, 1, 2, w3, E, A, L1, alfa, cambiodetemp, rho, errorTol, NSubSteps, 0)
        element('CatenaryCable', 2, 2, 3, w3, E, A, L2, alfa, cambiodetemp, rho, errorTol, NSubSteps, 0)
        timeSeries('Linear', 1)
        pattern('Plain', 2, 1)
        eleLoad('-ele', 1, 2, '-type', '-beamUniform', 0., -1)
        system('FullGeneral')
        constraints('Plain')
        numberer('Plain')
        test('NormDispIncr', 1.0e-5, 100, 0)
        integrator('LoadControl', 1. / NSteps)
        algorithm("Newton")
        analysis("Static")
        analyze(NSteps)
        nd = nodeDisp(2)
    print(nd)


def test3d(x, y):
    wipe()
    model('basic', '-ndm', 3, '-ndf', 3)
    node(1, 0, 0, 90)
    node(2, x, 0, y)
    node(3, 90, 0, 0)
    fix(1, 1, 1, 1)  # 塔顶
    fix(2, 0, 1, 1)  # 塔顶
    fix(3, 1, 1, 1)  # 承台
    element('CatenaryCable', 1, 1, 2, w3, E, A, L1, alfa, cambiodetemp, rho, errorTol, NSubSteps, 0)
    element('CatenaryCable', 2, 2, 3, w3, E, A, L2, alfa, cambiodetemp, rho, errorTol, NSubSteps, 0)
    timeSeries('Linear', 1)
    pattern('Plain', 2, 1)
    eleLoad('-ele', 1, 2, '-type', '-beamUniform', 0., 0., -1)
    system('FullGeneral')
    constraints('Plain')
    numberer('Plain')
    test('NormDispIncr', 1.0e-5, 100, 0)
    integrator('LoadControl', 1. / NSteps)
    algorithm("Newton")
    analysis("Static")
    # printModel()
    analyze(NSteps)
    nd = nodeDisp(2)
    print(nd)


if __name__ == '__main__':
    P1 = Vec2(0, 90)
    P3 = Vec2(90, 0)
    xx = 30
    yy = 40
    P2 = Vec2(xx, yy)
    w3 = -0.00001
    E = 3.e7
    A = 1.

    L1 = 1 * P1.distance(P2)
    L2 = 0.85 * P3.distance(P2)
    alfa = 6.5e-6
    cambiodetemp = 120.
    rho = w3 / 9.81
    errorTol = 1e-3
    NSubSteps = 50
    NSteps = 10

    test2d(xx, yy)
    test3d(xx, yy)
    test2d_beam(100, 100)
