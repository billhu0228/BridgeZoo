"""Affine (linear-model) continuous tension optimization.

direct 后端(以及 OpenSees ``linear`` Truss 模式)是线性求解器,固定索股时终态索
应力、主梁线形误差、塔顶及全部塔锚水平位移都是预张力向量 T 的**仿射函数**::

    sigma(T) = sigma0 + M @ T   [MPa]   (M: MPa/N)
    err(T)   = err0   + D @ T   [m]     (D: m/N)
    ux(T)    = ux0    + q @ T   [m]     (q: m/N)
    ua(T)    = ua0    + Q @ T   [m]     (Q: m/N)

因此连续子问题先用 m+1 次真实 FEM 构造**精确**仿射模型,再在模型上(纯 numpy,
无 FEM 内层)分两相求解:

1. **LP 初值相**:最小化目标应力带最大偏离量 s,得到 s*(s*≈0 即该索股配置下
   [lower, upper] MPa 可达)与 warm-start 张力;应力带只作目标范围,不约束终解;
2. **二次相**:SLSQP + 解析梯度最小化完整目标(线形 + 塔顶水平位移 + 塔锚
   水平位移 RMSE + 均匀度 + hinge² 违反惩罚;索股项为常数故略去),
   最终应力可客观落在目标带外,越界代价仅由 hinge² 软惩罚计入。
   变量取缩放形式 x = T/hi ∈ [0,1](hi 为各索张拉上限):直接用牛顿量纲的
   T(~1e7 N)会令 SLSQP 的步长/收敛判据失效,首步即误判收敛并停在离最优
   数千倍处(回归见 tests/test_optim.py 缩放失速用例)。

LP 只提供目标应力带诊断和稳定初值;SLSQP 不对终态应力设硬边界。终点用真实
求解器复评并与模型交叉校核;若后端非线性(如 corot),校核失败并提示改用
``method="slsqp"``。单位:张力 N,应力 MPa,线形误差及塔节点位移 m。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from bridgezoo.optim.continuous import ContinuousOptimizationResult, ContinuousOptions
from bridgezoo.optim.evaluator import CableDesignEvaluator, EvaluationResult
from bridgezoo.optim.variables import validate_strand_vector, validate_tension_vector

# 终点模型-FEM 交叉校核容差:线性后端应在数值噪声内吻合,超过 1 MPa 说明
# 后端非线性或配置不一致,仿射模型的解不可信。
MODEL_MISMATCH_TOL_MPA = 1.0
MODEL_DISPLACEMENT_MISMATCH_TOL_M = 1.0e-6


@dataclass(frozen=True)
class AffineCableModel:
    """固定索股下应力、主梁误差和塔节点水平位移的精确仿射模型。"""

    cable_ids: tuple[int, ...]
    deck_nodes: tuple[int, ...]
    sigma0_mpa: np.ndarray        # (m,)
    m_mpa_per_n: np.ndarray       # (m, m)
    err0_m: np.ndarray            # (d,)
    d_m_per_n: np.ndarray         # (d, m)
    tower_dx0_m: float = 0.0
    tower_dx_m_per_n: np.ndarray | None = None  # (m,)
    anchor_nodes: tuple[int, ...] = ()
    anchor_dx0_m: np.ndarray | None = None       # (a,)
    anchor_dx_m_per_n: np.ndarray | None = None  # (a, m)

    def stress_mpa(self, tension_n: np.ndarray) -> np.ndarray:
        return self.sigma0_mpa + self.m_mpa_per_n @ tension_n

    def deck_err_m(self, tension_n: np.ndarray) -> np.ndarray:
        return self.err0_m + self.d_m_per_n @ tension_n

    def tower_dx_m(self, tension_n: np.ndarray) -> float:
        if self.tower_dx_m_per_n is None:
            return float(self.tower_dx0_m)
        return float(self.tower_dx0_m + self.tower_dx_m_per_n @ tension_n)

    def anchor_dx_m(self, tension_n: np.ndarray) -> np.ndarray:
        if self.anchor_dx0_m is None:
            return np.zeros(0)
        if self.anchor_dx_m_per_n is None:
            return self.anchor_dx0_m.copy()
        return self.anchor_dx0_m + self.anchor_dx_m_per_n @ tension_n


def build_affine_model(
    evaluator: CableDesignEvaluator,
    strands: np.ndarray,
    tension_step_n: float = 1.0e6,
) -> AffineCableModel:
    """用 m+1 个真实 FEM 工况(T=0 + 逐索单位扰动)构造仿射模型。

    对线性后端逐项精确(扰动差商即真实斜率);``tension_step_n`` 只影响数值
    条件,不影响线性模型的精确性。这 m+1 个工况索股相同(结构刚度不变),经
    :meth:`CableDesignEvaluator.evaluate_batch` 一次批量求解(direct 后端每施工阶段
    刚度只分解一次),与逐个 FEM 机器精度一致。
    """
    strands = validate_strand_vector(
        strands, evaluator.layout, evaluator.problem.bounds.strand_min, evaluator.problem.bounds.strand_max
    )
    ids = evaluator.layout.cable_ids
    m = evaluator.layout.size

    # 多右端:第 0 列 T=0,其余各列为单索单位扰动。
    tension_matrix = np.zeros((m, m + 1))
    tension_matrix[:, 1:] = np.eye(m) * tension_step_n
    cases = evaluator.evaluate_affine_batch(
        strands,
        tension_matrix,
        include_anchor_displacements=True,
    )

    base_err, base_sigma, tower_dx0, base_anchor_dx = cases[0]
    deck_nodes = tuple(base_err.keys())
    anchor_nodes = tuple(base_anchor_dx.keys())
    sigma0 = np.asarray([base_sigma[cid] for cid in ids], dtype=float)
    err0 = np.asarray([base_err[nid] for nid in deck_nodes], dtype=float)
    anchor_dx0 = np.asarray([base_anchor_dx[nid] for nid in anchor_nodes], dtype=float)

    M = np.zeros((m, m))
    D = np.zeros((len(deck_nodes), m))
    tower_dx_sensitivity = np.zeros(m)
    anchor_dx_sensitivity = np.zeros((len(anchor_nodes), m))
    for j in range(m):
        err_j, sigma_j, tower_dx_j, anchor_dx_j = cases[j + 1]
        sigma = np.asarray([sigma_j[cid] for cid in ids], dtype=float)
        err = np.asarray([err_j[nid] for nid in deck_nodes], dtype=float)
        anchor_dx = np.asarray([anchor_dx_j[nid] for nid in anchor_nodes], dtype=float)
        M[:, j] = (sigma - sigma0) / tension_step_n
        D[:, j] = (err - err0) / tension_step_n
        tower_dx_sensitivity[j] = (tower_dx_j - tower_dx0) / tension_step_n
        anchor_dx_sensitivity[:, j] = (anchor_dx - anchor_dx0) / tension_step_n

    return AffineCableModel(
        cable_ids=ids,
        deck_nodes=deck_nodes,
        sigma0_mpa=sigma0,
        m_mpa_per_n=M,
        err0_m=err0,
        d_m_per_n=D,
        tower_dx0_m=tower_dx0,
        tower_dx_m_per_n=tower_dx_sensitivity,
        anchor_nodes=anchor_nodes,
        anchor_dx0_m=anchor_dx0,
        anchor_dx_m_per_n=anchor_dx_sensitivity,
    )


class LinearTensionOptimizer:
    """固定索股的预张力优化(仿射模型 LP + 解析梯度 SLSQP)。

    与 :class:`bridgezoo.optim.continuous.FixedStrandTensionOptimizer` 同接口,
    供 :class:`bridgezoo.optim.hybrid.CableHybridOptimizer` 按
    ``ContinuousOptions.method`` 切换。
    """

    def __init__(
        self,
        evaluator: CableDesignEvaluator,
        options: ContinuousOptions | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        self.evaluator = evaluator
        self.options = options or ContinuousOptions()
        self.layout = evaluator.layout
        self.problem = evaluator.problem
        self.progress = progress

    def _emit(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def tension_upper_bounds(self, strands: np.ndarray) -> np.ndarray:
        """张拉(安装时刻)上限:tension_bound_stress × 索面积 [N]。"""
        area = self.problem.strand_area * strands.astype(float)
        return self.problem.bounds.tension_bound_stress_mpa * 1e6 * area

    # ------------------------------------------------------------- LP 相
    def _min_violation_lp(self, model: AffineCableModel, hi: np.ndarray) -> tuple[np.ndarray, float]:
        """min s  s.t.  lower − s ≤ sigma0 + M·T ≤ upper + s, 0 ≤ T ≤ hi, s ≥ 0。"""
        from scipy.optimize import linprog

        m = self.layout.size
        lower = self.problem.bounds.stress_lower_mpa
        upper = self.problem.bounds.stress_upper_mpa
        c = np.zeros(m + 1)
        c[-1] = 1.0
        ones = np.ones((m, 1))
        a_ub = np.block([[-model.m_mpa_per_n, -ones], [model.m_mpa_per_n, -ones]])
        b_ub = np.concatenate([model.sigma0_mpa - lower, upper - model.sigma0_mpa])
        bounds = [(0.0, float(h)) for h in hi] + [(0.0, None)]
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"feasibility LP failed: {res.message}")
        return np.asarray(res.x[:-1], dtype=float), float(res.x[-1])

    # ------------------------------------------------------------- 二次相
    def _model_objective_and_grad(self, model: AffineCableModel):
        """仿射模型上的连续目标及解析梯度。

        与 :func:`bridgezoo.optim.objectives.objective_breakdown` 一致(rmse² =
        mean(err²)、std² = var(σ)、rms² = mean(v²));索股项与 T 无关,略去。
        """
        w = self.problem.weights
        lower = self.problem.bounds.stress_lower_mpa
        upper = self.problem.bounds.stress_upper_mpa
        m = self.layout.size
        d = model.err0_m.size
        a = len(model.anchor_nodes)

        def value_and_grad(tension: np.ndarray) -> tuple[float, np.ndarray]:
            sigma = model.stress_mpa(tension)
            total = 0.0
            grad = np.zeros(m)
            if w.shape != 0.0 and d > 0:
                err = model.deck_err_m(tension)
                total += w.shape * np.mean(err * err) / w.shape_scale_m**2
                grad += w.shape * (2.0 / d) * (model.d_m_per_n.T @ err) / w.shape_scale_m**2
            if w.tower_displacement != 0.0:
                tower_dx = model.tower_dx_m(tension)
                tower_sensitivity = (
                    np.zeros(m)
                    if model.tower_dx_m_per_n is None
                    else model.tower_dx_m_per_n
                )
                total += w.tower_displacement * tower_dx**2 / w.shape_scale_m**2
                grad += (
                    w.tower_displacement
                    * 2.0
                    * tower_dx
                    * tower_sensitivity
                    / w.shape_scale_m**2
                )
            if w.tower_anchor_displacement != 0.0 and a > 0:
                anchor_dx = model.anchor_dx_m(tension)
                anchor_sensitivity = (
                    np.zeros((a, m))
                    if model.anchor_dx_m_per_n is None
                    else model.anchor_dx_m_per_n
                )
                total += (
                    w.tower_anchor_displacement
                    * np.mean(anchor_dx * anchor_dx)
                    / w.shape_scale_m**2
                )
                grad += (
                    w.tower_anchor_displacement
                    * (2.0 / a)
                    * (anchor_sensitivity.T @ anchor_dx)
                    / w.shape_scale_m**2
                )
            if w.stress_uniform != 0.0:
                centered = sigma - np.mean(sigma)
                total += w.stress_uniform * np.mean(centered * centered) / w.stress_scale_mpa**2
                grad += w.stress_uniform * (2.0 / m) * (model.m_mpa_per_n.T @ centered) / w.stress_scale_mpa**2
            if w.stress_violation != 0.0:
                low = np.maximum(0.0, lower - sigma)
                high = np.maximum(0.0, sigma - upper)
                violation = low + high
                total += w.stress_violation * np.mean(violation * violation) / w.stress_scale_mpa**2
                sign = np.where(high > 0.0, 1.0, 0.0) - np.where(low > 0.0, 1.0, 0.0)
                grad += (
                    w.stress_violation
                    * (2.0 / m)
                    * (model.m_mpa_per_n.T @ (violation * sign))
                    / w.stress_scale_mpa**2
                )
            return float(total), grad

        return value_and_grad

    # ------------------------------------------------------------- 主入口
    def optimize(self, strands, initial_pretension=None) -> ContinuousOptimizationResult:
        """两相求解;``initial_pretension`` 仅做合法性校验(模型上求解廉价且从
        LP warm-start 出发,无需外部初值)。"""
        try:
            from scipy.optimize import minimize
        except ImportError as exc:
            raise RuntimeError("scipy is required for cable pretension optimization") from exc

        strands = validate_strand_vector(
            strands, self.layout, self.problem.bounds.strand_min, self.problem.bounds.strand_max
        )
        if initial_pretension is not None:
            validate_tension_vector(initial_pretension, self.layout)
        hi = self.tension_upper_bounds(strands)

        model = build_affine_model(self.evaluator, strands)
        t_lp, s_star = self._min_violation_lp(model, hi)
        self._emit(
            f"    linear LP: s*={s_star:.3f} MPa "
            f"({'target band reachable' if s_star < 1e-6 else 'minimum target-band departure'})"
        )

        value_and_grad = self._model_objective_and_grad(model)

        # 二次相在缩放变量 x = T/hi ∈ [0,1] 上求解(见模块 docstring)。
        if np.any(hi <= 0.0):
            raise ValueError("tension upper bounds must be positive for all cables")

        def objective(x):
            return value_and_grad(x * hi)[0]

        def gradient(x):
            return value_and_grad(x * hi)[1] * hi

        res = minimize(
            objective,
            t_lp / hi,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * self.layout.size,
            options={"maxiter": max(200, self.options.maxiter), "ftol": min(self.options.ftol, 1.0e-12)},
        )
        x = np.clip(np.asarray(res.x, dtype=float), 0.0, 1.0) * hi

        evaluation = self.evaluator.evaluate(strands, x)
        sigma_fem = np.asarray([evaluation.cable_stress_mpa[cid] for cid in self.layout.cable_ids], dtype=float)
        mismatch = float(np.max(np.abs(model.stress_mpa(x) - sigma_fem))) if sigma_fem.size else 0.0
        displacement_mismatch = abs(model.tower_dx_m(x) - evaluation.metrics.tower_top_dx_m)
        anchor_dx_fem = np.asarray(
            [evaluation.tower_anchor_dx_m[nid] for nid in model.anchor_nodes],
            dtype=float,
        )
        anchor_displacement_mismatch = (
            float(np.max(np.abs(model.anchor_dx_m(x) - anchor_dx_fem)))
            if anchor_dx_fem.size
            else 0.0
        )
        if (
            mismatch > MODEL_MISMATCH_TOL_MPA
            or displacement_mismatch > MODEL_DISPLACEMENT_MISMATCH_TOL_M
            or anchor_displacement_mismatch > MODEL_DISPLACEMENT_MISMATCH_TOL_M
        ):
            raise RuntimeError(
                "affine model mismatch: "
                f"stress={mismatch:.3f} MPa (tol {MODEL_MISMATCH_TOL_MPA:.1f} MPa), "
                f"tower displacement={displacement_mismatch:.3e} m "
                f"and anchor displacement={anchor_displacement_mismatch:.3e} m "
                f"(tol {MODEL_DISPLACEMENT_MISMATCH_TOL_M:.1e} m); backend appears nonlinear; "
                "use ContinuousOptions(method='slsqp') instead"
            )
        self._emit(
            "    linear QP done: "
            f"success={bool(res.success)} objective={evaluation.objective:.6g} "
            f"shape_rmse={evaluation.metrics.shape_rmse_m * 1000.0:.3f} mm "
            f"anchor_dx_rmse={evaluation.metrics.tower_anchor_dx_rmse_m * 1000.0:.3f} mm "
            f"stress=[{evaluation.metrics.stress_min_mpa:.1f}, {evaluation.metrics.stress_max_mpa:.1f}] MPa "
            f"stress_violation_rms={evaluation.metrics.stress_violation_rms_mpa:.3f} MPa "
            f"model_mismatch={mismatch:.2e} MPa "
            f"tower_mismatch={displacement_mismatch:.2e} m"
            f" anchor_mismatch={anchor_displacement_mismatch:.2e} m"
        )
        return ContinuousOptimizationResult(
            evaluation=evaluation,
            success=bool(res.success),
            message=str(res.message),
            nfev=getattr(res, "nfev", None),
            feasibility_violation_mpa=s_star,
        )
