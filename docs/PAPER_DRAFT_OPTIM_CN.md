# 基于可行性证书仿射代理的斜拉桥施工调索两层优化方法

> 中文核心期刊论文初稿  
> 作者：待补充  
> 单位：待补充  
> 基金项目：待补充  
> 中图分类号：待补充  
> 文献标志码：待补充  
> 初稿日期：2026-07-09  

## 摘要

斜拉桥悬臂施工过程中的索力确定与调索控制具有显著的路径相关性。索股数为离散整数变量，张拉力为连续变量，成桥线形、索应力安全区间、索力均匀性与用材经济性又相互耦合，因而直接形成混合整数非线性优化问题。针对传统影响矩阵法多基于完成态结构、难以显式处理整数索股与施工全过程路径效应，以及直接将有限元分析嵌入非线性优化时计算量大、易受不可行初值影响的问题，本文提出一种基于可行性证书仿射代理的斜拉桥施工调索两层优化方法。固定索股配置后，在线性小位移分阶段前进分析中，各施工阶段刚度矩阵与张拉力无关，张拉力仅以等效荷载形式进入右端项，因此成桥索应力与主梁线形误差均可表示为张拉力的精确仿射函数。本文以有限次真实前进分析标定该仿射映射，并在连续张力内层采用“线性规划可行性相 + 解析梯度二次目标相”的两相求解策略：第一相最小化应力带最大违反量，得到索股配置的可行性证书；第二相在不劣于该可行性下界的应力带约束内优化线形、应力均匀性与越界惩罚。外层则以整数索股为设计变量，结合应力比例整体缩放、应力引导坐标搜索和以可行性证书优先的字典序接受准则搜索索股配置。算例表明，在 6 阶段、12 根索的示例桥型中，本文方法可将初始方案的线形 RMSE 从 2060.02 mm 降至 55.59 mm，应力控制在 400 MPa 至 600 MPa 目标带内，最大违反量为 0，且线性规划证书给出应力带可达结论。该方法在保持有限元物理一致性的同时，将昂贵的重复有限元优化转化为低成本的仿射模型优化，并通过真实有限元复评对代理适用域进行自检，适用于线性小位移假定下的施工态快速调索与方案比选。

**关键词**：斜拉桥；施工控制；索力优化；调索；仿射代理模型；可行性证书；混合整数优化；前进分析

## English Abstract

**Title**: A Two-Level Cable Tuning Optimization Method for Cable-Stayed Bridge Construction Based on an Affine Surrogate with Feasibility Certificates

**Abstract**: Cable force determination during staged cantilever construction of cable-stayed bridges is path-dependent and involves both discrete strand numbers and continuous jacking forces. This paper proposes a two-level optimization method based on an exact affine surrogate and a stress-band feasibility certificate. For a fixed strand configuration, the stiffness matrices in linear small-displacement staged analysis are independent of jacking forces, while the jacking forces enter only through equivalent load vectors. Therefore, final cable stresses and deck profile errors are affine functions of the jacking force vector. The inner continuous problem is solved in two phases: a linear programming phase minimizes the maximum stress-band violation and provides a feasibility certificate, followed by a scaled SLSQP phase with analytical gradients to optimize deck profile, stress uniformity, and violation penalties. The outer integer layer searches strand configurations using stress-ratio resizing, stress-guided coordinate moves, and a lexicographic acceptance rule prioritizing the feasibility certificate over the weighted objective. A six-stage numerical example shows that the proposed method reduces deck-profile RMSE from 2060.02 mm to 55.59 mm while keeping all cable stresses within the target band of 400-600 MPa and reducing the total strand count from 2400 to 1599. The method provides an efficient, interpretable, and self-checking optimization kernel for construction-stage cable tuning under the linear small-displacement assumption.

**Key words**: cable-stayed bridge; construction control; cable force optimization; affine surrogate model; feasibility certificate; mixed-integer optimization; staged analysis

## 1 引言

斜拉桥施工控制的核心任务之一是在分阶段架设过程中确定合理索力，使结构在成桥状态下同时满足线形、内力、安全储备与经济性要求。对于悬臂拼装体系，梁段、拉索和边界条件随施工阶段逐步激活，结构刚度和荷载路径持续变化；某一阶段的张拉决策不仅影响当前线形，也会随之后的梁段安装、合龙和二次调索累积到成桥终态。因此，索力计算不是单一完成态结构上的静力修正问题，而是一个具有施工路径记忆的全过程优化问题。

工程实践中常用方法包括经验试调、基于影响矩阵的最小二乘修正、智能优化算法以及将有限元模型直接嵌入优化器的数值优化方法。经验试调依赖工程师经验，适合局部修正，但难以在多目标、多约束和多阶段耦合下系统搜索。影响矩阵法具有清晰的力学意义和较高的计算效率，但其常见形式多基于完成态或局部线性化结构，对整数股数、张拉上限和全过程路径相关性处理不足。遗传算法、粒子群、CMA-ES 等随机搜索方法对非凸性和离散变量较友好，但需要大量有限元样本，难以在阶段数较多时保持工程可用的计算效率。直接把分阶段有限元分析嵌入 SLSQP 等非线性优化器虽然形式统一，但每次目标和约束评估都需调用真实有限元，且当初始点不满足应力带约束时，硬约束求解容易陷入不可行或“原地不动”的数值失效。

本文面向“索股数 + 预张力”的协同调索问题，提出一种两层优化框架。外层搜索整数索股配置，内层在固定索股下求解连续预张力。本文的关键不在于简单套用分层优化，而在于证明并利用了线性小位移分阶段前进分析中的精确仿射结构：固定索股后，张拉力对成桥应力与线形的影响可由有限次前进分析一次性标定，随后连续优化不再反复调用有限元。进一步地，内层先用线性规划给出应力带可行性证书，再在该可行性下界内优化线形和应力分布；外层则把该证书作为优先准则，以避免只追求加权目标而误杀应力带更可达的股数方案。

本文主要贡献如下。

1. 建立了贯穿斜拉桥分阶段施工全过程的张拉力至成桥响应精确仿射代理模型，区别于仅在完成态结构上建立影响矩阵的常规做法。
2. 提出“LP 可行性相 + 解析梯度二次目标相”的连续张力求解策略，使固定索股子问题同时具备可解释的应力带可行性证书和稳定的数值优化过程。
3. 构建索股数外层搜索与连续张力内层优化的两层框架，引入应力比例整体缩放、应力引导坐标搜索和可行性优先接受准则。
4. 在代理优化终点引入真实有限元复评与偏差校核，使代理模型具有适用域自检能力。
5. 通过可复现算例验证该方法在应力带控制、线形改善和求解效率方面的有效性，并明确其线性小位移适用边界。

## 2 分阶段施工力学模型

### 2.1 基本假定与符号

本文考虑二维斜拉桥简化模型。主梁采用 Euler-Bernoulli 梁单元，拉索采用仅承受轴力的杆索单元。结构材料在线弹性范围内工作，直接求解后端采用小位移假定。设施工阶段数为 \(n\)，每阶段张拉一对对称或非对称拉索，则拉索总数为

\[
m=2n .
\]

对第 \(i\) 根拉索，整数索股数记为 \(z_i\)，单股面积为 \(a_s\)，则拉索面积为

\[
A_i=a_s z_i .
\]

张拉力向量记为

\[
\mathbf{T}=[T_1,T_2,\ldots,T_m]^T ,
\]

其中 \(T_i\ge 0\)。索股数向量记为

\[
\mathbf{z}=[z_1,z_2,\ldots,z_m]^T,\quad z_i\in \mathbb{Z}.
\]

实际实现中，变量按施工阶段优先排序，即“右 1、左 1、右 2、左 2、……”，与分阶段建模器的输入顺序一致。

### 2.2 前进分析过程

分阶段施工采用正向前进分析。第 \(k\) 阶段激活对应梁段和拉索，装配当前阶段整体刚度矩阵 \(\mathbf{K}_k\)，施加新增自重、二期恒载以及拉索预张力等效荷载，求解增量位移

\[
\mathbf{K}_k \Delta \mathbf{u}_k=\Delta \mathbf{F}_k .
\]

随后将增量位移累加到结构总位移，并据此恢复拉索轴力和应力。由于施工过程中梁段和索逐步激活，\(\mathbf{K}_k\) 随阶段变化；又由于索股数改变会改变索面积和轴向刚度，\(\mathbf{K}_k\) 也随 \(\mathbf{z}\) 变化。因此，单一完成态影响矩阵不足以完整描述施工全过程。

本文方法在固定索股配置 \(\mathbf{z}\) 后构造张拉力代理模型。此时每根索的 \(EA\) 固定，各阶段刚度矩阵 \(\mathbf{K}_k(\mathbf{z})\) 固定，张拉力 \(\mathbf{T}\) 只通过等效荷载项进入 \(\Delta \mathbf{F}_k\)。这是后续仿射代理成立的力学基础。

### 2.3 双后端校核思想

算法原型采用自研直接刚度法作为快速内核，同时保留 OpenSees 后端用于独立校核。两者共享同一几何、单元编号、施工阶段定义与变量排序。对于线性 Truss 后端，理论上两套求解器应在浮点误差范围内一致；对于几何非线性的 corotTruss 后端，差异随挠度增大而增加，此时仿射代理仅可视为局部近似，应切换到真实有限元在环的非线性优化路径或使用代理校核机制拒绝不可信结果。

在本文本机验证中，`scripts.validate_staged --n 6 --cable-element linear` 通过直接刚度法与 OpenSees 对比。阶段远端竖向位移最大差异为 15.406 mm，相对差异 1.132%，小于验证脚本设定的 2.5% 容许阈值；合龙锁定位移残差为 0，说明分阶段建模和边界处理在该验证算例下保持一致。该结果用于证明求解流程与阶段定义可靠，但正式投稿仍应补充与主优化算例一致的后端校核表。

## 3 精确仿射代理模型

### 3.1 仿射性推导

固定索股配置 \(\mathbf{z}\) 后，第 \(k\) 阶段的刚度矩阵 \(\mathbf{K}_k(\mathbf{z})\) 与预张力 \(\mathbf{T}\) 无关。张拉力只进入右端荷载向量，可写为

\[
\Delta \mathbf{F}_k(\mathbf{T})=\Delta \mathbf{F}_{k,0}+\mathbf{B}_k\mathbf{T}.
\]

于是该阶段增量位移为

\[
\Delta \mathbf{u}_k(\mathbf{T})
=\mathbf{K}_k^{-1}\Delta \mathbf{F}_{k,0}
+\mathbf{K}_k^{-1}\mathbf{B}_k\mathbf{T}.
\]

总位移为各阶段增量的累加，仍为 \(\mathbf{T}\) 的仿射函数。成桥索应力由终态位移、索初始几何和预张力共同恢复；在线性小位移杆索模型下，该恢复过程也是线性算子与常数项的组合。因此成桥拉索应力和主梁线形误差可表示为

\[
\boldsymbol{\sigma}(\mathbf{T})=\boldsymbol{\sigma}_0+\mathbf{M}\mathbf{T},
\]

\[
\mathbf{e}(\mathbf{T})=\mathbf{e}_0+\mathbf{D}\mathbf{T},
\]

其中 \(\boldsymbol{\sigma}\in\mathbb{R}^{m}\) 为拉索终态应力，单位 MPa；\(\mathbf{e}\in\mathbb{R}^{d}\) 为主梁控制节点竖向位移相对目标线形的误差，单位 m；\(\mathbf{M}\in\mathbb{R}^{m\times m}\)，单位 MPa/N；\(\mathbf{D}\in\mathbb{R}^{d\times m}\)，单位 m/N。

上述仿射性不是经验拟合，也不是响应面近似，而是线性小位移前进分析的直接结果。它的新价值在于将施工全过程路径相关的有限元映射压缩为可优化、可解释、可校核的显式代理。

### 3.2 标定方法

设单位扰动张力为 \(\Delta T\)。先在 \(\mathbf{T}=\mathbf{0}\) 下运行一次完整前进分析，得到 \(\boldsymbol{\sigma}_0\) 与 \(\mathbf{e}_0\)。然后对每根索分别施加单位扰动，即第 \(j\) 个工况取

\[
\mathbf{T}^{(j)}=\Delta T\mathbf{e}_j ,
\]

其中 \(\mathbf{e}_j\) 为第 \(j\) 个标准基向量。运行前进分析后得到 \(\boldsymbol{\sigma}^{(j)}\) 和 \(\mathbf{e}^{(j)}\)，差商为

\[
\mathbf{M}_{:,j}=\frac{\boldsymbol{\sigma}^{(j)}-\boldsymbol{\sigma}_0}{\Delta T},
\quad
\mathbf{D}_{:,j}=\frac{\mathbf{e}^{(j)}-\mathbf{e}_0}{\Delta T}.
\]

因此，标定一个固定索股配置仅需 \(m+1\) 次前进分析。在线性后端下，差商即精确斜率，\(\Delta T\) 只影响数值条件。项目测试 `test_affine_model_matches_fem` 使用随机张力向量验证代理与真实有限元复评的一致性，应力最大误差小于 \(10^{-6}\) MPa，线形误差小于 \(10^{-9}\) m。

### 3.3 同结构多右端加速

\(m+1\) 个标定工况具有相同索股配置，因此每个施工阶段的整体刚度矩阵完全相同，仅右端荷载不同。计算上可将各工况组织为同矩阵多右端问题：每阶段只装配并分解一次 \(\mathbf{K}_k\)，随后对多个右端一次性回代。该设计把逐工况重复分解转化为批量求解，显著降低代理标定开销。

根据项目历史基准，在 \(n=19\)、\(m=38\) 的标定任务中，批量多右端实现将标定耗时由 4730 ms 降至 194 ms，约提速 24.4 倍；同时与逐列标定在浮点噪声范围内一致。这一加速不是改变优化模型，而是去除线性代数层面的冗余，是两层优化能够在工程规模下实用化的重要支撑。

### 3.4 代理监理校核

仿射代理的精确性依赖线性小位移假定和固定索股下刚度与张拉力无关这一条件。为避免误用，本文在内层优化结束后使用真实有限元对最优张力进行复评，并比较代理预测应力与真实有限元应力的最大偏差：

\[
\delta_{\sigma}=\max_i \left|\sigma_i^{\text{proxy}}-\sigma_i^{\text{FEM}}\right| .
\]

当 \(\delta_{\sigma}\) 超过给定阈值时，说明后端可能存在几何非线性或模型配置不一致，当前代理解不可信，应拒绝结果并提示切换到真实有限元在环的 SLSQP 路径。本文原型采用 1 MPa 作为线性后端校核阈值。该机制使代理模型不仅用于加速，也承担适用域自检功能。

## 4 两层优化方法

### 4.1 问题形式化

索力优化的设计变量包括整数索股数 \(\mathbf{z}\) 和连续张拉力 \(\mathbf{T}\)。索股数满足

\[
z_i^{\min}\le z_i\le z_i^{\max},\quad z_i\in\mathbb{Z},
\]

张拉力满足

\[
0\le T_i\le T_i^{\max}= \sigma_T^{\max} a_s z_i ,
\]

其中 \(\sigma_T^{\max}\) 为张拉控制应力上限。

目标函数综合线形、用材、应力均匀性和应力越界惩罚：

\[
J(\mathbf{z},\mathbf{T})
=w_e\left(\frac{\operatorname{RMSE}(\mathbf{e})}{e_s}\right)^2
+w_z\frac{\sum_i z_i}{z_s}
+w_u\left(\frac{\operatorname{Std}(\boldsymbol{\sigma})}{\sigma_s}\right)^2
+w_v\left(\frac{\operatorname{RMS}(\mathbf{v})}{\sigma_s}\right)^2 .
\]

其中 \(e_s\)、\(z_s\)、\(\sigma_s\) 为尺度参数，越界量 \(v_i\) 定义为

\[
v_i=\max(0,\sigma_i^{\min}-\sigma_i)+\max(0,\sigma_i-\sigma_i^{\max}).
\]

在本文主算例中，权重取 \(w_e=1.0\)、\(w_z=0.02\)、\(w_u=0.2\)、\(w_v=100.0\)，线形尺度取 100 mm，应力尺度取 100 MPa，股数尺度取 8200。应力越界权重显著高于其他项，使应力带满足在加权目标中占主导地位；同时，后文的 LP 可行性证书进一步将“能否进入应力带”从普通软惩罚提升为搜索接受准则中的第一优先级。

### 4.2 内层：连续张力两相优化

固定 \(\mathbf{z}\) 后，首先构造仿射代理

\[
\boldsymbol{\sigma}(\mathbf{T})=\boldsymbol{\sigma}_0+\mathbf{M}\mathbf{T},
\quad
\mathbf{e}(\mathbf{T})=\mathbf{e}_0+\mathbf{D}\mathbf{T}.
\]

第一相为线性规划可行性相：

\[
\begin{aligned}
\min_{\mathbf{T},s}\quad & s \\
\text{s.t.}\quad
& \boldsymbol{\sigma}^{\min}-s\mathbf{1}
\le
\boldsymbol{\sigma}_0+\mathbf{M}\mathbf{T}
\le
\boldsymbol{\sigma}^{\max}+s\mathbf{1}, \\
& 0\le \mathbf{T}\le \mathbf{T}^{\max}, \\
& s\ge 0 .
\end{aligned}
\]

最优值 \(s^\*\) 是该索股配置下应力带的最小可达最大违反量。当 \(s^\*\approx 0\) 时，应力带可达；当 \(s^\*>0\) 时，无论如何张拉，该索股配置都至少存在 \(s^\*\) MPa 的最大带宽违反。该值是一个可解释的结构可行性证书。

第二相以第一相结果为起点，在不劣于 \(s^\*\) 的应力带放宽内最小化完整连续目标。由于 \(\boldsymbol{\sigma}\) 与 \(\mathbf{e}\) 都是 \(\mathbf{T}\) 的仿射函数，目标函数关于 \(\mathbf{T}\) 的梯度可解析计算。例如线形项梯度为

\[
\nabla_{\mathbf{T}}
\left[
\frac{1}{d}\mathbf{e}^T\mathbf{e}
\right]
=\frac{2}{d}\mathbf{D}^T\mathbf{e}.
\]

应力均匀项和 hinge 越界惩罚项同理可由 \(\mathbf{M}\) 链式求导得到。实际求解时采用缩放变量

\[
\mathbf{x}=\mathbf{T}/\mathbf{T}^{\max},\quad 0\le x_i\le 1 ,
\]

避免直接以 \(10^7\) N 量级张力进入 SLSQP 导致步长和收敛判据失真。缩放后，目标梯度和应力约束雅可比同步乘以 \(\mathbf{T}^{\max}\)，保证数值尺度合理。

### 4.3 外层：整数索股搜索

外层搜索以索股数 \(\mathbf{z}\) 为变量。每个候选 \(\mathbf{z}\) 调用内层两相优化，返回最优张拉力、加权目标值和可行性证书 \(s^\*\)。搜索由三部分组成。

1. 初始解与随机试探。若用户未指定，默认取统一初始股数；也可进行随机重启以探索其他股数组合。
2. 应力比例整体缩放。根据当前最优方案的索应力，按

\[
z_i^{\text{new}}\approx
\operatorname{round}
\left(
z_i^{\text{old}}
\frac{\sigma_i}{\sigma^{\text{tar}}}
\right)
\]

一次性调整索股数，其中 \(\sigma^{\text{tar}}\) 默认取应力带中值。应力偏高的索增加股数，应力偏低的索减少股数，松弛或受压索取最小股数。该步骤用于快速把索股配置推至合理量级。
3. 应力引导坐标搜索。逐根索尝试 \(+1\) 或 \(-1\) 股调整。若当前索应力低于下限，则优先减股以提高应力；若高于上限，则优先增股以降低应力；若在带内，则优先减股以节省材料。搜索采用 first-improvement 策略，即一旦某候选被接受，立即更新当前最优解。

改股数时，旧张拉力按股数比例缩放作为内层 warm-start：

\[
T_i^{\text{new}}=T_i^{\text{old}}\frac{z_i^{\text{new}}}{z_i^{\text{old}}}.
\]

虽然线性内层最终主要依赖 LP 起点，该 warm-start 仍有助于兼容 SLSQP 后端和保持跨层物理连续性。

### 4.4 可行性优先接受准则

旧的整数搜索只比较加权目标 \(J\)。然而对于调索问题，应力带可达性更接近硬性工程要求，而非普通偏好项。若某个候选显著降低 \(s^\*\)，即使暂时牺牲线形或用材目标，也应优先接受。为此本文采用字典序接受准则：

1. 若候选 \(s^\*\) 比当前最优 \(s^\*\) 至少降低给定容差，则接受。
2. 若候选 \(s^\*\) 比当前最优 \(s^\*\) 至少升高给定容差，则拒绝。
3. 若二者 \(s^\*\) 持平，则比较加权目标 \(J\)，仅接受目标改善的候选。
4. 若使用非线性 SLSQP 后端且没有 \(s^\*\)，则退回只比较加权目标。

该准则避免了“应力带可达性明显改善，但加权目标短期变差”时被误拒绝的问题，使外层搜索更符合工程调索的优先级。

### 4.5 算法流程

算法 1 给出本文两层优化方法的整体流程。

```text
算法 1 基于可行性证书仿射代理的斜拉桥调索两层优化

输入：桥梁几何与施工阶段计划，索股边界，张拉上限，应力带，
      目标函数权重，初始索股 z0
输出：最优索股 z*, 最优张拉力 T*, 成桥线形与索应力指标

1  z_best ← z0
2  对 z_best 构造仿射代理 σ(T), e(T)
3  求解 LP 可行性相，得到 s_best 和 LP 起点
4  求解二次目标相，得到 T_best 和 J_best
5  用真实 FEM 复评并校核代理误差
6  for 外层迭代 r = 1,2,...,R do
7      根据当前应力执行整体 resize，生成候选 z_c
8      若 z_c 有效，则调用内层两相优化得到 s_c, J_c, T_c
9      按字典序准则比较 (s_c, J_c) 与 (s_best, J_best)
10     若接受，则更新 z_best, T_best, s_best, J_best
11     for 每根索 i do
12         根据当前应力决定 +step 与 -step 的尝试顺序
13         生成单坐标候选 z_c
14         调用内层两相优化得到 s_c, J_c, T_c
15         按字典序准则决定接受或拒绝
16         若接受，则更新当前最优并转至下一根索
17     end for
18     若本轮无候选被接受，则提前停止
19  end for
20  对最终设计执行真实 FEM 复评并输出指标
```

## 5 算例与结果

### 5.1 算例设置

本文初稿采用项目内置二维分阶段斜拉桥算例进行方法验证。主算例取施工阶段数 \(n=6\)，拉索数 \(m=12\)。优化命令显式指定参数如下：

```bash
py -3.12 -m scripts.optimize_cables --n 6 --outer-iterations 3 \
    --strand-min 100 --strand-max 500 --initial-strands 200 \
    --stress-lower 400 --stress-upper 600 \
    --quiet --out results/cable_opt_paper_draft_main
```

该命令采用 direct 线性后端、仿射代理连续优化路径、应力比例 resize、应力引导坐标搜索与可行性优先接受准则。应力带取 400 MPa 至 600 MPa，索股边界取 100 至 500，初始每根索取 200 股，外层迭代上限为 3 轮。

### 5.2 优化前后指标

表 1 给出初始方案与优化后方案的主要指标。初始方案对应评估历史第 0 项，优化后方案对应最终输出。

| 指标 | 初始方案 | 优化后方案 |
|---|---:|---:|
| 目标函数 \(J\) | 425.5636 | 0.3983 |
| 线形 RMSE / mm | 2060.0220 | 55.5886 |
| 最大线形误差 / mm | 3477.9757 | 104.6734 |
| 总股数 | 2400 | 1599 |
| 索应力均值 / MPa | 449.736 | 498.629 |
| 索应力标准差 / MPa | 93.978 | 65.335 |
| 索应力最小值 / MPa | 388.972 | 400.000 |
| 索应力最大值 / MPa | 611.028 | 600.000 |
| 应力越界 RMS / MPa | 10.060 | 0.000 |
| LP 可行性证书 \(s^\*\) / MPa | 未记录 | 0.000 |

可以看出，初始方案已有较合理的平均应力，但存在上下限越界，且线形误差很大。优化后，应力最小值和最大值分别贴近目标带下限和上限，最大违反量为 0；线形 RMSE 降至 55.59 mm，总股数由 2400 降至 1599。该结果说明，本文方法并非单纯通过增加材料来满足应力约束，而是在满足应力带的同时显著改善线形并降低总股数。

### 5.3 收敛过程

本次优化共产生 49 次历史评估。初始评估的目标函数为 425.56。经整体 resize 后，目标迅速降至 58.77，且应力带已基本可达；随后坐标搜索继续调整局部索股，使目标进一步降至 0.398。最终三条相邻历史记录显示，目标在总股数 1597 至 1600 附近小幅变化，最优点位于总股数 1599，对应更小线形 RMSE 和更低综合目标。

从工程角度看，resize 步骤承担“把解从不合理量级拉回合理量级”的任务，坐标搜索承担“在合理量级附近细调”的任务。LP 证书 \(s^\*=0\) 则说明最终索股配置下，存在张拉力可使所有索应力进入 400 MPa 至 600 MPa 目标带。

### 5.4 可行性优先准则对照

为考察可行性优先接受准则的影响，本文在相同参数下关闭 `band_priority` 进行对照：

```bash
py -3.12 -m scripts.optimize_cables --n 6 --outer-iterations 3 \
    --strand-min 100 --strand-max 500 --initial-strands 200 \
    --stress-lower 400 --stress-upper 600 \
    --no-band-priority --quiet \
    --out results/cable_opt_paper_draft_no_band_priority
```

在该算例中，关闭可行性优先准则后最终指标与默认方法相同。这说明主算例的搜索路径并未触发“可行性改善但加权目标暂时变差”的分歧。该结果应如实报告：可行性优先准则是鲁棒性机制，其价值在于避免特定参数和搜索阶段下的误拒绝，而非保证每个算例都改变最终解。正式投稿前应补充窄应力带、较小索股上限或不良初值下的消融实验，以突出该机制的必要性。

### 5.5 与直接有限元在环优化的差异

直接有限元在环 SLSQP 的优势是对非线性后端更通用，但其目标和约束评估成本高，且需要在不可行初始点附近处理非线性约束。本文方法将固定索股下的连续优化转化为显式仿射模型上的 LP 与解析梯度优化，具有三个直接收益。

1. 可行性先验明确。LP 给出 \(s^\*\)，能回答“这套股数是否可能满足应力带”。
2. 内层优化稳定。SLSQP 不再反复调用有限元，也不再从可能严重不可行的硬约束问题直接起步。
3. 外层搜索可解释。股数调整不只看加权目标，还能根据 \(s^\*\) 判断结构可行性是否改善。

上述收益解释了为什么该算法能提高调索效果和调索水平：它把“试调”转化为有可行性证书、有灵敏度模型、有真实有限元校核的系统搜索过程。

## 6 讨论

### 6.1 工程解释性

调索的工程目标并非单一最小化线形误差。若只追求线形，可能出现局部拉索过张、松弛或应力分布不均；若只追求应力带，则可能牺牲成桥线形；若只追求省料，则可能使索应力靠近上限。本文目标函数和两层策略将这些目标显式拆分并重新组合：应力带由 LP 证书和高权重越界惩罚共同保障，线形由 RMSE 项控制，材料用量由总股数项约束，应力均匀性由标准差项调节。

最终算例中，优化后应力均值为 498.63 MPa，接近 400 MPa 至 600 MPa 带宽中值，标准差由 93.98 MPa 降至 65.33 MPa；这说明优化不是简单把全部索推向同一边界，而是在应力带内形成相对均衡的分布。总股数减少同时线形显著改善，说明股数搜索和张力优化之间形成了有效协同。

### 6.2 适用边界

本文方法的精确性依赖线性小位移假定。若结构存在显著几何非线性、索垂度效应、塔梁强耦合非线性、索鞍摩阻或非线性材料行为，则 \(\boldsymbol{\sigma}(\mathbf{T})\) 与 \(\mathbf{e}(\mathbf{T})\) 不再严格仿射。此时可采用两种处理方式：第一，在每次代理优化终点使用真实有限元复评并以偏差阈值拒绝不可信代理解；第二，切换到真实有限元在环的非线性优化路径，将本文方法得到的线性解作为初值。

此外，本文初稿采用二维模型和确定性荷载，尚未包含温度、收缩徐变、施工误差、测量噪声、索力识别误差以及施工设备张拉精度等实际不确定性。面向工程应用时，应进一步扩展为鲁棒优化或可靠度约束优化。

### 6.3 与传统影响矩阵法的关系

本文方法可视为影响矩阵思想在分阶段施工全过程中的扩展，但二者并不等同。传统影响矩阵多描述完成态结构对局部调整的线性响应；本文的 \(\mathbf{M}\) 与 \(\mathbf{D}\) 则由完整前进分析标定，包含阶段激活、边界锁定和后续施工累积效应。更重要的是，本文把影响矩阵从“线性修正工具”推进为“内层优化模型”，与 LP 可行性相、解析梯度优化、整数股数外层搜索和真实有限元监理共同构成完整算法。

### 6.4 投稿前需要补强的实验

本文初稿已经给出主算例和方法闭环，但若面向中文核心期刊投稿，还应补充以下内容。

1. 增加不同阶段数 \(n=4,6,8,12\) 的规模扩展实验，报告计算时间、评估次数和最终指标。
2. 增加与传统完成态影响矩阵最小二乘、遗传算法或粒子群、直接 FEM-in-SLSQP 的对比。
3. 增加消融实验，包括关闭 resize、关闭应力引导、关闭可行性优先准则、逐列标定与批量标定对比。
4. 增加 OpenSees 与 direct 在主优化结果上的复核表，分别报告线形 RMSE、应力最大偏差和阶段响应差异。
5. 补充近五年中文桥梁施工控制与索力优化文献，避免参考文献过度偏向软件和数值优化基础文献。

## 7 结论

本文针对斜拉桥分阶段施工调索中的索股数与预张力协同优化问题，提出了基于可行性证书仿射代理的两层优化方法，得到以下结论。

1. 在线性小位移分阶段前进分析中，固定索股配置后，成桥索应力和主梁线形误差均可表示为张拉力的精确仿射函数。该性质使连续张力优化能够从反复有限元调用转化为显式代理模型上的低成本求解。
2. LP 可行性相给出的 \(s^\*\) 可作为索股配置能否满足应力带的证书。与仅使用加权越界惩罚相比，该证书更符合工程调索中“先保证应力带可达，再优化线形和经济性”的决策逻辑。
3. 采用缩放变量的解析梯度 SLSQP 能避免张力数量级过大导致的数值失速，使连续子问题稳定贴近仿射模型最优解。
4. 外层索股搜索通过应力比例整体缩放快速进入合理股数量级，再通过应力引导坐标搜索精修；可行性优先接受准则进一步提高了搜索的工程合理性。
5. 在 6 阶段示例桥型中，本文方法将线形 RMSE 从 2060.02 mm 降至 55.59 mm，应力完全进入 400 MPa 至 600 MPa 目标带，总股数由 2400 降至 1599，验证了方法在该算例下的有效性。

本文方法适合用于线性小位移假定下的快速施工调索、参数敏感性分析和方案比选。对于显著几何非线性或实桥施工误差场景，应结合真实有限元复评、非线性后端和监测数据同化进一步扩展。

## 参考文献

[1] Gimsing N J, Georgakis C T. Cable Supported Bridges: Concept and Design[M]. 3rd ed. Chichester: John Wiley & Sons, 2012.

[2] Troitsky M S. Cable-Stayed Bridges: Theory and Design[M]. 2nd ed. London: Crosby Lockwood Staples, 1988.

[3] Nocedal J, Wright S J. Numerical Optimization[M]. 2nd ed. New York: Springer, 2006.

[4] Kraft D. A Software Package for Sequential Quadratic Programming[R]. DFVLR-FB 88-28, 1988.

[5] Virtanen P, Gommers R, Oliphant T E, et al. SciPy 1.0: fundamental algorithms for scientific computing in Python[J]. Nature Methods, 2020, 17: 261-272.

[6] Harris C R, Millman K J, van der Walt S J, et al. Array programming with NumPy[J]. Nature, 2020, 585: 357-362.

[7] McKenna F. OpenSees: a framework for earthquake engineering simulation[J]. Computing in Science & Engineering, 2011, 13(4): 58-66.

[8] Huangfu Q, Hall J A J. Parallelizing the dual revised simplex method[J]. Mathematical Programming Computation, 2018, 10: 119-142.

[9] Fernandes G, Lourenço N, Correia J. Reducing the price of stable cable stayed bridges with CMA-ES[EB/OL]. arXiv:2304.00641, 2023.

[10] BridgeZoo 项目源码与测试：`bridgezoo/optim/linear.py`, `bridgezoo/optim/hybrid.py`, `tests/test_optim.py`, `scripts/optimize_cables.py`.

## 投稿前修订清单

- 将作者、单位、基金项目、中图分类号等期刊格式信息补齐。
- 按目标期刊格式重写摘要字数、英文题名、英文摘要和图表编号。
- 用正式批量实验补齐表 2 至表 5，包括对比基线和消融结果。
- 重新跑主算例并固定随机种子、硬件环境、Python 与依赖版本。
- 补充中文核心近五年相关文献，尤其是斜拉桥施工控制、索力优化、影响矩阵法和智能优化调索方向。
- 将本文中的“项目源码与测试”引用替换为可公开复现的数据仓库或匿名补充材料说明。
