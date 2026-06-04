import openseespy.opensees as ops


# ============================================================
# Units: kN, m
# 2D Euler-Bernoulli beam model
# ============================================================

L = 5.0          # length of each beam segment, m
E = 3.0e7        # kN/m2
A = 1.0          # m2
Iz = 2.0e6 / E   # m4, so EI = 2.0e6 kN*m2

EI = E * Iz

P1 = 100.0       # kN, downward load at node 1
P2 = 100.0       # kN, downward load at node 2

Kv = 5000.0      # kN/m, vertical spring at the cantilever tip


def setup_analysis():
    ops.system("BandGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")


def get_ele_local_force(ele_tag):
    """
    For 2D elasticBeamColumn, localForce usually returns:
    [N_i, V_i, M_i, N_j, V_j, M_j]
    Sign convention follows OpenSees local element convention.
    """
    return ops.eleResponse(ele_tag, "localForce")


def get_spring_force(ele_tag):
    """
    zeroLength element with one uniaxial material.
    material stress is the spring force.
    """
    resp = ops.eleResponse(ele_tag, "material", 1, "stress")
    if isinstance(resp, (list, tuple)):
        return resp[0]
    return resp


def run_one_step():
    """
    One-step completed structure:

        fixed -- beam 1 -- node 1 -- beam 2 -- node 2
                                                   |
                                             vertical spring
                                                   |
                                                ground

    P1 and P2 are applied at the same time.
    The tip spring exists from the beginning.

    This is equivalent to a completed structure analysis.
    """

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    # Nodes
    ops.node(0, 0.0, 0.0)
    ops.node(1, L, 0.0)
    ops.node(2, 2.0 * L, 0.0)

    # Ground node for vertical spring
    ops.node(102, 2.0 * L, 0.0)

    # Boundary conditions
    ops.fix(0, 1, 1, 1)
    ops.fix(102, 1, 1, 1)

    # Beam transformation
    ops.geomTransf("Linear", 1)

    # Beam elements
    ops.element("elasticBeamColumn", 1, 0, 1, A, E, Iz, 1)
    ops.element("elasticBeamColumn", 2, 1, 2, A, E, Iz, 1)

    # Tip vertical spring
    ops.uniaxialMaterial("Elastic", 100, Kv)
    ops.element("zeroLength", 100, 102, 2, "-mat", 100, "-dir", 2)

    # Loads
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)

    ops.load(1, 0.0, -P1, 0.0)
    ops.load(2, 0.0, -P2, 0.0)

    setup_analysis()

    ok = ops.analyze(1)
    if ok != 0:
        raise RuntimeError("One-step analysis failed.")

    result = {
        "u1_y": ops.nodeDisp(1, 2),
        "r1_z": ops.nodeDisp(1, 3),
        "u2_y": ops.nodeDisp(2, 2),
        "r2_z": ops.nodeDisp(2, 3),
        "beam1_force": get_ele_local_force(1),
        "beam2_force": get_ele_local_force(2),
        "spring_force": get_spring_force(100),
    }

    return result


def run_staged_construction():
    """
    Staged construction:

    Stage 1:
        fixed -- beam 1 -- node 1

        Apply P1 at node 1.

    Stage 2:
        Add beam 2 and the vertical tip spring.

        The second beam is installed after node 1 has already displaced
        and rotated. Therefore, node 2 is first placed on the current
        tangent line of node 1, so beam 2 is stress-free at installation.

        Then P2 is applied at node 2.

    The tip spring is also installed at Stage 2, stress-free at installation.
    """

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    # ========================================================
    # Stage 1: first cantilever segment only
    # ========================================================

    ops.node(0, 0.0, 0.0)
    ops.node(1, L, 0.0)

    ops.fix(0, 1, 1, 1)

    ops.geomTransf("Linear", 1)

    ops.element("elasticBeamColumn", 1, 0, 1, A, E, Iz, 1)

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)

    ops.load(1, 0.0, -P1, 0.0)

    setup_analysis()

    ok = ops.analyze(1)
    if ok != 0:
        raise RuntimeError("Stage 1 analysis failed.")

    u1_stage1 = ops.nodeDisp(1, 2)
    r1_stage1 = ops.nodeDisp(1, 3)

    beam1_stage1_force = get_ele_local_force(1)

    # Keep Stage 1 load and deformation state
    ops.loadConst("-time", 0.0)

    # ========================================================
    # Stage 2: add second beam segment and tip spring
    # ========================================================

    ops.wipeAnalysis()

    # The new beam segment is installed from the already deformed node 1.
    # For small-deformation beam theory, the stress-free position of node 2
    # can be approximated by extending the tangent at node 1:
    #
    # u2_install = u1_stage1 + rotation1_stage1 * L
    # r2_install = r1_stage1
    #
    # This makes beam 2 stress-free immediately after installation.
    u2_install = u1_stage1 + r1_stage1 * L
    r2_install = r1_stage1

    ops.node(2, 2.0 * L, 0.0)
    ops.node(102, 2.0 * L, 0.0)

    ops.fix(102, 1, 1, 1)

    # Set the installed displacement state of the new node.
    # This simulates that the second segment is erected on the current
    # deformed tangent of node 1, not on the original undeformed line.
    ops.setNodeDisp(2, 1, 0.0, "-commit")
    ops.setNodeDisp(2, 2, u2_install, "-commit")
    ops.setNodeDisp(2, 3, r2_install, "-commit")

    # Add beam 2
    ops.element("elasticBeamColumn", 2, 1, 2, A, E, Iz, 1)

    # Add the vertical spring at node 2.
    # The spring should also be stress-free when installed.
    #
    # zeroLength deformation in dir 2 is approximately:
    #     u_node2 - u_ground = u2_install
    #
    # InitStrainMaterial gives:
    #     stress = K * (deformation + initStrain)
    #
    # To make initial spring force zero:
    #     initStrain = -u2_install
    ops.uniaxialMaterial("Elastic", 200, Kv)
    ops.uniaxialMaterial("InitStrainMaterial", 201, 200, -u2_install)
    ops.element("zeroLength", 100, 102, 2, "-mat", 201, "-dir", 2)

    try:
        ops.domainChange()
    except Exception:
        pass

    # Apply Stage 2 load P2
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)

    ops.load(2, 0.0, -P2, 0.0)

    setup_analysis()

    ok = ops.analyze(1)
    if ok != 0:
        raise RuntimeError("Stage 2 analysis failed.")

    result = {
        "u1_stage1": u1_stage1,
        "r1_stage1": r1_stage1,
        "u2_install": u2_install,
        "r2_install": r2_install,
        "beam1_stage1_force": beam1_stage1_force,
        "u1_y": ops.nodeDisp(1, 2),
        "r1_z": ops.nodeDisp(1, 3),
        "u2_y": ops.nodeDisp(2, 2),
        "r2_z": ops.nodeDisp(2, 3),
        "beam1_force": get_ele_local_force(1),
        "beam2_force": get_ele_local_force(2),
        "spring_force": get_spring_force(100),
    }

    return result


def fmt_force_list(f):
    return "[" + ", ".join(f"{x: .6f}" for x in f) + "]"


def print_results(one, staged):
    print("\n" + "=" * 90)
    print("Cantilever beam staged installation verification by OpenSeesPy")
    print("=" * 90)

    print(f"\nParameters:")
    print(f"L  = {L:.3f} m per segment")
    print(f"EI = {EI:.3f} kN*m2")
    print(f"P1 = {P1:.3f} kN downward")
    print(f"P2 = {P2:.3f} kN downward")
    print(f"Kv = {Kv:.3f} kN/m vertical spring")

    print("\n" + "-" * 90)
    print("[1] One-step completed structure")
    print("-" * 90)
    print(f"Node 1 vertical displacement Uy1 = {one['u1_y']:.9f} m")
    print(f"Node 1 rotation Rz1              = {one['r1_z']:.9f} rad")
    print(f"Node 2 vertical displacement Uy2 = {one['u2_y']:.9f} m")
    print(f"Node 2 rotation Rz2              = {one['r2_z']:.9f} rad")
    print(f"Beam 1 local force [N_i, V_i, M_i, N_j, V_j, M_j]:")
    print(fmt_force_list(one["beam1_force"]))
    print(f"Beam 2 local force [N_i, V_i, M_i, N_j, V_j, M_j]:")
    print(fmt_force_list(one["beam2_force"]))
    print(f"Tip spring force = {one['spring_force']:.6f} kN")

    print("\n" + "-" * 90)
    print("[2] Staged construction")
    print("-" * 90)
    print("Stage 1 result:")
    print(f"Node 1 vertical displacement Uy1_stage1 = {staged['u1_stage1']:.9f} m")
    print(f"Node 1 rotation Rz1_stage1              = {staged['r1_stage1']:.9f} rad")
    print(f"Beam 1 local force after Stage 1:")
    print(fmt_force_list(staged["beam1_stage1_force"]))

    print("\nStage 2 installation state:")
    print(f"Node 2 installed vertical displacement Uy2_install = {staged['u2_install']:.9f} m")
    print(f"Node 2 installed rotation Rz2_install              = {staged['r2_install']:.9f} rad")
    print("This makes the newly installed second beam segment stress-free at birth.")

    print("\nFinal after Stage 2:")
    print(f"Node 1 vertical displacement Uy1 = {staged['u1_y']:.9f} m")
    print(f"Node 1 rotation Rz1              = {staged['r1_z']:.9f} rad")
    print(f"Node 2 vertical displacement Uy2 = {staged['u2_y']:.9f} m")
    print(f"Node 2 rotation Rz2              = {staged['r2_z']:.9f} rad")
    print(f"Beam 1 local force [N_i, V_i, M_i, N_j, V_j, M_j]:")
    print(fmt_force_list(staged["beam1_force"]))
    print(f"Beam 2 local force [N_i, V_i, M_i, N_j, V_j, M_j]:")
    print(fmt_force_list(staged["beam2_force"]))
    print(f"Tip spring force = {staged['spring_force']:.6f} kN")

    print("\n" + "-" * 90)
    print("Comparison")
    print("-" * 90)
    print(f"One-step Uy2 = {one['u2_y']:.9f} m")
    print(f"Staged   Uy2 = {staged['u2_y']:.9f} m")
    print(f"Difference  = {staged['u2_y'] - one['u2_y']:.9f} m")

    print(f"\nOne-step tip spring force = {one['spring_force']:.6f} kN")
    print(f"Staged   tip spring force = {staged['spring_force']:.6f} kN")
    print(f"Difference                = {staged['spring_force'] - one['spring_force']:.6f} kN")

    print("\nConclusion:")
    print("The final structure and final external loads are the same,")
    print("but because the second segment and the vertical spring are installed later,")
    print("the final displacement, element forces, and spring force are different.")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    one_step_result = run_one_step()
    staged_result = run_staged_construction()
    print_results(one_step_result, staged_result)