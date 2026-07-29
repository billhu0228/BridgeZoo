# Bridge YAML configuration

`staged_analysis`, `validate_staged`, and `optimize_cables` accept a bundled
configuration name or a YAML path:

```bash
python -m scripts.staged_analysis --bridge p4b
python -m scripts.staged_analysis --bridge omo
python -m scripts.staged_analysis --bridge scripts/bridges/model_defaults.yaml
```

Command-line scalar options still override the corresponding YAML values.

## Bridge type

Every bridge configuration must declare its model family:

```yaml
bridge_type: normal  # bridgezoo.fem.staged
```

The supported values are:

- `normal`: use `bridgezoo.fem.staged`.
- `single`: use `bridgezoo.fem.single_staged`.

The bundled `model` and `p4b` configurations are `normal`; `omo` is `single`
and uses the single-tower construction model. Analysis, validation, and
optimization entry points all dispatch from this field.

## OMO 3D configuration

`omo_bridge_3d.yaml` is the complete input for the new 3D single-tower
grillage. It is intentionally separate from the 2D schema and declares
`bridge_type: single_3d`. Use its bundled alias with the 3D analysis script:

```bash
python -m scripts.single_staged_3d --bridge omo3d --backend opensees --render both
```

The YAML explicitly contains:

- inherited OMO longitudinal cable/tower/span dimensions;
- twin-main-girder spacing, equal cross-girder target spacing, deck width,
  thickness and beam-axis eccentricity;
- H-section dimensions for main and cross girders and hollow-box dimensions
  for the tower;
- elastic modulus, Poisson ratio and density for structural steel, concrete
  and cable steel;
- strand area/count, total cable pretension, the explicit pretension-A ratio,
  gravity and superimposed deck dead load.

The loader rejects missing or unknown keys. `cross_girder_spacing` is the
maximum target spacing; the builder divides the complete deck length into an
integer number of equal bays not exceeding that value.

Every 3D erection stage has three incremental substeps: steel girders and
stays are activated with pretension A; wet-deck self-weight is converted to
line loads on the newly erected main girders while pretension B is applied;
then the temporary load definition expires and the eccentric slab grillage is
activated through rigid links. `pretension_a_ratio` may be a scalar or use the
same stage-major `(backstay, main_stay)` shape as `pretension_per_cable`.

Before each steel/A substep, the newly erected frame-and-cable group receives
an unloaded actual-stiffness virtual solve.  Previously active connection
nodes retain their committed displacement, and the new nodes take the elastic
extension compatible with the new beam/truss stiffness.  That virtual field is
then used as the stress-free birth geometry.  A free extension with no new
restraint naturally reduces to rigid tangent extrapolation.

### Current `single` construction topology

The current `single_staged` implementation models an asymmetric single-tower
bridge:

- the right girder has one segment ending at node 1, whose translations are
  fixed and rotation is released; its x coordinate is the `single`-only YAML
  field `right_fix`;
- each right stay terminates at an independent fully fixed ground anchor at
  `right_start + (i - 1) * right_spacing`, rather than at a girder node;
- the deck node at the tower-girder intersection has no support constraint;
- cable stage 1 activates the first girder segment on both sides together with
  the first left/right stays, and every later cable stage activates one left
  girder segment plus its left/right stays;
- `tip_free` installs and solves the final free left segment;
- the final `left_tip_uy_lock` stage locks node 201's vertical displacement at
  its current deformed position;
- the `left_span` stage then tangent-activates an additional segment extending
  left by the `single`-only YAML length `left_span`, and locks its new end node
  202 vertically at its birth position;
- when `dw != 0`, the final `phase2` stage applies `-dw` to every active deck
  frame, including the right fixed segment and the new `left_span` segment.

Both `right_fix` and `left_span` are required, positive, `single`-only YAML
fields. `right_end` remains an accepted shared configuration field but is not
applied by the current `single_staged` process. The `normal` model retains its
existing symmetric double-cantilever sequence and its own `phase2` behavior.

## Tower stiffness

The tower is a fixed-base 2D Euler-Bernoulli frame. Its stiffness fields are:

```yaml
tower_stiffness:
  - [0.0, 1.0e18]
  - [110.0, 5.0e17]
tower_element_size: 2.0
tower_axial_rigidity: 1.0e15
```

- `tower_stiffness`: `(z, k)` pairs where `z` is elevation above deck level
  in metres and `k = EI` is flexural rigidity in `N·m²`.
- `tower_element_size`: maximum tower beam-element length in metres. Cable
  anchors and stiffness control elevations are always retained as nodes.
- `tower_axial_rigidity`: tower axial rigidity `EA` in newtons.

`EI` is linearly interpolated between control points and held constant below
the first and above the last point. The tower currently has no self-weight.
The tower base and deck root are coincident but use separate nodes: the tower
base is fixed, while the deck root has no support constraint.
