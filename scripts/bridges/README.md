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

### Current `single` construction topology

The current `single_staged` implementation models an asymmetric single-tower
bridge:

- the right girder has one segment ending at one fully fixed node;
- each right stay terminates at an independent fully fixed ground anchor at
  `right_start + (i - 1) * right_spacing`, rather than at a girder node;
- construction starts by activating the first girder segment on both sides;
- cable stage 1 activates the first left/right stays, and every later cable
  stage activates one left girder segment plus its left/right stays;
- the process currently ends with the free left tip segment at `tip_free`.

No closure support or `phase2` step is added after `tip_free` yet. Accordingly,
`dw` and `right_end` remain accepted shared configuration fields but are not
applied by the current `single_staged` process. The `normal` model retains its
existing symmetric double-cantilever sequence and `phase2` behavior.

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
base is fixed, while the existing deck-root translation fixity and rotation
release remain unchanged.
