# Bridge YAML configuration

`staged_analysis`, `validate_staged`, and `optimize_cables` accept a bundled
configuration name or a YAML path:

```bash
python -m scripts.staged_analysis --bridge p4b
python -m scripts.staged_analysis --bridge scripts/bridges/model_defaults.yaml
```

Command-line scalar options still override the corresponding YAML values.

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
