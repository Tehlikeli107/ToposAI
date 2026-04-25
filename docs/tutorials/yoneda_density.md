# Yoneda Density Tutorial

This tutorial uses the finite walking-arrow category `0 -> 1` and a presheaf `F` with `F(0) = {a, b}`, `F(1) = {u}`, and `F(up)(u) = a`.

Run:

```bash
python examples/formal_yoneda_density.py
```

The example constructs `int F`, the category of elements, and reconstructs `F` as `colim_{(c, x) in int F} y(c)`.

Expected summary keys:

- `objects_in_category_of_elements`
- `projection_target_objects`
- `round_trip_to_presheaf`
- `round_trip_to_density`

Mathematical statement: in the finite setting used here, every presheaf is recovered from representables indexed by its category of elements.