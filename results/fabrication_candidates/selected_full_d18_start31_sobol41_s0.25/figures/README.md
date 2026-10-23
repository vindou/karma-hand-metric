# Figures: Full-D18 start31 (selected by criterion v2, pending the owner's review)

Renders, animations and charts of the design in the parent directory (`../optimized_morphology.design.json`), generated
2026-10-23 by `dkarma/experiments/visualize_design.py`. Everything shown is either the design's geometry or a number
already reported in `docs/phase10_report.md` (§10.2-10.4); nothing here is a new evaluation.

## Files

| file | what it shows |
| --- | --- |
| `hand_views.png` | The hand holding the sphere at grasp seed 8 (the exact seed state): a 3/4 perspective view and orthographic views from +x, +y, +z and −z with a 20 mm scale bar, plus link lengths, joint ranges and the admissibility margins |
| `design_before_after.png` | The anthropomorphic start, start31's initial design (Sobol point 41, scale 0.25) and the optimized design, all at the same joint angles (0°, 30°, 30°, 30° per finger) and scale, from +x and +y |
| `grasp_gallery.png` | The 12 three-contact grasps of the default seed configuration (the 10.2 suite's and export's seeds), each with its weakest-direction wrench capacity, its settle drift in simulation and its strict primitive count of 84 |
| `manipulation_seed8.mp4` | 34 s, 1280 × 720. The MuJoCo replay of four commands from grasp seed 8: rotation about +ê1 (10 of 10), translation along −ê1 (4 of 4), rotation about −ê1 (8 of 10) and translation along +ê1 (1 of 4; the sphere is released in primitive 3). On-screen: per-primitive verdicts, the three contact normal forces with the controller's floor, the live centre and orientation errors, and all-seeds context for that command type |
| `manipulation_seed8_rotation.gif` | The first clip of the video (rotation, 10 of 10) as an 800 px GIF |
| `all_commands_seed8.mp4` | 11 s, 1600 × 964. All twelve commands of grasp seed 8 side by side, synchronized |
| `command_success.png` | Strict primitives per grasp seed and command (12 × 12): the table behind E = 45.0 |
| `wrench_capacity.png` | α(d) at each grasp seed in the 14 standard directions; the outlined cell is each seed's weakest |
| `standing_among_designs.png` | E and the exact Full score of all 48 10.2 designs, paired by start; start31 highlighted |
| `load_sweep.png` | Exact Full score and seeds holding at the stall, primary and continuous ceilings and with a 70 g object |

## Conditions

* **Simulation** (videos, `command_success.png`, the gallery's primitive counts): MuJoCo 3.15 with compliant silicone
  pads (Shore 20A, 2 mm, `dkarma/configs/compliant_pads.json`) and the force-floor controller with a 0.75 mm squeeze, at
  the 0.108 N·m ceiling. These are the stored 10.2 settings. The videos are re-runs of the same replay through
  `simulation_replay.simulate_design`. Each replayed trajectory's per-primitive errors were checked to be bit-identical
  to the stored 10.2 rows before any frame was written.
* **Strict success** (green box): at the primitive end the sphere centre is within 4 mm and its orientation within
  0.2 rad of the plan, and all three fingertips touch it. E counts primitives per command up to the first failure. A
  grey box passes but comes after a failure and is not counted.
* **Drop criterion** (9.3): the centre is more than 20 mm from the plan, or no fingertip touches the sphere. The replay
  stops there. The labels say which happened: *released* (no contact) or *> 20 mm off the plan* (still touching).
* **ê1, ê2, ê3**: the axes of each grasp seed's frame. The commands are ±ê translations (4 × 10 mm) and rotations about
  ±ê (10 × 0.15 rad), as in `multicontact.multi_library`.
* **Wrench capacity**: α(d) is the largest multiple of the standard load along d that the grasp carries on top of the
  object's weight. It comes from the exact SLSQP force problem at 0.108 N·m. The standard load is 0.343 N (35 g) for
  forces along the palm axes, or 0.343 N × 15 mm for moments. A seed with α ≥ 1 in all 14 directions is in `S_task`.
* **Rendering**: a separate visual copy of the simulated scene, with materials, lights, the palm-frame axes (x red,
  y green, z blue) and, in the static 3/4 view, a floor. The copy only receives `qpos`. The floor has no collision, which
  is why the animations leave it out. The palm itself is not modelled, and links are drawn as the modelled capsules
  (r = 11.8 mm). Gravity is along −z.

## What the figures show that the summary tables do not

* **E = 45.0 is mostly rotation.** 38.6 primitives per seed come from rotations (64 % of the possible 60) and 6.4 from
  translations (27 % of 24).
  * Translations, over all 12 seeds: only 1 of 72 completes all four primitives, and it is the translation along −ê1 in
    `manipulation_seed8.mp4`. 43 of 72 release the sphere and 12 end more than 20 mm off the plan.
  * Rotations: 23 of 72 complete all ten, and 16 of 72 release the sphere.

  The runner-up of criterion v2 (Full-D18 start0) shows the same pattern: 21.8 from rotations and 5.4 from
  translations.
* Every grasp is robust: the weakest direction over all 12 seeds is α = 1.92 (seed 5). The capacity does not predict
  which commands succeed: seed 6 has α_min 2.41 but scores 16 of 84, while seed 8 has 4.40 and scores 67.
* The design is not the best of the 48 on the exact Full score (start15 is). It is the best on E.

## Dimensions (from `../optimized_morphology.design.json`)

| finger | base position (mm, palm frame) | link lengths, base → tip (mm) | joint ranges (deg), joint 0 → 3 |
| --- | --- | --- | --- |
| index | (1.0, −7.9, 3.2) | 12.0 / 43.5 / 27.0 / 26.1 | −20…20, −15…75, −15…75, −15…75 |
| middle | (14.5, 41.7, 10.9) | 12.0 / 47.1 / 24.0 / 23.7 | −20…20, −15…75, −15…75, −15…75 |
| thumb | (−14.6, −37.0, −31.4) | 14.3 / 42.8 / 28.0 / 30.7 | −45…45, −15…60, −15…60, −15…75 |

Base orientations and joint axes are in the design JSON (`base_rotvec`, `axis_theta`, `axis_phi`) and in the URDF.
Minimum base separation 47.9 mm (≥ 24.1 mm, D18); reach 115.8 mm (≤ 250 mm). The terminal link length includes the
fingertip; the capsule of the last link ends one radius before the tip point.

## Sources and how to regenerate

```bash
MUJOCO_GL=osmesa JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.visualize_design   # ~4 min, all stages
# one part only: --stages views compare gallery charts animate;  another seed: --seed K;  another 10.2 design: --key <arm>/<start> --out <dir>
```

Reads `workspace/phase10/10.2/`:
* `sim/default/full_d18_start31_{design.json,trajectories.npz}`
* `sim/default/20A_2mm/simulation_force__full_d18_start31.json`
* `suite/full_d18__start31_sobol41_s0.25/suite.json`
* `sweeps/full_d18__start31_sobol41_s0.25.json`
* `comparison.json` and `summary.json`
* `full_d18/start31_sobol41_s0.25/{initial_morphology,optimized_morphology,validation_history.csv}`

It also reads `workspace/phase10/10.3/stask_report.json` and, for the anthropomorphic start,
`workspace/phase9/9.3/start_design.json`.

`workspace/` is gitignored, so these figures are the kept copy. Rendering is software OpenGL (OSMesa; EGL is not available
to this user on linbox01). The renders are deterministic in content, but re-encoded videos can differ byte for byte.
