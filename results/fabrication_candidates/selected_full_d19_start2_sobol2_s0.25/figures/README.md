# Figures: Full-D19 start2 (selected by criterion v2 in Phase 11, pending the owner's review)

Renders, animations and charts of the design in the parent directory (`../optimized_morphology.design.json`), generated
2026-10-24 by `dkarma/experiments/visualize_design.py --phase-dir workspace/phase11`. Everything shown is either the
design's geometry or a number already reported in `docs/phase11_report.md` (§11.3–11.5); nothing here is a new
evaluation.

## Files

| file | what it shows |
| --- | --- |
| `hand_views.png` | The hand holding the sphere at grasp seed 4 (the exact seed state): a 3/4 perspective view and orthographic views from +x, +y, +z and −z with a 20 mm scale bar, plus link lengths, joint ranges and the admissibility margins (base separation ≥ 33.3 mm, D19) |
| `design_before_after.png` | N11 (the nominal start), start2's initial design (Sobol point 2, scale 0.25) and the optimized design, all at the same joint angles (0°, 30°, 30°, 30° per finger) and scale, from +x and +y |
| `grasp_gallery.png` | The 12 three-contact grasps of the default seed configuration (the 11.4 suite's and export's seeds), each with its weakest-direction wrench capacity, its settle drift in simulation and its strict primitive count of 84 |
| `manipulation_seed4.mp4` | 36 s, 1280 × 720. The MuJoCo replay of four commands from grasp seed 4: rotation about +ê1 (10 of 10), translation along −ê1 (4 of 4), rotation about −ê1 (10 of 10) and translation along +ê1 (3 of 4, no drop). On-screen: per-primitive verdicts, the three contact normal forces with the controller's floor, the live centre and orientation errors |
| `manipulation_seed4_rotation.gif` | The first clip of the video (rotation, 10 of 10) as an 800 px GIF |
| `all_commands_seed4.mp4` | 11 s, 1600 × 964. All twelve commands of grasp seed 4 side by side, synchronized |
| `command_success.png` | Strict primitives per grasp seed and command (12 × 12): the table behind E = 29.75 |
| `wrench_capacity.png` | α(d) at each grasp seed in the 14 standard directions; the outlined cell is each seed's weakest |
| `standing_among_designs.png` | E and the exact Full score of the 24 Phase 11 designs; start2 highlighted |
| `load_sweep.png` | Exact Full score and seeds holding at the stall, primary and continuous ceilings and with a 70 g object |

## Conditions

* **Simulation** (videos, `command_success.png`, the gallery's primitive counts): MuJoCo 3.15 with compliant silicone
  pads (Shore 20A, 2 mm, `dkarma/configs/compliant_pads.json`) on the 11.8 mm fingertips and the force-floor controller
  with a 0.75 mm squeeze, at the 0.108 N·m ceiling. These are the stored 11.4 settings. The videos are re-runs of the same
  replay through `simulation_replay.simulate_design`. Each replayed trajectory's per-primitive errors were checked to be
  bit-identical to the stored 11.4 rows before any frame was written.
* **Strict success** (green box): at the primitive end, the sphere centre is within 4 mm and its orientation within
  0.2 rad of the plan, and all three fingertips touch it. E counts primitives per command up to the first failure. A grey
  box passes but comes after a failure and is not counted.
* **Drop criterion** (9.3): the centre is more than 20 mm from the plan, or no fingertip touches the sphere. The replay
  stops there. *Released* means no contact; *> 20 mm off the plan* means still touching.
* **ê1, ê2, ê3**: the axes of each grasp seed's frame. The commands are ±ê translations (4 × 10 mm) and rotations about
  ±ê (10 × 0.15 rad).
* **Wrench capacity**: α(d) is the largest multiple of the standard load along d that the grasp carries on top of the
  object's weight. It comes from the exact SLSQP force problem at 0.108 N·m. The standard load is 0.343 N (35 g) for forces
  along the palm axes, or 0.343 N × 15 mm for moments. A seed with α ≥ 1 in all 14 directions is in `S_task`.
* **Rendering**: a separate visual copy of the simulated scene that only receives `qpos`. Links are drawn as the modelled
  capsules: r = 16.4 mm for links 0–2 (the XL330 servo envelope, D19) and 11.8 mm for the fingertips. The palm is not
  modelled. Gravity is along −z.

## What the figures show that the summary tables do not

* **E = 29.75 is mostly rotation.** 24.08 primitives per seed come from rotations (40 % of the possible 60) and 5.67 from
  translations (24 % of 24).
  * Translations, over all 12 seeds: 4 of 72 complete all four primitives. 57 of 72 release the sphere and 3 end more than
    20 mm off the plan.
  * Rotations: 15 of 72 complete all ten, and 33 of 72 release the sphere.

  This is the Phase 10 selection's pattern (38.6 + 6.4), at a lower level.
* Two grasps are not robust: seeds 3 and 9, α_min 0.78 and 0.75. Seeds 7 and 9 do not hold in simulation (settle drift
  181 mm and 236 mm), although seed 7 is robust (α_min 1.28). As in Phase 10, the capacity does not predict which commands
  succeed: seed 4 (α_min 3.19) scores 69 of 84, while seed 2 (2.62) scores 49.
* The design is the best of the 24 on E and the second best on the exact Full score; start6 is first on that score but
  fails criterion v2's R5.
* At the continuous ceiling (0.043 N·m) only 1 of 12 grasps holds (`load_sweep.png`); Phase 10's selection held 10 of 12.

## Dimensions (from `../optimized_morphology.design.json`)

| finger | base position (mm, palm frame) | link lengths, base → tip (mm) | joint ranges (deg), joint 0 → 3 |
| --- | --- | --- | --- |
| index | (−4.3, −8.9, 11.6) | 40.4 / 40.4 / 40.5 / 30.4 | −20…20, −15…75, −15…75, −15…75 |
| middle | (9.7, 47.7, 1.0) | 40.2 / 41.6 / 38.5 / 31.1 | −20…20, −15…75, −15…75, −15…75 |
| thumb | (−10.9, −44.4, −28.3) | 39.6 / 44.7 / 39.0 / 25.7 | −45…45, −15…60, −15…60, −15…75 |

Capsule radii: 16.40 mm (links 0–2), 11.8 mm (fingertips). Base orientations and joint axes are in the design JSON
(`base_rotvec`, `axis_theta`, `axis_phi`). The URDF carries the joint tree and every link's capsule as `<visual>`
geometry.

Admissibility margins:

* minimum base separation 53.9 mm (≥ 33.30 mm, D19);
* shortest servo link 38.5 mm (≥ 34 mm);
* reach 151.8 mm (≤ 250 mm).

The terminal link length includes the fingertip; the capsule of the last link ends one radius before the tip point.

## Sources and how to regenerate

```bash
MUJOCO_GL=osmesa JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.visualize_design --phase-dir workspace/phase11 --key full_d19/start2_sobol2_s0.25 --seed 4 --out results/fabrication_candidates/selected_full_d19_start2_sobol2_s0.25/figures
```

Reads `workspace/phase11/`:

* `11.4/sim/default/full_d19_start2_{design.json,trajectories.npz}` and `11.4/sim/default/20A_2mm/simulation_force__full_d19_start2.json`;
* `11.4/suite/full_d19__start2_sobol2_s0.25/suite.json`, `11.4/sweeps/full_d19__start2_sobol2_s0.25.json`;
* `11.4/evaluation.json` and `11.4/stask_report.json`;
* `11.3/full_d19/start2_sobol2_s0.25/{initial_morphology,optimized_morphology,validation_history.csv}`;
* `11.2/n11/n11.design.json` (N11).

`workspace/` is gitignored, so these figures are the kept copy. Rendering is software OpenGL (OSMesa). The renders are
deterministic in content, but re-encoded videos can differ byte for byte.
