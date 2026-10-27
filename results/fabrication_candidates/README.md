# Fabrication candidates (preserved in Phase 8.7, 2026-10-11)

Two 3x4 designs (thumb + index + middle, 4 DOF each, one XL330-M077-T per joint, direct drive)
from Experiment IV of `docs/phase7_report.md`, kept here with their exported specifications so
that the realization protocol of `SPEC.md` §12 IV can start from them. **Nothing has been
selected for fabrication**: the selection criterion is written in `docs/phase8_report.md`
§8.7 and the selection waits for items 8.1 to 8.6 of SPEC §11.1.

| candidate | origin | files |
| --- | --- | --- |
| `armA/` | CMA-ES against the original KaRMA (fast proxy, 12 generations x 8) from the anthropomorphic start, `workspace/phase7/exp34/armA_design/`; scored in Phase 7 as the best two-finger hand of the set and with no three-contact grasp found (verified in Phase 8.5) | `armA.urdf`, `robot_armA.yaml` (KaRMA robot config), `armA.design.json` (dkarma design), `actuator_spec.yaml` |
| `armB/` | Full KaRMA + Adam (40 iterations, 2 stages, 6 three-contact seeds), `workspace/phase7/exp34/run_full/optimized_morphology/`; the design with the largest and most uniform three-contact wrench margins of Experiment III | `optimized_morphology.urdf`, `robot_optimized_morphology.yaml`, `optimized_morphology.design.json`, `actuator_spec.yaml`, `config.yaml`, `random_seeds.json`, `results_summary.json` |

The actuator specification is the frozen one of SPEC §2.1: torque ceiling 0.108 N·m per joint
(50 % of the 0.215 N·m stall at 5 V), link mass 28 g (terminal 10 g), transmission efficiency
1.0; these are assumptions until a joint-side torque curve is measured. Geometry: capsule links
of radius 11.8 mm; the URDF carries the joint tree (origins, axes, limits) and the tip lengths
(`tip_length_m` in the robot YAML); `effort` / `velocity` fields are placeholders.

Re-derive any number from the design JSON with `dkarma.export.design_from_dict` and the
`dkarma.experiments` scripts named in the Phase 7 and Phase 8 reports.


## Phase 9 (2026-10-14): selection by the written criterion, nothing selected

The criterion of `docs/phase8_report.md` §8.7(b), amended for D18 (no two modelled fingers in the same
space: finger bases ≥ 24.1 mm apart, reach ≤ 250 mm per finger, all-finger clearance), was applied in its
operational form `docs/phase9_report.md` §9.5, fixed before the data. The three candidates were arm B,
the arm A′ design that scores best on its own objective (`armAprime_start6`), and the best Full-KaRMA D18
design (`full_d18_start0`). All three fail the robustness requirement: some seed has wrench capacity
< 1 × the object's weight in some direction. All three also fail the simulation-agreement requirement:
three-contact false-positive rate in MuJoCo above the two-finger baseline's, under both 9.3 controllers.
**Nothing is selected**, and the criterion was not loosened. Arm A is ineligible (index and middle bases
17.8 mm apart) and stays as the failure case.

| directory | design | status | key numbers (fixed physical task; 12 three-contact / 16 two-finger fresh seeds × 3 seed configurations) |
| --- | --- | --- | --- |
| `full_d18_start0/` | Full KaRMA + Adam under D18 from the anthropomorphic start (`workspace/phase9/9.2/full_d18/start0_nominal/`) | candidate, not selected | exact Full score 0.0215 ± 0.0017, the best of Phase 9 under every seed configuration; two-finger exact score 0.1747; holds 11 of 12 seeds at the primary ceiling; weakest seed capacity 0.00 × weight |
| `armAprime_start6/` | original KaRMA + CMA-ES under D18 (`workspace/phase9/9.2/armAprime/start6_sobol9_s0.5/`) | candidate, not selected | exact Full score 0.0093 ± 0.0010; two-finger exact score 0.2374, the best of Phase 9; published KaRMA-T (all-finger gate) 0.0408; weakest seed capacity 0.16 × weight |
| `armB/` | Phase 7 Full KaRMA + Adam | candidate, not selected | exact Full score 0.0146 ± 0.0014; two-finger 0.2055; weakest seed capacity 0.00 × weight |
| `armA/` | Phase 7 original KaRMA + CMA-ES, no D18 | ineligible: failure case | no three-contact seed found (792 / 1,485 / 726 IK attempts under three seed-search configurations, none kinematically valid; see also Phase 8.5); physical original KaRMA 0.0140 with the pair gate, 0.0001 with the all-finger gate |

Each preserved directory holds the design JSON, URDF, robot YAML, actuator specification, the run
record, and `candidate_numbers.json` with the criterion's numbers (written by
`dkarma.experiments.candidate_selection --preserve`). Whether to amend the robustness or agreement
requirement, which would make `full_d18_start0` the leading candidate on the primary measure, is the
owner's decision.


## Phase 10 (2026-10-23): criterion v2 selects `selected_full_d18_start31_sobol41_s0.25/`; the original criterion still selects nothing

Criterion v2 (`docs/phase10_report.md` "P10.4"; SPEC §16 item 12) was frozen by the session on 2026-10-14, before any
Phase 10 data was produced and before any candidate was evaluated against it, **subject to the owner's review**. It is
written from the deployment requirements: three-fingertip rolling of the 15 mm / 35 g sphere at the 0.108 N·m ceiling,
compliant silicone fingertips, the 9.3 force-floor controller, and a planner that chooses among grasps. Its parts:

* R1, eligibility as in 9.5;
* R2, at least 6 of 12 robust grasps (`S_task = {s : min_d α_s(d) ≥ 1}`) under every seed configuration;
* R3, an exact Full score on `S_task` at least the anthropomorphic start's (0.0196);
* R4, the 9.5 load sweep;
* R5, a three-contact false-positive rate under the force-floor controller with the central compliant fingertips at most
  the median of the 9.2 libraries (0.239).

| directory | design | status | key numbers (fixed physical task) |
| --- | --- | --- | --- |
| `selected_full_d18_start31_sobol41_s0.25/` | Full KaRMA + Adam under D18 from 10.2 start31 (`workspace/phase10/10.2/full_d18/start31_sobol41_s0.25/`, optimizer seed 0) | **selected by criterion v2**; not selected by the original 9.5 criterion (fails requirement 4: rigid three-contact FP 0.49 / 0.44 against the two-finger baseline's 0.19 / 0.18) | robust grasps 12 / 11 / 12 of 12; exact Full 0.0237 on all seeds and 0.0241 on `S_task` (3 seed configurations); weakest seed 1.92 × weight; 70 g: 12 of 12 seeds hold, 100 % retention; continuous ceiling 73 %; simulated three-contact manipulation E = 45.0 primitives per seed (the best of 48 10.2 designs; compliant pad, force-floor); compliant FP 0.07 (386 primitives); two-finger exact score 0.2767, against 0.3256 for the best two-finger design of all 66 (arm A′ start16, which has no three-contact seed), −15 %, and above every Phase 9 design (best 0.2374) |

Criterion v2 tie-break: Full-D18 start0 (`full_d18_start0/`) also passes R1–R5, with the same `S_task` score within the
seed-configuration spread (0.02410 against 0.02411). The frozen tie-break, the larger mean `|S_task|` (11.67 against
10.33), chose start31. Arm B, arm A′ start6 and arm A′ start26 fail R3. The exported copy was re-scored (default
configuration) and reproduces the suite exactly (`selection_numbers.json`); the Phase 10 numbers are in
`phase10_numbers.json`. Renders, MuJoCo replay videos and charts of the selected design are in its `figures/`
(`figures/README.md`; `dkarma/experiments/visualize_design.py`). The original criterion was applied unchanged to the same candidates and selects nothing. Every
candidate fails requirement 4, which judges three-contact agreement against a two-contact reference; arm B, Full-D18
start0 and arm A′ start6 also fail requirement 3. The two verdicts are not equivalent, and criterion v2 awaits the owner's
review.


## Phase 11 (2026-10-24): inside the servo envelope (D19), criterion v2 selects `selected_full_d19_start2_sobol2_s0.25/`; the original criterion still selects nothing

D19 (SPEC §8.1, §16 item 14; owner's decision 2026-10-23, link 0 included 2026-10-24) makes the XL330 servo case part of
the design box:

* links 0–2 are 16.40 mm capsules at least 34 mm long;
* the fingertips keep 11.8 mm and 12–90 mm;
* the finger bases are at least 33.30 mm apart.

None of the Phase 7–10 designs in this directory is admissible under D19. **The Phase 10 selection
(`selected_full_d18_start31_sobol41_s0.25/`) is superseded for fabrication**; its results stand for the D18 box.

Phase 11 (`docs/phase11_report.md`) re-ran Full KaRMA + Adam inside D19 from 24 starts and evaluated every returned design
with the 10.2 pipeline. It then applied criterion v2 exactly as frozen in Phase 10 (eligibility under D19; R3 against the
D19 nominal start N11, 0.0197; R5 0.239), and the original 9.5 criterion alongside. The two are not equivalent.

| directory | design | status | key numbers (fixed physical task, D19 box) |
| --- | --- | --- | --- |
| `selected_full_d19_start2_sobol2_s0.25/` | Full KaRMA + Adam under D19 from start2 (`workspace/phase11/11.3/full_d19/start2_sobol2_s0.25/`, optimizer seed 0) | **selected by criterion v2** (2 of 24 pass; the runner-up start9 trails by more than the seed-configuration spread); not selected by the original criterion (requirement 3: weakest grasp 0.75 × weight; requirement 4: rigid three-contact FP 0.66 / 0.59 against the two-finger baseline's 0.19 / 0.18) | links 38.5–44.7 mm (servo) and 25.7–31.1 mm (fingertips); bases ≥ 53.9 mm apart; reach 151.8 mm; exact Full 0.0211 on all seeds and 0.0224 on `S_task`; robust grasps 10 / 11 / 8 of 12; simulated three-contact manipulation E = 29.75 per seed (24.1 rotations + 5.7 translations; the best of 24); compliant FP 0.10 (323 primitives); 70 g: 11 of 12 seeds hold, 92 %; **continuous ceiling: 1 of 12 seeds hold**; two-finger exact 0.2621 |

*What the envelope costs, against the Phase 10 selection (descriptive):*

* exact Full score −11 %;
* E −34 % (45.0 → 29.75);
* weakest grasp 1.92 → 0.75 × weight;
* continuous-ceiling retention 73 % → 3 %;
* two-finger score −5 %;
* reach +36 mm.

The exported copy was re-scored (default configuration) and reproduces the suite exactly (`selection_numbers.json`); the
Phase 11 numbers are in `phase11_numbers.json`. Renders, MuJoCo replay videos and charts are in `figures/`
(`figures/README.md`).

Criterion v2 awaits the owner's review. Before fabrication:

* a measured joint-side torque curve and a thermal check of a sustained hold, since the design keeps almost no grasp at
  the continuous estimate;
* the pad silicone's indentation test.
