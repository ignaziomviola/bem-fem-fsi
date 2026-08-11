# Time-dependent free-wake panel method for wings with thickness

A Python code computes the unsteady loads on a finite wing of finite thickness in incompressible flow, with a wake that is shed at the trailing edge and convected as material fluid, one row per time step. The wing surface is panelled with constant-strength source and doublet panels and the perturbation potential is set to zero just inside the surface, the formulation of Morino; the newest wake strip carries the current trailing-edge potential jump through the Morino Kutta condition and every older strip keeps the strength it was shed with, which is Kelvin's theorem written discretely. The pressure is the unsteady Bernoulli relation, so the added mass of the body is predicted rather than modelled.

The wing may be held in a stream or driven through the fluid: prescribed rigid motions, plunge, pitch and acceleration among them, are supplied with the code, and the surface geometry enters through a single entry point, so a deforming surface from a structural solver fits the same interface.

The steady solver of the companion repository [free-wake-thick-panel-method](https://github.com/ignaziomviola/free-wake-thick-panel-method) is carried over here unchanged in behaviour and remains available: its full test suite and its verification programme pass in this repository. The thin lifting-surface codes both grew from are in [free-wake-lifting-surface](https://github.com/ignaziomviola/free-wake-lifting-surface). The Biot–Savart kernel and the input helpers are duplicated in `freewake_kernels.py` so that a fresh clone runs on its own.

## Requirements

Python 3 with numpy; matplotlib is needed only for the figures, and is imported lazily inside the plotting functions, so importing the solver as a library pulls in no figure backend. scipy is deliberately not used.

```bash
python3 -m pip install --user numpy matplotlib
```

## Quick start

Generate the sample meshes and onset profiles, then march a wing from an impulsive start:

```bash
python3 make_thick_sample_inputs.py
python3 unsteady_wing.py
```

The code prompts for the mesh file, the onset profile, the reference speed, the mean incidence, the motion and the number of steps. Pressing enter accepts the sample inputs. Output is the lift history, the split between the quasi-steady and the added-mass parts of it, the strength of the newest shed strip, the pitching moment, and a figure of six panels ending in the wake seen from the side and in three dimensions.

The steady solver still runs as it did:

```bash
python3 thick_panel_wing.py
```

Both command-line paths end by displaying a figure, so a batch run needs the non-interactive backend:

```bash
printf "naca0012_rect_mesh.npz\nuniform_profile.csv\n1.0\n5\nstart\n60\n" | MPLBACKEND=Agg python3 unsteady_wing.py
```

## Using the code as a library

Importing either module creates no prompt, no print and no figure. A time history takes one call:

```python
import numpy as np
import thick_panel_wing as tpw
import unsteady_wing as usw

points = tpw.load_mesh('naca0012_rect_mesh.npz')
points = tpw.pitch_mesh(points, np.radians(5.0))
z, u = tpw.load_velocity_profile('uniform_profile.csv')
onset = tpw.make_onset(z, u)

motion = usw.heave(points, amplitude=0.05, omega=0.4)      # or usw.still(points)
history, state = usw.time_march(points, onset, motion, dt=0.05, nsteps=200,
                                u_ref=1.0, rho=1025.0)

history['CL'], history['s']            # lift against reduced time 2 U t / c
history['CL'] - history['CL_quasi_steady']    # the added-mass part of it
```

`history` also carries the time, the force and moment vectors, the pitching moment coefficient, the strength of the newest shed strip, the number of wake rows and the root mean square of the material derivative of the doublet strength. `state` is the ordinary solver state at the last level, so every steady diagnostic — `get_loads`, `leakage`, `spanwise_loading` — applies to it unchanged.

### Motions

A motion is any callable `t -> (points, point_velocities)` returning arrays of the mesh shape, carrying a boolean attribute `rigid`. The velocities are analytic, never differenced, and are sampled at the collocation points with the operator that placed them, so a rigid velocity field is transferred without error. Supplied: `still`, `translate`, `accelerate`, `heave`, `pitch` and `pitch_and_heave`. Writing another is four lines.

Setting `rigid` truthfully matters for cost, not only for bookkeeping: the doublet and source blocks are invariant under a rotation and a translation, so a rigid motion skips the reassembly that otherwise dominates a time step. The claim is checked against the enclosed volume and the wetted area on every call.

### A deforming surface

`update_points(state, points, point_velocities)` is the single geometry entry point, unchanged from the steady code apart from the second argument. It takes absolute coordinates of the same wrap topology, rebuilds the metric quantities while the solution and the wake persist, re-pins the attached wake row onto the moved trailing edge, and re-welds the trailing edge and the tip pinch so the body stays watertight. A structural driver builds its transfer operators once from `interface(state)`.

## Method

### What is carried over

The spatial formulation is that of the steady code and is documented in [docs/DESIGN.md](docs/DESIGN.md): source strengths fixed pointwise from the onset before the solve, doublet strengths from setting the interior potential to zero at the panel centroids, a Fredholm equation of the second kind, wake columns folded into the two trailing-edge columns of the matrix at every solve rather than lagged, velocities from the Biot–Savart law with the regularisation of van Garrel, surface velocities from the tangential onset plus the surface gradient of the doublet strength through the dual basis of the stencil directions.

### What time dependence adds

Four things, and they are specified in [docs/UNSTEADY.md](docs/UNSTEADY.md).

The body moves through an inertial frame in which the onset is fixed, and the boundary condition sees the difference: the relative velocity at each collocation point is the onset minus the local surface velocity. A wing advancing into still fluid and a wing held in a stream are therefore both exact, and the sheared onset survives motion because it is still sampled pointwise.

The wake grows by one strip per step. The fluid particle at the trailing edge convects with the local velocity and becomes the downstream edge of the new strip; the strip takes the current trailing-edge potential jump and never changes it again. There is no shortening fraction and no relaxation, because the nodes are material points and the transient is the answer being sought.

The velocity that convects the newly shed vorticity is taken from the mean of the upper and lower surface velocities, not from evaluating the kernels at the trailing-edge node. Those nodes lie exactly on the perimeter rings and the source-panel edges where the kernels are singular; the regularised value there was measured at 0.08 of the onset speed where the answer is the onset speed itself.

The pressure gains the material rate of change of the doublet strength, which is legitimate because the panel numbering is fixed by the mesh format and so panel *h* is the same material panel at every time level. That single term carries the whole added mass of the body: it predicts the added mass of a sphere to 0.16%, and there is no separate model.

## Verification

All values below were produced by the committed code and are tabulated with their conditions in [docs/UNSTEADY.md](docs/UNSTEADY.md). Density and reference speed are unity and the section chord is one.

| Case | Result |
| --- | --- |
| Added mass of a sphere accelerated from rest in still fluid, 400 panels | Force −2.0910 against the exact −(2/3)πρa³a = −2.0944, an error of 0.16%, and constant in time to 1.4×10⁻⁴ under constant acceleration |
| Steady limit: a stationary wing marched to a settled wake | Aspect ratio 8, 80 steps to a reduced time of 40: C_L = 0.43834, against 0.43345 for the steady relaxed wake and 0.44014 for the steady frozen wake of the same length, so within 1.2% of both |
| Wagner: shed circulation after an impulsive start, aspect ratio 20 | Bound circulation follows Wagner's function to a root mean square of 0.0315 over 1 &le; s &le; 16, crossing it between s = 2 and s = 4 |
| Wagner: lift after an impulsive start | Lift runs above Wagner's function, by 0.14 at s = 1 falling to 0.05 at s = 16, the excess being a time-discretisation error that falls monotonically under time refinement |
| Theodorsen: lift in plunge at reduced frequencies 0.1, 0.2 and 0.4 | Amplitude ratios 1.045, 1.088 and 1.202 and phase leads of 8.4&deg;, 10.4&deg; and 10.8&deg; at 48 steps per cycle, against Theodorsen's function with the measured section lift slope |
| Time-step refinement of the indicial response | At 16, 32 and 64 steps per cycle the plunge amplitude ratio at k = 0.2 falls 1.141, 1.104, 1.078 and the phase lead 15.4&deg;, 12.0&deg;, 9.4&deg;; aspect ratio, wake core radius and record length were each tested and excluded, and chordwise resolution contributes weakly |
| Steady verification programme of the companion repository | Reproduced in full: 14 tests including the pinned regressions, and the exact cylinder and Joukowski comparisons |
| Fresh clone from GitHub | Reproduces both test suites, 34 tests in all, and the documented command-line paths, so the pushed copy is self-contained |

## Limitations

The formulation is inviscid and the flow separates only at the trailing edge, so the tip vortex must emerge from the roll-up of the trailing-edge sheet alone and a side-edge Kutta condition remains future work. The Kutta condition equates potential jumps rather than pressures, so a mismatch survives at a swept trailing edge and grows with sweep.

The wake is convected explicitly with a fixed core radius, which limits the roll-up that can be resolved before the sheet becomes ragged; there is no core growth model and no vortex-sheet regularisation beyond the van Garrel core, and no exclusion prevents a wake node from being swept into the body. The cost of a step grows linearly with the number of wake rows, so a long run without truncation grows quadratically in total; truncation is available and closes the cut at infinity, but it discards the starting vortex and is therefore invalid during the indicial transient it would be used to shorten.

A section thinner than about 6% of the chord has a leading-edge radius smaller than the panels that resolve it at the default resolution, and the pressure peak there is a discretisation artefact. The limitations of the steady code in [docs/DESIGN.md](docs/DESIGN.md) apply unchanged.

## Repository layout

| Path | Contents |
| --- | --- |
| `thick_panel_wing.py` | Physics blocks and the steady driver: kernels, geometry, wake, assembly, state interface, loads, plotting, command line |
| `unsteady_wing.py` | Time loop, prescribed motions, history bundle, unsteady figures, command line |
| `freewake_kernels.py` | Biot–Savart kernel and input helpers shared with the companion repositories |
| `make_thick_sample_inputs.py` | Sample meshes and onset profiles |
| `test_thick_panel_wing.py` | Steady unit checks and pinned regressions |
| `test_unsteady_wing.py` | Unsteady unit checks and the analytical gates |
| `verify_analytic.py` | Comparison against the exact cylinder and Joukowski solutions |
| `verify_unsteady.py` | Comparison against added mass, Wagner and Theodorsen |
| `docs/UNSTEADY.md` | Time-dependent formulation, shedding scheme, unsteady verification results |
| `docs/DESIGN.md` | Steady formulation, sign conventions, steady verification results |
| `docs/ARCHITECTURE.md` | Block contracts, state definitions, public signatures, and what the time extension cost |

## Tests

```bash
python3 test_thick_panel_wing.py
python3 test_unsteady_wing.py
UNSTEADY_SLOW=1 THICK_SLOW=1 python3 -m unittest test_thick_panel_wing test_unsteady_wing
```

The fast sets run in about three seconds together and the slow sets in about a minute. They pin the sign conventions of the steady kernels, the array bookkeeping of the wake time step, the transfer of body motion onto the collocation points, the sign and size of the unsteady pressure term, and the recovery of the steady solution in the long-time limit. The slow sets add the pinned steady regressions and the three analytical unsteady gates.

## Licence and citation

Released under the MIT licence, in [LICENSE](LICENSE): the code may be used, modified and redistributed, including in commercial work, provided the copyright notice and the permission notice are retained in any copy or substantial portion.

Retaining that notice is the licence condition. Citation is the academic one, and is asked for rather than compelled: if the code contributes to published work, please cite it. The metadata in [CITATION.cff](CITATION.cff) drives the "Cite this repository" button and yields, in the meantime,

> Viola, I. M. (2026). *time-dependent-free-wake-panel-method: an unsteady free-wake source–doublet panel method for finite wings with thickness*. https://github.com/ignaziomviola/time-dependent-free-wake-panel-method

Should a paper describing the method appear, cite that in preference and this repository for the implementation.
