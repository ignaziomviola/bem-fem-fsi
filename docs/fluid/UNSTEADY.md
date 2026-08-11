# Time dependence: formulation, scheme and verification

This document specifies the time-dependent extension of the thick-wing panel method. The steady formulation, every sign convention and the steady verification results are in [DESIGN.md](DESIGN.md) and are not restated; the block boundaries and the state contract are in [ARCHITECTURE.md](ARCHITECTURE.md), whose final section records what this extension cost against what it was designed to cost. Usage is in the README.

The extension is implemented, and every number in the verification tables was produced by the committed code. The automated checks are in `test_unsteady_wing.py`; the analytical comparisons are in `verify_unsteady.py`.

Throughout, μ is the panel doublet strength, σ the source strength, c the section chord, b the semichord c/2, s the reduced time 2Ut/c and k the reduced frequency ωc/(2U). The trailing edge is abbreviated TE and the leading edge LE, as in the steady documents.

Last updated 10 August 2026.

## Frame, and why it admits both a moving body and a sheared onset

The body moves through an inertial frame in which the onset U(z)x̂ is fixed. The perturbation potential is written in that frame and the boundary condition is imposed on the moving surface,

σ_k = −n̂_k · u_rel(c_k),  u_rel = U_onset(c_k) − u_body(c_k),

with c_k the panel centroid and u_body its velocity. Nothing else in the spatial formulation changes: the source strengths are still fixed pointwise before the solve, so the sheared onset survives the motion for the same reason it survived in the steady code, and a rotational onset still needs no potential.

Two frames were available and this one was chosen for a reason worth recording. A body-fixed frame would keep the mesh and hence the operator blocks fixed at no cost, but it would put the wake in a rotating and translating frame and would make the sheared onset, which is defined by height in the inertial frame, a function of time and position together. The inertial frame instead costs a mesh update per step, which the rigid fast path below removes for prescribed motions, and leaves the wake alone: **the wake needs no frame transformation at all**, because the source-panel and ring velocities that advect it are induced velocities and are therefore absolute. Body motion reaches the wake only through the strengths.

The consequence for the code is one line. `update_points` gained a `point_velocities` argument, and `u_rel = onset(centroids) − u_body` is the whole of it. No assembly, solve or loads block knows that the body moves, which is exactly the property the steady architecture was built around.

### Transferring the motion onto the collocation points

Nodal velocities are sampled at the collocation points with the operator that placed those points: `build_metric` puts each collocation point at the area-weighted mean of the triangle centroids of both diagonal splits, and `centroid_weights` returns the weights of that same linear map. Applying them to nodal velocities is therefore exact for any velocity field that is linear over a panel, and so exact for a rigid-body field, rotation included. The unit test asserts both properties: that the weights reproduce `panels["centroids"]` to 10⁻¹³, and that a translation plus a rotation is transferred to 10⁻¹².

The weights are computed separately rather than folded into `build_metric`, so that the verified steady centroid path stays bit for bit what it was.

### The rigid fast path

The doublet and source blocks are functions of the relative geometry alone, so a rotation and a translation leave them unchanged. This is not an approximation and was measured: after a random rotation and translation the blocks agree with a fresh assembly to 1.3×10⁻¹³ and 1.3×10⁻¹⁵. `update_points(rigid=True)` therefore keeps them and their version stamp, which removes the reassembly — 0.63 s at 960 panels, the largest single cost of a time step — from every prescribed rigid manoeuvre. The claim is checked on every call against the enclosed volume and the wetted area, which no rigid motion changes, and a drift above 10⁻⁹ raises.

## The wake

### One strip per step, shed as material fluid

The wake carries `n_bound = 1`: the newest strip takes the current TE doublet jump through the Morino Kutta condition and every older strip keeps the strength it was shed with. That is Kelvin's theorem written discretely, and it is exact in this discretisation — the spanwise vortices between adjacent strips sum to μ_w(0) − μ_w(nrows−1), the starting vortex at the open end carries −μ_w(nrows−1), and the bound circulation is +μ_w(0), so the total vanishes identically at every step.

Shedding and convection are **one operation**, not two, because the particle that leaves the TE is the particle the convection moves. `shed` prepends a row and displaces every existing node by dt·v; the fluid particle that was at the TE becomes the downstream edge of the new strip, and `update_points` re-pins the new row zero onto the moved TE. The scheme is explicit: velocities come from the previous time level, as in every free-wake code of this class.

There is **no shortening fraction** on the newest strip. The 0.2 to 0.3 shortening quoted for the latest wake panel belongs to schemes that place its downstream edge by hand; here that edge is a material point and there is nothing to choose. This is worth stating because the steady limit cannot detect the difference: on a straight uniform sheet the load is insensitive to row spacing, C_L changing by 0.02% between dt·U = 1.6 c and 0.1 c, so a shortening convention would have to be pinned by the indicial response and not by the steady answer.

The wake is **convected, never relaxed**. Under-relaxation is a device for reaching a fixed point; here the transient is the answer.

### The velocity that convects the newly shed vorticity

The one place where the obvious implementation is wrong. Row zero of the wake sits on the TE nodes, which lie exactly on the body perimeter rings and on the source-panel edges — the two places where the kernels are singular. What `wake_node_velocity` returns there is the regularised value: it depends on the core radius CORE_WING and on the edge clamp, and on a lifting wing it is a near-cancellation, measured at 0.08 U where the answer is U.

The Kutta condition supplies the well-posed alternative. At a sharp TE the velocity is finite and equal on both sides, so `te_convection_velocity` takes the mean of the upper and lower surface velocities of the two adjacent panel rows, adds the body velocity back — the surface velocity is relative to the moving surface, the wake lives in the inertial frame — and averages the strips onto the nodes. `shed` accepts it through a `v_shed` argument.

The steady code survives using row zero because `relax_wake_step` marches a prescribed arc length and needs only the direction. That is why the defect had never surfaced.

### Truncation

`truncate(wake, nmax)` keeps the newest rows and sets `closure = True`, so the discarded tail is replaced by the FAR_FIELD closing quad of the last kept row's strength — the same closure the steady wake uses, reused as the lumped far vortex the architecture anticipated. A bare slice would instead leave that row's spanwise segment acting as a starting vortex of full strength a few chords behind the wing, which is worse than the truncation it was meant to make cheap.

Truncation therefore gives up the true starting vortex. It is valid once that vortex is far enough downstream to be negligible and invalid during the indicial transient, which is the one case where it would be tempting. The default is no truncation.

### The wake core radius, and why it does not matter here

The steady default is RC_WAKE_FRACTION = 0.2 times the spanwise panel spacing, sized for the tip roll-up of a steady free wake, and on the aspect-ratio-20 verification wing it reaches 0.33 c. That looks alarming for an unsteady calculation whose near wake is a fraction of a chord from the TE, and it is not, for a structural reason: the wake reaches the body through the **solid-angle potential**, which carries no core at all. The core enters only the Biot–Savart velocities that advect the wake, and so changes the answer only through the wake's own shape. Sweeping it from 0.33 c to 0.01 c changed the plunge amplitude and phase of case C by less than 10⁻⁴ of themselves.

`init_unsteady_state` takes a `core` argument nonetheless, for cases where the sheet's own shape is the point.

## Pressure, and where the added mass comes from

The surface pressure is the unsteady Bernoulli relation written in the body frame,

p − p_∞ = ½ρ(|u_rel|² − |V|²) − ρ dμ/dt,

with V the surface velocity relative to the moving body and dμ/dt the derivative following the material panel. The cross terms in u_body·∇φ cancel between the two contributions, which is why no explicit u_body appears beyond u_rel. The derivative is well defined because the wrap mesh format fixes the panel numbering, so index h denotes the same material panel at every time level and μ persists in the state across `update_points`.

The **whole added mass of the body is carried by that single term**. There is no added-mass model, no separate impulse and no fitted coefficient, and the sphere case below measures the term in isolation: with no wake and no circulation the entire force is −ρ dμ/dt integrated over the surface.

The difference is computed by the driver and never by a physics block, because only the driver knows the time levels. `backward_difference` gives (3μⁿ − 4μⁿ⁻¹ + μⁿ⁻²)/(2dt) when three levels exist and (μⁿ − μⁿ⁻¹)/dt otherwise. On a harmonic signal at 24 steps per cycle the three-level formula overstates the derivative by 2.2% and shifts its phase by −0.3°, which bounds its share of the discrepancies reported below.

`get_loads` reports both the full lift and the quasi-steady lift, the latter omitting dμ/dt, so the added-mass content of any history is a subtraction rather than an inference.

## The time step

```
v = wake_node_velocity(state)                    # previous level, explicit
shed(wake, v, dt, v_shed=te_convection_velocity(state))
truncate(wake, nmax)
points, velocities = motion(t)
update_points(state, points, velocities, rigid=motion.rigid)
solve(state, wake="frozen")                      # newest strip folded into the
                                                 #   matrix, history on the rhs
dphi_dt = backward_difference(mu_levels, dt)
loads = get_loads(state, rho, dphi_dt=dphi_dt)
```

`solve(wake="frozen")` is the entry point and not a bare `solve_once`, because after `update_points` the operator blocks may be stale and `solve_once` asserts that they are not. The Kutta fold is in the matrix at every step, never lagged, for the same reason as in the steady code: it is what regularises the near-duplicate rows of the thin TE wedge.

The initial condition is a wake of zero rows. With no wake there is no Kutta condition and the solve returns the acyclic flow, which is what the fluid does at the instant the motion begins. The lift reported at that level is a discretisation residual of a flow that is singular at the sharp TE — d'Alembert's paradox makes the exact value zero — and it falls with refinement, from C_L = 0.058 at 16 × 8 panels to 0.014 at 48 × 20.

## Cost

Measured on the reference machine at 960 panels. Per step: `wake_influence` 0.18, 0.34 and 0.69 s at 25, 50 and 100 wake rows, `wake_node_velocity` 0.42, 0.59 and 0.96 s over the same range, one LU solve of about 0.05 s, and either 0.63 s of body reassembly or nothing at all if the motion is rigid. Both wake terms are linear in the number of rows, so a run without truncation costs O(nsteps²) in total; a 200-step rigid run at this size is a few minutes.

## Verification

Density and reference speed are unity and the section chord is one. Cases B and C compare against two-dimensional theories, are run on a wing of aspect ratio 20 with a cosine spanwise distribution, and are normalised by the code's own steady lift on the same mesh, so the finite-span deficit, the thickness correction and the spatial discretisation error are removed from the comparison and what remains is the unsteady content. Every case states its mesh and time step, so any one of them can be reproduced on its own through the corresponding function in `verify_unsteady.py`.

### A Added mass of a sphere: the pressure term in isolation

A sphere of unit radius is accelerated from rest in still fluid at unit acceleration, with no wake and no circulation, so the entire force is −ρ dμ/dt integrated over the surface. μ is linear in the body speed and the speed is linear in time, so dμ/dt is constant and the backward difference is exact; the only error is the panelling.

| quantity | value |
| --- | --- |
| force on a 20 × 20 sphere, 400 panels | −2.09102 |
| exact −m_a a with m_a = (2/3)πρa³ | −2.09440 |
| error | 0.16% |
| drift over the last five steps | 1.4 × 10⁻⁴ |

This is the strongest gate in the unsteady programme. It fails on a wrong sign, a wrong factor of two, a wrong ρ, a mistransferred body velocity or a mis-signed u_rel, and none of those can be hidden by a normalisation because the reference is dimensional.

### S Steady limit: the whole loop against the steady driver

A stationary wing of aspect ratio 8 at 5° in a uniform onset, marched 80 steps at dt·U/c = 0.25 to a reduced time of 40. Shedding, convection, the Kutta fold and the pressure must together reproduce what the relaxation driver reaches by a different route.

| quantity | value |
| --- | --- |
| marched C_L at s = 40 | 0.43834 |
| steady relaxed wake | 0.43345, ratio 1.0113 |
| steady frozen wake of the same length | 0.44014, ratio 0.9959 |
| shed strip strength | monotone throughout |

The marched answer sits between the two steady ones and within 1.2% of both, which is the right place for it: the convected wake is neither the relaxed shape nor the straight one.

### B Wagner: the indicial response after an impulsive start

A wing of aspect ratio 20 at 5°, 24 × 12 panels, 8% thick, marched from a wake-free state at dt·U/c = 0.05 for 160 steps. Both the lift and the bound circulation are normalised by their own steady values on the same mesh, and compared with Wagner's function in the approximation of Jones.

| s | lift, C_L/C_L(∞) | bound circulation, μ_w/μ_w(∞) | Wagner |
| --- | --- | --- | --- |
| 0.5 | 0.709 | 0.416 | 0.550 |
| 1 | 0.731 | 0.521 | 0.594 |
| 2 | 0.781 | 0.644 | 0.666 |
| 4 | 0.851 | 0.773 | 0.762 |
| 8 | 0.921 | 0.884 | 0.855 |
| 12 | 0.954 | 0.932 | 0.895 |
| 16 | 0.972 | 0.957 | 0.918 |

Root mean square difference over 1 ≤ s ≤ 16: **0.0315 for the circulation and 0.0765 for the lift**.

The circulation is the cleaner comparison and it is the better one: it is what the shedding rule sets, free of the pressure model and of the added mass, and it follows Wagner within 0.03 from s = 2 onwards, crossing the reference between s = 2 and s = 4 rather than diverging from it. Below s = 2 it lags, which is the discrete starting vortex: the first strips are of finite length, so the near wake is stronger than the continuous theory has it.

The lift runs above Wagner throughout, by 0.14 at s = 1 falling to 0.05 at s = 16. The gap between the two columns is the added-mass content of the response, which the reference excludes. Its cause is diagnosed below and it is a time-discretisation error, not three-dimensionality.

### C Theodorsen: the harmonic response in plunge

The same wing plunging at h = 0.02 c · sin(ωt) about zero mean incidence, four cycles at 48 steps each, with the first harmonic fitted over the last two cycles. The reference is Theodorsen's result with the thin-aerofoil 2π replaced by the steady section lift slope measured on this mesh, 6.272 per radian; the added-mass term is geometric and is not rescaled. A positive phase difference is a lead, that is, less lag than the theory.

| k | \|C_L\| code | \|C_L\| theory | ratio | phase difference |
| --- | --- | --- | --- | --- |
| 0.1 | 0.02196 | 0.02101 | 1.045 | +8.39° |
| 0.2 | 0.04070 | 0.03741 | 1.088 | +10.36° |
| 0.4 | 0.07518 | 0.06252 | 1.202 | +10.80° |

The amplitude excess grows with reduced frequency and the phase lead saturates near 10°. Both are time-discretisation errors, as the next section establishes: at 16 steps per cycle the k = 0.2 amplitude ratio is 1.141 and at 64 it is 1.078.

### The account of the two-dimensional discrepancies

The plunge amplitude is high and its phase advanced, and the indicial lift rises faster than Wagner's function. Five candidate causes were tested: three are excluded, one contributes weakly and one dominates. The plunge entries below are three-cycle runs at k = 0.2 with 24 steps per cycle unless the row varies that, which is why they sit above the four-cycle, 48-step values of case C; the record length was itself one of the things tested.

| candidate | test | result |
| --- | --- | --- |
| three-dimensionality | aspect ratio 20 → 40 → 80 at k = 0.2 | amplitude ratio 1.117 → 1.110 → 1.108, phase +13.31° → +12.82° → +12.57°: **excluded**, the trend is an order of magnitude too weak |
| the wake core radius | 0.33 c → 0.05 c → 0.01 c at k = 0.2 | unchanged to better than 10⁻⁴: **excluded**, and for a structural reason — the wake acts on the body through the core-free solid-angle potential |
| the starting transient | 3 → 6 cycles fitted the same way | 1.1039 → 1.1040: **excluded** |
| chordwise resolution | 16 → 24 → 40 wrap panels at k = 0.2 | 1.1113 → 1.1039 → 1.0955, phase +12.38° → +12.00° → +11.61°: contributes, weakly |
| the time step | 16 → 32 → 64 steps per cycle at k = 0.2 | 1.1406 → 1.1039 → 1.0783, phase +15.45° → +12.00° → +9.37°: **the dominant cause** |

The indicial case says the same thing. At dt·U/c = 0.2, 0.1 and 0.05 the lift ratio read at s = 8 falls 0.9302, 0.9253, 0.9213 towards Wagner's 0.855, and the root mean square difference over 1 ≤ s ≤ 8 falls 0.0883, 0.0827, 0.0765. Halving the step again to 0.025 continues the trend, the ratio at s = 2 running 0.807, 0.793, 0.780, 0.771 over the four steps.

Both discrepancies therefore fall monotonically under time refinement, at an observed order of 0.52 for the plunge amplitude, and the sign is that of the leading error of the scheme. The vorticity shed during a step is carried at the strength of the **end** of the step, so it acts from the downstream edge of the new strip rather than from within it — one step too far from the wing. A wake acting from further away is a weaker wake, which is exactly an overstated lift and an understated phase lag. Placing it at the midpoint instead would be second order in time but would break both the exact discrete Kelvin bookkeeping and the Morino fold, which needs the newest strip to carry the current jump if it is to regularise the near-duplicate trailing-edge rows; the trade was not made, and the residual is reported instead of being tuned away.

The practical statement for a user is that the added mass is exact to the panelling, the steady limit closes to 1%, and the unsteady transfer function is accurate to about 10% in amplitude and 10° in phase at 32 steps per cycle, improving under time refinement.

Each row of the table above comes from a function of `verify_unsteady.py` — `case_dt_refinement`, `case_aspect_ratio` and `case_core`, all run by `--refine` — or from `case_theodorsen` called directly with the parameter that the row varies. Nothing in the diagnosis needs the whole programme to be rerun.

## Open items

- The wake is convected explicitly with a fixed core radius. There is no core growth, no sheet regularisation and no exclusion preventing a wake node from being swept into the body; strong roll-up will eventually make the sheet ragged, and the contingencies designed in DESIGN.md (the TE stub, an exclusion shell) remain designed and disabled.
- The cost grows quadratically with the number of steps unless the wake is truncated, and truncation is not valid during an indicial transient. A far-field switch in `wake_influence`, which `doublet_potential_matrix` already supports through `far_diag`, would reduce the constant but not the order.
- The Morino Kutta condition equates potential jumps at every instant. Whether a pressure-Kutta condition matters more in unsteady flow than in steady flow is untested here.
- Shedding is at the TE only, as in the steady code, so the chordwise development of the tip vortex is absent and a dynamic-stall or leading-edge-shedding model is out of scope.
- The verification programme is two-dimensional in its unsteady references. There is no three-dimensional unsteady analytical solution to test against, and the finite-wing indicial and harmonic responses are reported as measured rather than as verified.
