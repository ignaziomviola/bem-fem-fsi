# The coupling: transfer, partitioned schemes and verification

This document owns the interface operators, the partitioned schemes, the two
resolution constraints of a coupled march, and the coupled verification results
with their measured numbers. The structural formulation is in [FEM.md](FEM.md),
the block boundaries in [ARCHITECTURE.md](ARCHITECTURE.md), and the fluid in
the upstream repository recorded in [FLUID.md](FLUID.md).

Every number below is from `python3 verify_fsi.py`, which takes about thirty
seconds in total. They are actuals.

Last updated 11 August 2026.

## The interface

Three properties are wanted from a transfer pair. All three are obtained by
construction here, not by tolerance, and case C1 measures them at round-off.

### Conservation of force and moment in the lumping

The fluid returns a force per **panel**, acting at the collocation point; the
structure wants forces at nodes. The operator that does that conservatively
already existed on the fluid side and was written for something else entirely.

`thick_panel_wing.centroid_weights` returns the weights of the linear map that
**placed** the collocation point - `build_metric` puts it at the area-weighted
mean of the triangle centroids of both diagonal splits. Those weights are a
partition of unity and they reproduce the centroid exactly. Scattering a panel
force to its four corners with the same weights therefore preserves the total
force, because the weights sum to one, and the total moment about any point,
because `sum_a w_a x_a` is the centroid exactly:

    sum_a w_a (x_a - x_ref) x F = (x_c - x_ref) x F

Upstream, that operator exists to sample the body velocity at the collocation
points for `u_rel = onset - u_body`. Here its **adjoint** is exactly the
conservative load lumping, which means the discrete power of the tractions
against the transferred nodal velocities is exactly the power the fluid sees.
Neither property was designed; both fall out of the fluid architecture's rule
that no surface block ever receives the onset callable.

### Conservation of virtual work across a non-matching interface

Motion crosses the interface through one operator `H` and loads come back
through `H^T`. The virtual work is then identical on the two sides by
construction:

    <f_struct, u_struct> = <H^T f_fluid, u_struct> = <f_fluid, H u_struct>

`H` is built once, on the two **reference** configurations, and never rebuilt.
That is what makes the coupling residual a fixed linear function of the
structural state, which is what quasi-Newton acceleration requires.

Each fluid node is projected onto the nearest structural surface face by a
Newton iteration on the two orthogonality conditions of a bilinear quad, then
clamped to the reference square so that a node overhanging the structure
attaches to the nearest edge instead of extrapolating a shape function outside
its element.

### Exactness for rigid-body motion

The isoparametric map of a bilinear quad is itself bilinear, so a displacement
field linear in `x` - a translation, a rotation, or any combination - is
represented exactly by the bilinear shape functions. A transfer that failed
this would feed spurious strain into a structure that is merely being carried
along, which is the classic way a partitioned code develops a slow drift nobody
can find.

### Welded nodes

The fluid mesh has duplicated nodes: the trailing-edge seam, where row `nwrap`
repeats row 0, and the mirrored tip-pinch pairs. Every member of a welded group
takes the row of its canonical node through `weld_map`, so duplicates receive
identical displacements and `update_points` never has to average a disagreement
into existence, while their loads still accumulate onto the one structural
footprint.

### What does NOT survive a non-matching interface

Total force does, because the weights are a partition of unity. Total **moment**
does not: the load then acts at the projected point rather than at the fluid
node, and the difference is bounded by the projection offset. Case C1 measures
both, on matching and on genuinely non-matching meshes, and `offset_max` is
reported by `build_transfer` rather than hidden behind a tolerance.

The lofted solid foil is a special case worth knowing about: its outer surface
**is** the wetted surface, so the offset is 2.8e-17 and the moment is conserved
exactly too. The fluid mesh pinches its tip stations onto the camber line and
the structure does not, but the pinched tip curve lies inside the structural
tip cap, so even there the projection is exact.

## The coupled time step

```
v = wake_node_velocity(fluid)                          # converged level n
shed(wake, v, dt, v_shed=te_convection_velocity(fluid))    # ONCE per step
truncate(wake, nmax)
u = predictor
repeat:
    update_points(fluid, displaced_points(u), point_velocities(v(u)))
    solve(fluid, wake='frozen')
    dphi_dt = backward_difference(committed levels + this iterate, dt)
    f = structural_forces(get_loads(fluid, rho, dphi_dt))
    u_tilde = step_dynamic(level n, f)                 # pure: level n intact
    u = accelerate(u, u_tilde)
until ||u_tilde - u|| < tol
commit the structural level and the doublet strengths
```

Two things about this had to be checked against the fluid source rather than
assumed, and both are load-bearing.

**`shed` is outside the subiteration loop.** It prepends a wake row *and*
convects every existing node, mutating the wake irreversibly, so a step that
shed once per subiteration would grow the wake by the subiteration count.
Everything inside the loop is `update_points` followed by
`solve(wake="frozen")`, which is deterministic and idempotent at fixed
geometry; `update_points` re-pins wake row zero onto the moved trailing edge on
every call, so the newly shed strip follows the deforming surface without being
shed again. A unit test asserts the wake has exactly one row per step.

**The doublet history advances only on commit.** Otherwise dmu/dt, and with it
the entire added mass, would depend on the path the iteration took rather than
on the time level. The history is seeded with the acyclic level-zero solution -
omitting that seed makes the first step's dmu/dt zero and lags the added mass
by one level for the whole march, which is a defect that produces a
plausible-looking history and was found only by comparing a near-rigid coupled
run against the fluid code's own.

The coupling unknown is the **displacement alone**; the integrator's own
Newmark relations then fix velocity and acceleration exactly, so an accelerated
iterate stays a consistent kinematic state rather than an interpolation of
three of them.

## Acceleration

`FixedPointAccelerator` implements Aitken relaxation, IQN-ILS and a fixed
relaxation. Aitken carries one scalar; IQN-ILS builds a least-squares system
from the differences of successive residuals and outputs.

The Aitken factor is **not** clipped to a small range. The optimal factor for a
fixed point whose dominant eigenvalue is `lambda` is `1/(1 - lambda)`, which is
50 at `lambda = 0.98`, so a clip to plus or minus two removes exactly the
acceleration Aitken exists to supply. There is a magnitude guard against a
degenerate denominator and nothing more.

On a linear fixed point IQN-ILS with full history is a Krylov method and
terminates in at most the dimension of the problem, which a unit test asserts.
Aitken converges steadily but not at that rate, and cannot: one scalar cannot
cure many modes. It works in a partitioned coupling because the added-mass mode
is dominant there. Both statements are gated.

## Two resolution constraints

A coupled march has to resolve two things, and the driver reports and warns on
both.

| measure | threshold | symptom of violating it |
| --- | --- | --- |
| chords convected per step | 0.5 | the near wake is under-resolved; the indicial response is wrong while the steady limit is not |
| steps per structural period | 10 | unresolved modes ring undamped, with tiny displacement and large acceleration |

The second has no counterpart upstream, because a prescribed rigid motion has
no spectrum, and the way it presents does not look like a resolution problem:
**the lift carries a two-step ripple while the quasi-steady lift is perfectly
smooth**. The mechanism is that generalised-alpha at `rho_inf = 1` damps
nothing, so a mode far above the step rings at the Nyquist frequency; its
displacement is negligible but its acceleration is not, and the acceleration
reaches the pressure through dmu/dt. Three cures, in order of preference:
resolve the mode; start from the static aeroelastic equilibrium instead of from
rest; or set `rho_inf` below one. `verify_fem.py` case F8 shows what the third
does - at `rho_inf = 0.8` an unresolved mode is annihilated while a resolved
one is untouched to four decimal places.

## Cost

A deforming surface is never `rigid`, so the upstream rigid fast path - which
removes the 0.63 s body reassembly at 960 panels - is unavailable to a coupled
run by construction. **Every subiteration pays a full body reassembly, a wake
influence and an LU solve.** That is the dominant cost of the coupling and is
the accepted contract, as it is upstream. The structural side is negligible
beside it at these sizes: the operators are factored once and reused across
every step and every subiteration.

The verification cases use 72 to 128 panels for this reason. At 960 panels and
five subiterations a step costs of the order of five seconds.

## Ventilation, and the third time level

`docs/VENTILATION.md` owns the ventilation model; three of its consequences belong
here, because they are properties of the coupled step.

**The regime is a time level.** `commit` advances the structural level, the doublet
strengths and - on a ventilating case - the ventilation regime. The regime obeys
the same rule as the doublet history for the same reason, and for a stronger one:
the flow regimes are bi-stable, so a regime that flipped inside a subiteration
would not merely make the load path-dependent, it would stop the load being a
function of the displacement at all. Everything the cavity iteration computes is
returned in `loads["cav"]` and discarded.

**The cavity fixed point nests inside the coupling subiteration**, cold-started
from the committed extent every call. A warm start would make the converged cavity
depend on the iteration path at the level of the cavity tolerance, and the outer
accelerator differentiates that noise. The cost is bounded: the geometry blocks and
the wake influence are assembled once per subiteration, because the cavity changes
only which columns of D and S are selected, so each cavity iteration costs one
extra dense solve. The doubled mesh costs four times the assembly and eight times
the factorisation, which is why the ventilated verification cases stay at 320
panels or fewer.

**Three new warnings** join the two resolution measures: the Weber number below
250, where surface tension would inhibit inception and this model has none; the
depth Froude number below one, where the free surface deforms steeply and the
image's linearisation fails; and a washout margin within 0.1 of zero, where the
ventilated regime is metastable and a marched run will flip. The last is
bi-stability, not a failure.

## Verification

### C1 Transfer conservation

Solid foil lofted from the wetted surface, so the two surfaces coincide.

| quantity | error |
| --- | --- |
| projection offset | 2.8e-17 |
| total force, panels to fluid nodes | 1.0e-16 |
| total force, fluid nodes to structure | 2.1e-16 |
| total moment, panels to fluid nodes | 4.7e-18 |
| virtual work, the two sides | 3.1e-16 |
| translation transferred | 1.1e-16 |
| rotation transferred | 1.1e-16 |
| general linear field transferred | 1.7e-16 |

The same fluid mesh over a **flat plate** structure, which the wetted surface
genuinely does not lie on: projection offset 6.6e-2, total force still
conserved to 3.1e-16. Force survives a non-matching interface; moment does not,
and the offset is what bounds it.

### C2 Rigid limit

Ten steps at `dt = 0.05`, `rho_inf = 0.5`, started from the static aeroelastic
equilibrium, against `unsteady_wing.time_march` on the same mesh (final
`C_L = 0.328993`).

| E | max displacement | max relative lift difference | mean subiterations |
| --- | --- | --- | --- |
| 2e10 | 2.395e-04 | 2.85e-01 | 4.30 |
| 2e11 | 1.980e-05 | 6.78e-02 | 2.90 |
| 2e12 | 2.050e-06 | 8.11e-03 | 2.50 |

First order in the compliance, as it must be: the displacement is `O(1/E)` and
the lift perturbation is proportional to it. The whole coupled loop - transfer,
integrator, shedding, dmu/dt, acceleration - reduces to the rigid unsteady
answer.

The start matters and is stated rather than buried. A stiff structure released
from rest rings at a frequency no affordable step resolves, which is the second
resolution constraint above, not a defect of either solver.

### C3 Static divergence, by three routes

A wing on grounded springs - the classical elastically supported wing - at 2
degrees, aspect ratio 6, 10% thick. Two of the routes share a model and must
agree with each other; the third is a different model, which is what makes the
comparison a verification.

Measured on the rigid wing by the panel method: `dC_L/dalpha = 4.7267` per
radian, aerodynamic centre at `x_ac = 0.2208` (the quarter chord is at 0.2498),
reference area 6.0000. Measured on the structure by the finite element model:
spring centroid `x_ea = 0.4997`, eccentricity `e = 0.2789`, torsional stiffness
`K_theta = 72.5999`.

| route | q_div | ratio to classical |
| --- | --- | --- |
| classical `K_theta / (S c_l,alpha e)` | 9.1798 | 1.0000 |
| modal: smallest q with `K - qA` singular | 9.0875 | 0.9899 |
| Southwell extrapolation of the fixed point | 9.0876 | 0.9900 |

The fixed point approaching it:

| q/q_div | max vertical displacement | iterations |
| --- | --- | --- |
| 0.30 | 1.430201e-02 | 8 |
| 0.40 | 2.229236e-02 | 8 |
| 0.50 | 3.354335e-02 | 8 |
| 0.60 | 5.058351e-02 | 9 |
| 0.70 | 7.953115e-02 | 11 |

The two coupled routes agree to five figures - they measure the same operator -
and both sit 1.0% below the classical formula, which assumes the whole lift
acts at one chordwise point on a wing twisting rigidly about one axis. The
iteration count rising as `q` approaches divergence is the fixed point ceasing
to contract, which is what divergence is.

**A note on what does not diverge.** The same solid foil clamped at midspan,
without springs, has no divergence within its first six modes. Its torsional
stiffness is far above its bending stiffness - the first modes are bending at
0.06 Hz and the torsional ones are much higher - and bending of a straight
unswept wing produces no twist. That is physics, and it is why the gate uses an
elastically supported wing.

### C4 Added mass in still water

Solid foil, chord 1, span 4, 12% thick, structural density 1200 in water at
1000, so the mass ratio is of order one - the regime a partitioned scheme finds
hardest. Run **without a wake**, which is the right model for added mass: the
fluid repository's own added-mass gate, the accelerating sphere, has none
either.

Added mass over structural mass, three ways:

| route | value |
| --- | --- |
| one dmu/dt difference quotient | 7.1341 |
| two-dimensional strip theory, `rho pi b^2` | 14.8682 |
| implied by the marched wet frequency | 7.4491 |

Wet frequency, from a dry 2.07271 Hz:

| route | value |
| --- | --- |
| from the difference quotient | 0.72675 Hz |
| marched 140 steps at `dt = 0.05` and fitted on the zero crossings | 0.71307 Hz |
| ratio | 0.9812 |
| growth rate of the envelope | -0.0075 per second |

The difference quotient and the march agree to 2% in frequency and 4% in the
ratio, having nothing in common but the two solvers. The envelope is neutral,
which is right: with no wake there is no radiation damping.

Strip theory is the outlier, at twice the measured value, and is expected to
be: it gives every section the two-dimensional value while this mode bends
strongly along the span, where the relief is large. The ratio of the two is the
three-dimensional relief and is **reported, not verified** - there is no closed
form for it on this shape.

### C5 The added-mass instability

Eight steps at `dt = 0.05`, varying only the structural density.

| structural density | added mass / structural | loose | strong + Aitken | strong + IQN-ILS |
| --- | --- | --- | --- | --- |
| 20000 | 0.427 | diverges | 6.6 subiterations | 5.0 |
| 5000 | 1.710 | diverges | 10.0 | 7.9 |
| 1200 | 7.124 | diverges | 17.8 | 10.5 |

A staggered scheme is not merely less accurate below a mass ratio of order one:
it has no stable step size, because the fluid's response to an acceleration of
the surface is instantaneous and proportional to it, so the explicit lag acts
as a negative mass. In water this is the ordinary case, not an extreme one.
`coupling="loose"` exists in the driver so that this table can be produced.

IQN-ILS costs about 60% of Aitken's subiterations at the hardest mass ratio and
the two converge to the same answer.

### C6 Energy balance

Fifteen steps at `dt = 0.05`, `rho_inf = 0.9`. The work done by the fluid,
integrated trapezoidally over the displacement increments, against the strain
plus kinetic energy of the structure: largest difference 4.5e-3 of the largest
work. The work integral is second order in the step, and the residual is that
plus whatever the numerical dissipation removes.

## Open items

- **The still-water wake.** With no mean flow there is nothing to carry shed
  vorticity away from the trailing edge, so a lifting run in still water piles
  the sheet up against the body and eventually destabilises: in one 4-cycle
  test the response was neutral for three cycles and then grew, with the
  subiteration count hitting its cap. This is the roll-up limitation the fluid
  repository already records - no core growth, no sheet regularisation, no
  exclusion shell - reached from an unusual direction. Added-mass work should
  be done without a wake, as C4 is.
- **No flutter case.** Bending-torsion flutter needs the bending and torsion
  frequencies to be comparable, and a solid foil's are separated by more than
  an order of magnitude. Reaching coalescence needs a tuned structure - a
  sprung typical section or a plate with a spar - and that has not been built.
  The machinery is present: `growth_rate` measures the neutral speed and the
  vendored `theodorsen()` supplies a reference.
- **Kinematics are linear on the structural side.** Large deflections of a
  flexible foil are outside the model even though the fluid tolerates them.
- **The coupling is monolithic in cost, not in method.** Every subiteration
  reassembles the body operator; an incremental or reduced-order fluid
  representation would change the cost by a large factor and does not exist.
- **The verification is two-dimensional or steady in its references.** The
  divergence formula and strip theory are two-dimensional; the rigid limit is
  against the code's own fluid solver. There is no three-dimensional unsteady
  aeroelastic analytical solution to test against, and the coupled unsteady
  results are reported as measured rather than as verified.

- A ventilation transition is a load step, so the second resolution constraint
  binds hardest there, and the energy balance of C6 does not close while the cavity
  is growing - entrained air does work this model does not account for. Both are
  measured in `verify_vent.py` case V6 and reported in `docs/VENTILATION.md`.
- A coupled ventilated march should be started from the static equilibrium. At the
  added-mass ratios of a surface-piercing strut a march begun from a
  non-equilibrium state can distort the pinched immersed tip enough on the first
  predictor for the vendored bowtie check to fire.
