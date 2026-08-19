# Ventilation of a surface-piercing wing in a pitch ramp

Ignazio Maria Viola's `bem-fem-fsi` code, branch
`claude/wing-free-surface-model-8h9b59`, driver `vent_ramp.py`.

Two runs are reported, identical but for the depth Froude number: Fn_h = 10 and
Fn_h = 1.

## Summary

A rigid surface-piercing wing ramped from zero to twenty degrees of incidence
over twenty convective times ventilates at the stall angle, and how much lift
it then loses is set by the depth Froude number: three quarters at Fn_h = 10
and one third at Fn_h = 1. Ventilation is
suppressed while the incidence is below the stall angle of 15 degrees, which is
the gate on the free-surface seal; the regime then switches from fully wetted
to partially ventilated within one time step, the cavity front advances at the
imposed rate of 0.2 chords per chord of travel, and the cavity reaches the
trailing edge at t = 20. Half of the immersed surface, which is the whole
suction side, is then dry. The lift coefficient falls from 0.980 at inception
to a median of 0.213 over the last chord of travel at Fn_h = 10, against 0.825
for the wetted baseline evaluated on the same geometry and wake, and the
pressure drag coefficient falls from 0.103 to 0.044. At Fn_h = 1 the cavity
carries a depth-dependent underpressure instead of an effectively atmospheric
pressure, and the same cavity costs a third of the lift rather than three
quarters. Ventilation onset is identical in the two runs, and so is the whole
wetted phase, to the last bit.

Two numerical effects are reported with the results because they are properties
of the discretisation rather than of the flow: the lift spikes each time the
cavity front crosses a chordwise panel, and the march diverges outright if the
front is allowed to advance at the default rate of one chord per chord of
travel.

## Methodology

The fluid is the free-wake source-doublet panel method vendored in this
repository; the free surface is an exact negative image realised by mesh
doubling, so the immersed wing and its mirror form one closed body and the
disturbance potential is antisymmetric about the plane of the undisturbed free
surface. The wing is rigid, so no structural state is built and no coupling
subiteration is needed: the geometry of each time level is the reference wing
rotated to the current incidence and the surface velocity is the rigid-body
velocity of that rotation, so the added mass still reaches the pressure through
the rate of change of the doublet strength. The cavity is closed with the
Dirichlet condition, at atmospheric pressure.

All quantities are nondimensional through the water density, the chord and the
free-stream speed, so one convective time is one chord of travel and the
reference area for the coefficients is the immersed planform, s = hc.

| parameter | symbol | value |
| --- | --- | --- |
| immersed span | h/c | 4 |
| section | | NACA 0012 |
| incidence | alpha | 0 at t = 0, linear to 20 degrees at t = 20 |
| depth Froude number | Fn_h | 10 and 1 |
| stall angle, the seal gate | alpha_s | 15 degrees |
| cavity underpressure | dsigma | 0, atmospheric |
| panels, doubled mesh | | 20 chordwise by 20 spanwise |
| panels, immersed half | | 20 chordwise by 10 spanwise |
| time step | dt | 0.05 |
| time levels | | 400 |
| wake | | free, truncated at 120 rows, six chords |
| cavity front speed | | 0.2 chords of cavity per chord of travel |

The chordwise wrap is cosine-spaced. The spanwise stations are cosine-spaced
within each half of the doubled mesh, so the fine spacing falls both at the
immersed tip and at the waterline, which are the two edges of the immersed
half; a cosine distribution over the whole doubled span would refine the two
mirror tips and coarsen the waterline, which is where the cavity is longest.
The station spacing ranges from 0.10c at the tip and at the waterline to 0.62c
at mid-depth (figure 1, right).

![The doubled mesh at twenty degrees of incidence, one section on the immersed
half, and the spanwise station spacing.](mesh.png)

**Figure 1.** The doubled mesh at 20 degrees of incidence, one section of the
immersed half, and the spanwise station spacing against depth. The free surface
is the plane y = 0.

## Results at Fn_h = 10

Figures 1 to 4 and the numbers of this section are the Fn_h = 10 run; the next
section reports what changes at Fn_h = 1.

### The wetted phase

The lift grows linearly with incidence at 0.061 per degree between five and
fourteen degrees and reaches 0.980 at the last wetted level. The free surface
stays exact throughout: the antisymmetry residual of the doublet strength is
2e-13 and that of the potential on the plane is 3e-15. The waterline drift that
the image projection discards, which is the linearised wave elevation, stays
below 0.018c, so the assumption of an undeformed free surface is consistent
with the solution it produces.

### Inception

The ventilation-ready area, which is the fraction of the suction side where the
wetted pressure lies below the cavity pressure and a path to the free surface
exists, reaches 0.93 of the suction side before inception, but the regime stays
fully wetted because the seal gate is closed below the stall angle. At
t = 15.0, when the incidence reaches 15 degrees, the regime switches to
partially ventilated in one time step. The switch is a property of the model:
the seal is a thin attached layer that a panel method cannot resolve, so it is
taken to break at the stall angle.

### The ventilated phase

The cavity front then advances at the imposed rate and the cavity deepens to
0.99h within one convective time, so the cavity is full-depth and grows in
length alone thereafter.

| t | alpha (deg) | regime | C_L | C_D | dry area | max L_c/c |
| --- | --- | --- | --- | --- | --- | --- |
| 5.0 | 5 | fully wetted | 0.374 | 0.015 | 0 % | 0 |
| 10.0 | 10 | fully wetted | 0.683 | 0.050 | 0 % | 0 |
| 15.0 | 15 | inception | 0.980 | 0.103 | 0 % | 0 |
| 16.0 | 16 | partially ventilated | 0.710 | 0.207 | 10.2 % | 0.20 |
| 17.0 | 17 | partially ventilated | 0.545 | 0.168 | 20.1 % | 0.40 |
| 18.0 | 18 | partially ventilated | 0.458 | 0.145 | 30.0 % | 0.60 |
| 19.0 | 19 | partially ventilated | 0.308 | 0.083 | 39.9 % | 0.80 |
| 20.0 | 20 | partially ventilated | -0.176 | -0.083 | 49.7 % | 1.00 |

The dry area grows linearly with time, as the front speed dictates, and reaches
49.7 % of the immersed surface, which is the whole suction side, at t = 20; the
pressure side stays wetted. Over the last chord of travel the median lift
coefficient is 0.213 and the median pressure drag coefficient is 0.044.
Evaluating the wetted pressure on the same final geometry and wake gives 0.825
and 0.073, so ventilation removes 74 % of the lift and 40 % of the pressure
drag at fixed incidence.

The mechanism is visible in the pressure (figure 4, centre): the wetted suction
peak of C_p = -6.9 at the leading edge is replaced by the cavity condition,
which at Fn_h = 10 is within 0.02 of zero over the whole depth, while the
pressure side is unchanged. The depth loading (figure 4, right) collapses from
a maximum of 0.41 at 2.3c below the free surface to below 0.04 everywhere.

The washout margin falls from 9.6 at inception to 3.5 at t = 20, so the
ventilated state is not close to washing out and the cavity would persist if
the incidence were reduced; the model is bi-stable, so a ventilated state is
reproducible only together with its history.

![The free wake at t = 20, seen from the side and from above.](wake.png)

**Figure 2.** The free wake at t = 20, seen from the side and from above. The
sheet leaves the trailing edge, deflects to leeward, and rolls up at the
immersed tip and at its mirror; it stays flat on the free surface, which the
image enforces.

![Time histories of the loads, the dry and wetted area, the cavity extent, the
regime and the washout margin.](history.png)

**Figure 3.** Time histories over the ramp: lift, dry and wetted area, pressure
drag, cavity length and depth, regime with the ventilation-ready area, and
yawing moment with the washout margin. The thin trace in the load panels is the
value at each time level and the thick trace is its running median.

![The cavity footprint, the pressure it imposes and the depth loading at
t = 20.](cavity.png)

**Figure 4.** The cavity at t = 20: its footprint on the immersed half, the
pressure it imposes at 1.69c below the free surface against the wetted
baseline, and the depth loading of both.

## The effect of the depth Froude number

The Froude number enters this model in one place only, the pressure inside the
cavity: with the cavity open to the atmosphere the cavitation number at depth z
is 2z/(h Fn_h^2), so the cavity pressure coefficient runs from zero at the
waterline to -0.02 at the tip when Fn_h = 10, and from zero to -2.0 when
Fn_h = 1. The wetted problem does not see the Froude number at all, and neither
does the seal gate, which is a function of incidence alone.

The two runs confirm both statements and separate what the Froude number
changes from what it does not.

| quantity | Fn_h = 10 | Fn_h = 1 |
| --- | --- | --- |
| lift over the wetted phase | identical to the last bit | identical to the last bit |
| inception time and incidence | t = 15.0, 15 degrees | t = 15.0, 15 degrees |
| C_L at inception | 0.980 | 0.980 |
| cavity C_p, waterline to tip | 0 to -0.02 | 0 to -2.0 |
| median C_L, last chord of travel | 0.213 | 0.762 |
| median C_D, last chord of travel | 0.044 | 0.197 |
| wetted C_L on the same final state | 0.825 | 1.141 |
| lift removed by the cavity | 74 % | 33 % |
| dry area at t = 20 | 49.7 % | 36.1 % |
| mean cavity length at t = 20 | 1.00c | 0.69c |
| washout margin at inception | 9.58 | 0.06 |
| washout margin at t = 20 | 3.49 | -0.14 |
| cavity iterations per time level | 2.1 | 8.0 |
| load departure at a front crossing, 95th percentile | 0.60 | 0.25 |
| discarded waterline drift | 0.017c | 0.018c |

Three differences matter.

**The cavity costs less lift at low Froude number.** The cavity replaces the
wetted suction with its own pressure, and that pressure is far below ambient at
depth when Fn_h = 1, so the loading is preserved over most of the immersed
span; only the shallow strips, where the cavity pressure approaches atmospheric,
lose their suction. The lift retained over the last chord of travel is 0.762
against 0.213, and the pressure drag is four times larger, because a cavity at
2.0 of underpressure applies a large suction to the leeward face.

**The cavity is shorter and does not close the whole chord.** The mean cavity
length at t = 20 is 0.69c against 1.00c and the dry area is 36.1 % against
49.7 %, although the front reaches the trailing edge at the deepest stations in
both runs. The extent rule closes the cavity where the wetted pressure has
recovered to the cavity pressure, and at Fn_h = 1 that condition is met before
the trailing edge over the shallower half of the span.

**The ventilated state is metastable at Fn_h = 1.** The washout margin is 0.06
at inception and -0.14 at t = 20, against 9.58 and 3.49; the margin changes
sign, so the ventilated branch is on the washout boundary and a marched run may
flip back. The model is bi-stable, and at this Froude number the two branches
are close in the sense that this diagnostic measures, so the Fn_h = 1 result
should be read as one branch of two rather than as the state the flow must take.

The cavity fixed point also works four times harder, at 8.0 iterations per time
level against 2.1, and the load carries a step-to-step oscillation over the
whole ventilated phase rather than isolated spikes.

![Lift, ventilated area and washout margin for the two Froude
numbers.](compare.png)

**Figure 5.** Lift, ventilated area and washout margin against time, for
Fn_h = 10 and Fn_h = 1. The thin traces are the value at each time level and the
thick traces their running medians. The two runs coincide until inception.

## Numerical behaviour that is not physics

**The load spikes at each front crossing.** After inception the lift and the
drag depart from their trend at isolated time levels, by up to 1.79 in C_L, in
13 % of the ventilated levels. The departures occur at max L_c/c = 0.05, 0.15,
0.27, 0.42, 0.57, 0.72 and 0.84, which are the positions of the chordwise panel
collocation points. A crossing switches one panel to the Dirichlet cavity
condition, which steps its doublet strength, and the step reaches the pressure
through the rate of change of that strength. The amplitude falls with the panel
size, so a finer chordwise mesh is the remedy; the running median is the trend.

**The cavity front must be resolved in time.** At the model default of one
chord of cavity per chord of travel, which is 0.05c per time step here, the
same case ran with 40 chordwise by 40 spanwise panels reached the trailing edge
in twenty time steps, the lift swung to -1.55, and the march diverged, giving
C_L = -3e11 at t = 19. The rate used here, 0.2 chords per chord of travel,
gives the smooth histories above. This is the open item recorded in
`docs/VENTILATION.md`, and at this incidence it is fatal rather than merely
inaccurate.

**The Fn_h = 1 run sits on the model's own validity boundary.** The
free-surface image is a high-Froude linearisation on an undeformed plane, and
the driver warns at or below Fn_h = 1, which is exactly where this run sits. The
discarded waterline drift, which is the linearised wave elevation, is 0.018c,
comparable with the Fn_h = 10 run, but the linearisation itself is what is in
question at order-one Froude number, not the size of the term it discards. The
Fn_h = 1 results are therefore the model's extrapolation limit and want a
free-surface solver to confirm.

**The Dirichlet cavity closure loses meaning at Fn_h = 1.** Re-evaluating the
cavity at the final state without the growth limit returns a mean closure angle
of 77.7 degrees, above the threshold at which the re-entrant jet has an upstream
component, and a cavity thickness that overflows. The loads reported above come
from the pressure and are unaffected, but the cavity thickness, its volume and
its closure angle are not results at this Froude number.

**Two cavity diagnostics are degenerate at these extents.** The closure line is
uniform across the span once the cavity is full-depth, so the fit that returns
the mean closure angle is degenerate and reports 90 degrees throughout the
ventilated phase; and the cavity thickness returned by the Dirichlet closure
reaches -0.42 at full-chord extent, with a closure residual of 0.51, so the
displacement effect of the cavity is not self-consistent once the cavity covers
the whole chord. Neither affects the loads reported above, which come from the
pressure, but neither should be quoted as a result.

## Reproducing

```bash
MPLBACKEND=Agg python3 vent_ramp.py --nc 10 --nspan 10 --dt 0.05 --steps 400 \
        --nwake 120 --growth 0.2 --fn 10 --out runs/ramp
MPLBACKEND=Agg python3 vent_ramp.py --nc 10 --nspan 10 --dt 0.05 --steps 400 \
        --nwake 120 --growth 0.2 --fn 1  --out runs/ramp_fn1
MPLBACKEND=Agg python3 vent_ramp.py --nc 10 --nspan 10 --out runs/ramp \
        --replot --compare runs/ramp_fn1
```

The first two commands march and write the four figures and `history.npz`,
which holds every recorded quantity at every time level; the third redraws the
figures from the saved history and state without marching again and adds the
comparison of the two runs. Each march takes 29 minutes on four cores and
checkpoints every ten time levels.
