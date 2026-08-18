"""Build a single self-contained reading page from the manuscript and its figures.

    python3 docs/paper/build_artifact.py

Writes docs/paper/ventilation.html with every figure inlined as a data URI, so the
page needs no network and no LaTeX toolchain. The prose is written here rather than
converted from the LaTeX, because the two have different jobs: the manuscript is
for submission and the page is for reading. Both take their numbers from
`verify_vent.py` and from `make_figures.py`, and neither holds a number of its own.
"""

import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "ventilation.html")


def uri(name):
    with open(os.path.join(FIGS, f"{name}.png"), "rb") as handle:
        return "data:image/png;base64," + base64.b64encode(handle.read()).decode()


def figure(number, name, caption, wide=True):
    return f"""
<figure class="fig{' wide' if wide else ''}">
  <img src="{uri(name)}" alt="Figure {number}">
  <figcaption><span class="fignum">Figure&nbsp;{number}</span>{caption}</figcaption>
</figure>"""


CSS = """
:root {
  --ground: #f5f8f7;
  --surface: #ffffff;
  --ink: #0f1f1d;
  --muted: #5b6b68;
  --faint: #8798954d;
  --rule: #d4e0dd;
  --accent: #0b5d63;
  --accent-soft: #0b5d6314;
  --signal: #8c3a2e;
  --signal-soft: #8c3a2e14;
  --shadow: 0 1px 2px #0f1f1d0f, 0 8px 24px #0f1f1d0a;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #0a1113;
    --surface: #101a1c;
    --ink: #e4edea;
    --muted: #93a5a2;
    --faint: #93a5a233;
    --rule: #1d2b2d;
    --accent: #6cc6cd;
    --accent-soft: #6cc6cd1a;
    --signal: #dd9686;
    --signal-soft: #dd96861a;
    --shadow: 0 1px 2px #0004, 0 10px 28px #0006;
  }
}
:root[data-theme="dark"] {
  --ground: #0a1113;
  --surface: #101a1c;
  --ink: #e4edea;
  --muted: #93a5a2;
  --faint: #93a5a233;
  --rule: #1d2b2d;
  --accent: #6cc6cd;
  --accent-soft: #6cc6cd1a;
  --signal: #dd9686;
  --signal-soft: #dd96861a;
  --shadow: 0 1px 2px #0004, 0 10px 28px #0006;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua",
               Georgia, serif;
  font-size: 17.5px;
  line-height: 1.62;
  -webkit-font-smoothing: antialiased;
}

.page {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  justify-items: center;
  padding: 0 clamp(16px, 5vw, 56px) 96px;
}
.page > * { width: min(100%, 40rem); }
.page > .wide { width: min(100%, 62rem); }

/* sections repeat the grid so that a figure nested inside one can still break out
   to the wider measure; a direct-child rule on .page alone never reaches them */
section {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  justify-items: center;
}
section > * { width: min(100%, 40rem); }
section > .wide { width: min(100%, 62rem); }

/* --- masthead: the rule is the free surface, and the strut crosses it ------ */

header.masthead {
  padding: clamp(48px, 9vh, 104px) 0 0;
}
.eyebrow {
  font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.7rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 18px;
}
h1 {
  font-size: clamp(2rem, 5.2vw, 3.1rem);
  line-height: 1.08;
  font-weight: 600;
  letter-spacing: -0.015em;
  text-wrap: balance;
  margin: 0 0 22px;
}
.byline {
  color: var(--muted);
  font-size: 0.95rem;
  margin: 0 0 4px;
}
.byline a { color: var(--accent); text-decoration: none; }
.byline a:hover { text-decoration: underline; }
.waterline {
  position: relative;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--rule) 8%,
              var(--rule) 92%, transparent);
  margin: 34px 0 0;
}
.waterline::after {
  content: "";
  position: absolute;
  left: 12%;
  top: -13px;
  width: 2px;
  height: 34px;
  background: var(--accent);
  opacity: 0.55;
}

/* --- abstract ------------------------------------------------------------- */

.abstract {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 2px;
  box-shadow: var(--shadow);
  padding: clamp(22px, 3.4vw, 34px);
  margin: 40px 0 0;
}
.abstract h2 {
  font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.7rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 500;
  margin: 0 0 14px;
}
.abstract p { margin: 0; }

/* --- sections ------------------------------------------------------------- */

h2.sec {
  display: flex;
  align-items: baseline;
  gap: 0.6em;
  font-size: 1.45rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  text-wrap: balance;
  margin: 68px 0 6px;
  padding-top: 26px;
  border-top: 1px solid var(--rule);
}
h2.sec .n {
  font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.85rem;
  font-weight: 400;
  color: var(--accent);
  flex: none;
}
h3 {
  font-size: 1.06rem;
  font-weight: 600;
  margin: 40px 0 2px;
  text-wrap: balance;
}
p { margin: 1em 0 0; }
p.lead { font-size: 1.06rem; }
em { font-style: italic; }
strong { font-weight: 600; }
code, .num {
  font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.88em;
  font-variant-numeric: tabular-nums;
}

/* --- the evidence chips: the paper's own honesty taxonomy ------------------ */

.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin: 12px 0 0;
}
.chip {
  font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.66rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 3px 8px;
  border-radius: 2px;
  border: 1px solid var(--rule);
  color: var(--muted);
  white-space: nowrap;
}
.chip.exact { color: var(--accent); border-color: var(--accent); background: var(--accent-soft); }
.chip.pct { color: var(--ink); border-color: var(--faint); }
.chip.reported { color: var(--signal); border-color: var(--signal); background: var(--signal-soft); }

/* --- figures -------------------------------------------------------------- */

figure.fig {
  margin: 40px 0 0;
  padding: 0;
}
figure.fig img {
  display: block;
  width: 100%;
  height: auto;
  /* the plates are saved with a white ground baked in, so the padding is white in
     both themes: a white plate on a dark page reads as printed, while a dark
     padding around a white plot leaves a visible seam */
  background: #fff;
  border: 1px solid var(--rule);
  border-radius: 2px;
  padding: 10px;
}
figcaption {
  font-size: 0.86rem;
  line-height: 1.5;
  color: var(--muted);
  margin: 12px auto 0;
  max-width: 40rem;
}
.fignum {
  font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent);
  margin-right: 0.6em;
}

/* --- tables --------------------------------------------------------------- */

.tablewrap { overflow-x: auto; margin: 36px 0 0; }
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 0.87rem;
  font-variant-numeric: tabular-nums;
}
caption {
  caption-side: top;
  text-align: left;
  font-size: 0.86rem;
  color: var(--muted);
  padding: 0 0 10px;
}
caption .fignum { color: var(--accent); }
th, td {
  padding: 7px 12px 7px 0;
  text-align: right;
  border-bottom: 1px solid var(--rule);
  white-space: nowrap;
}
th:first-child, td:first-child { text-align: left; white-space: normal; }
thead th {
  font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.7rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  font-weight: 500;
  color: var(--muted);
  border-bottom: 1px solid var(--ink);
}
tbody tr:last-child td { border-bottom: 1px solid var(--ink); }
td.zero { color: var(--accent); }

/* --- equation display ----------------------------------------------------- */

.eq {
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: center;
  gap: 16px;
  margin: 26px 0 0;
  padding: 16px 18px;
  background: var(--surface);
  border-left: 2px solid var(--accent);
  border-radius: 0 2px 2px 0;
  overflow-x: auto;
}
.eq .body {
  font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.92rem;
  line-height: 1.7;
}
.eq .tag {
  font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.78rem;
  color: var(--muted);
}

/* --- callout for limitations --------------------------------------------- */

.limit {
  margin: 30px 0 0;
  padding: 16px 18px;
  border-left: 2px solid var(--signal);
  background: var(--signal-soft);
  border-radius: 0 2px 2px 0;
  font-size: 0.95rem;
}
.limit .label {
  font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--signal);
  display: block;
  margin-bottom: 6px;
}
.limit p:first-of-type { margin-top: 0; }

ul, ol { margin: 1em 0 0; padding-left: 1.3em; }
li { margin: 0.4em 0 0; }

footer.colophon {
  margin: 80px 0 0;
  padding-top: 26px;
  border-top: 1px solid var(--rule);
  font-size: 0.85rem;
  color: var(--muted);
}
footer.colophon code { font-size: 0.82em; }

a:focus-visible, img:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 3px;
}
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
"""


BODY = """
<header class="masthead">
  <p class="eyebrow">Partitioned fluid&ndash;structure interaction &middot; ventilation</p>
  <h1>Ventilated cavities on a flexible surface-piercing hydrofoil</h1>
  <p class="byline">I.&thinsp;M. Viola &middot; School of Engineering, Institute for
     Energy Systems, University of Edinburgh</p>
  <p class="byline">Validated against Harwood, Young &amp; Ceccio,
     <em>J. Fluid Mech.</em> <strong>800</strong> (2016) 5&ndash;56</p>
  <div class="waterline"></div>
</header>

<div class="abstract">
  <h2>Abstract</h2>
  <p>Atmospheric ventilation limits the load a surface-piercing hydrofoil can
  carry: air entrained into separated suction-side flow forms a gas cavity that
  removes most of the suction, and the wetted and ventilated states coexist.
  Predictive models are sectional relations closed by a lifting line: they resolve
  neither the chordwise pressure nor the cavity thickness, and hold the hydrofoil
  rigid. The present paper asks whether it can instead be predicted inside a
  partitioned fluid&ndash;structure solver at a fidelity that keeps the coupling
  well posed. To this end a free-wake source&ndash;doublet panel method is extended
  by an exact linearised free surface, a negative image obtained by mesh doubling
  with the source strengths sign-flipped on the image half; by a cavity whose
  pressure, length and thickness are solved as a mixed doublet&ndash;source
  boundary-value problem; and by a hysteretic regime machine that advances only on
  commit. The free-surface condition holds to <span class="num">1.7&times;10<sup>&minus;16</sup></span>
  of the doublet scale, and the image antisymmetry, solved for rather than imposed,
  to <span class="num">2.4&times;10<sup>&minus;14</sup></span>. Ventilation removes
  44&thinsp;% of the lift and moves the centre of pressure to
  <span class="num">0.19c</span>, against the <span class="num">3c/16</span> of a
  supercavitating section. The computed mean cavity closure angle lies within
  4&thinsp;% of the value measured at washout, although its aspect-ratio trend does
  not, and bi-stability spans ten degrees of incidence. Softening the strut deepens
  its suction margin by a third and moves the operating point across the boundary
  beyond which a ventilated cavity is sustained, so a lifting surface&rsquo;s load
  limit can be assessed together with its structural response.</p>
</div>

<section>
<h2 class="sec"><span class="n">1</span>Why ventilation resists prediction</h2>

<p class="lead">Surface-piercing hydrofoils carry the loads of sailing craft, of
surface-effect ships and of tidal-turbine support structures, and their load limit
is set not by stall but by ventilation. Air drawn from the free surface into
separated flow on the suction side forms a gas cavity at, or close to, atmospheric
pressure; the suction that the cavity replaces is lost, and with it up to
70&thinsp;% of the lift.</p>

<p>Three features make the phenomenon hard to predict. The cavity pressure is
fixed by the atmosphere while the reference pressure varies with depth, so the
cavitation number is stratified and the cavity is long at the waterline and short
at the tip. The flow regimes are bi-stable: fully wetted, partially ventilated and
fully ventilated flows coexist over overlapping regions of the parameter space, and
which one is realised depends on the history. And formation and elimination are
controlled by different mechanisms &mdash; formation by the connection of separated
flow to the free surface, elimination by the re-entrant jet at cavity closure &mdash;
so a single threshold cannot describe both.</p>

<p>The state of the art in prediction is sectional: the classical linearised cavity
solutions of Acosta and of Tulin, fitted by smooth rational relations, closed
spanwise by an elliptic loading and rescaled to three dimensions by Helmbold&rsquo;s
small-aspect-ratio formula, with the stability of a fully ventilated cavity decided
by the mean angle of its closure line from the horizontal. That model reproduces
the measured regime boundaries and the scale of the lift loss, and it is the
reference against which any richer model must be judged.</p>

<p>What it cannot supply is what a structural designer needs. It gives no chordwise
pressure distribution, so it cannot be integrated to a load on a deforming surface.
It prescribes the cavity as a pressure condition with no thickness, so the
cavity&rsquo;s displacement effect on the circulation is absent. Its closure angle
is a measured input rather than a computed output. Its regime is selected by hand
from the branch the analyst expects. It is steady, so it carries no added mass and
no wake memory. Above all it is one-way: the hydrofoil is rigid, so the twist that
a real strut develops under load &mdash; which changes the incidence and therefore
the quantity that decides inception &mdash; cannot enter.</p>

<p><strong>The research question.</strong> Can atmospheric ventilation &mdash; its
inception, its effect on the loads, and its elimination &mdash; be predicted within
a partitioned fluid&ndash;structure solver at a fidelity sufficient to keep the
coupling well posed, and does the resulting model reproduce the published sectional
relations and measured scalars while supplying the three-dimensional and two-way
information those relations cannot?</p>

<h3>How the evidence is labelled</h3>
<p>Every result below carries one of three labels, and they are not
interchangeable. The discipline is the paper&rsquo;s, not a presentational
device: a number that is an identity of the construction is worth more than a
number that agrees with an experiment through a low-order model, and conflating
them is how a model comes to be trusted where it should not be.</p>
<div class="chips">
  <span class="chip exact">verified at round-off</span>
  <span class="chip pct">agreement, stated per cent</span>
  <span class="chip reported">reported, not verified</span>
</div>
</section>

<section>
<h2 class="sec"><span class="n">2</span>The model</h2>

<p>The fluid is an unsteady source&ndash;doublet panel method with a free wake, in
the Morino formulation, carried unmodified from a separate verified implementation.
The structure is a three-dimensional finite-element solid with incompatible-mode
hexahedra and a generalised-&alpha; integrator. The coupling is partitioned: loads
are lumped from panel centroids to the wetted finite-element surface by a partition
of unity that reproduces the collocation point, so force, moment and virtual work
transfer exactly, and the interface is built once on the reference configuration so
that the coupling residual is a fixed function of the displacement.</p>

<p>Five states carry the computation and only one is new: the ventilation state,
which holds the parameters, the mirror maps and &mdash; advancing only when a time
level is committed &mdash; the regime, the per-station cavity length, the cavity
depth and the mean closure angle. Everything a coupling subiteration computes is
discarded. That separation is not tidiness. A strongly coupled step re-solves the
same time level several times, and a regime that flipped inside the subiteration
would stop the load being a function of the displacement, because the regimes are
bi-stable.</p>

<h3>The free surface as an exact negative image</h3>

<p>At high Froude number the free-surface condition linearises to
<span class="num">&phi;&nbsp;=&nbsp;0</span> on the undisturbed plane, which is
satisfied identically by a flow field antisymmetric about that plane. The condition
is imposed here by geometry rather than by a constraint: the strut is meshed
together with its mirror above the waterline, giving one closed body of span
<span class="num">2h</span> with a full-chord section on the plane and both far tips
pinched, and the source strengths on the image half are reversed in sign. The
doublet antisymmetry is <em>not</em> imposed &mdash; it emerges from the solve, and
that it does is the test that the image is correct.</p>

<div class="eq">
  <div class="body">&sigma;<sub>image</sub> = &minus;&sigma;<sub>real</sub> &compfn; P</div>
  <div class="tag">(1)</div>
</div>

<p>Three properties fall out of the construction rather than being engineered. The
mesh builder is bit-exactly mirror-symmetric, so the symmetry residual is exactly
zero rather than small. The yaw rotation is about the spanwise axis and preserves
the symmetry bit-exactly. And the trailing-edge weld and tip pinch of the geometry
update act per station and commute with the reflection, so the symmetry survives
every geometry update of a coupled march.</p>

<div class="limit">
  <span class="label">The hazard, made precise</span>
  <p>Mirroring leaves the normal onset component unchanged for a flow with no
  spanwise component, so on a doubled mesh the <em>unflipped</em> solve is
  bit-for-bit the rigid-wall answer. A bare call to the underlying solver on a
  ventilating state therefore returns the zero-Froude solution &mdash; quietly, and
  with about twice the lift. For the same reason the lift coefficient of the
  doubled state is a near cancellation and not the answer: resultants are taken
  over the immersed half.</p>
</div>
__FIG_MESH__

<p>The image wake, however, cannot be convected. With the potential antisymmetric
the perturbation velocity reverses under the reflection while the onset does not,
so a mirror-consistent advection would require the perturbation to vanish. Left to
convect, the image wake drifts off the mirror position, the influence matrix stops
commuting with the permutation, and the free-surface condition is lost progressively
over a march. The image wake is therefore <em>placed</em>, not convected: projected
onto the mirror-symmetric set after every shedding step, with the real half
authoritative. The discarded drift has a physical reading &mdash; on the plane the
chordwise and vertical perturbations vanish while the depthwise one does not &mdash;
so what the projection throws away is the linearised wave elevation rate, and it is
reported as the measure of how far below the model&rsquo;s valid Froude range a run
has been pushed.</p>

<h3>The cavity boundary-value problem</h3>

<p>The reference pressure is the local still-water pressure. Against that reference
a wetted panel carries the dynamic pressure alone &mdash; which is what the
underlying solver already computes &mdash; and a ventilated panel carries
<span class="num">C<sub>p</sub> = &minus;&sigma;<sub>c</sub>(z&prime;)</span>.</p>

<div class="eq">
  <div class="body">&sigma;<sub>c</sub>(z&prime;) = &Delta;&sigma; + (z&prime;/h)&thinsp;&middot;&thinsp;2/Fn<sub>h</sub><sup>2</sup></div>
  <div class="tag">(2)</div>
</div>

<p>Two consequences follow. A hydrostatic reference term must be <em>rejected</em>
rather than merely left unset when ventilating, since it would count the
still-water field twice. And the still-water part of the pressure integrates to the
buoyancy of the immersed volume and is deliberately absent from the fluid load, as
are weight and gravity on the structure: both are body forces for a caller to
apply, so a static deflection compared with an experiment is missing both.</p>

<p>On a cavity panel the dynamic condition fixes the pressure, hence the total
surface speed, hence the doublet strength by chordwise integration from
detachment. The unknown on a cavity panel is then the source strength &mdash; the
cavity&rsquo;s transpiration &mdash; and linearised, its integral is the cavity
thickness. The per-panel choice of unknown is a mask over the assembled system, so
the cavity costs one extra solve per iteration and no new kernel. Because the
doublet strength is prescribed on the cavity, the pressure evaluation returns the
cavity pressure by construction, and that the computed pressure coefficient equals
the negative cavitation number is an end-to-end test of the whole cavity path
rather than a tautology.</p>

<div class="eq">
  <div class="body">&part;&mu;/&part;s = V<sub>s</sub> &minus; u<sub>rel</sub>&thinsp;&middot;&thinsp;s&#770;,
    &nbsp;&nbsp;V<sub>s</sub> = &plusmn;&radic;(q<sub>c</sub><sup>2</sup> &minus; V<sub>v</sub><sup>2</sup>)<br>
    t(s) = (1/V<sub>s</sub>)&thinsp;&int;<sub>detach</sub><sup>s</sup>
      [&sigma; &minus; &sigma;<sub>wetted</sub>]&thinsp;ds&prime;</div>
  <div class="tag">(3), (4)</div>
</div>

<p>Three decisions inside the cavity iteration are load-bearing. The cavity is
parameterised by a continuous per-station length with a fractional weight on the
closure panel, never by a boolean mask: a boolean mask makes the load
piecewise-constant in the displacement, the cavity chatters between adjacent
panels, and quasi-Newton acceleration fits its least-squares system to a staircase.
The cavity extent is read off the <em>baseline wetted</em> pressure and never off
the pressure the cavity has itself imposed, because the dynamic condition makes the
latter equal the cavity pressure on the cavity, so the margin would be identically
zero there and the cavity would collapse at every iteration. And the iteration is
cold-started from the committed extent every time, so the converged extent does not
depend on the subiteration path.</p>

<h3>Inception, regimes, elimination</h3>

<p>A panel method has no boundary layer, so separation is an assumption and is made
visible as one: the stall angle and the pressure-recovery fraction are
<em>inputs</em>, they live in the state where they can be inspected, and every case
reports them as inputs. Inception requires three things together &mdash; a
connected region of ventilation-ready flow covering more than a stated fraction of
the immersed suction area, a path from that region to the free surface, and a broken
surface seal &mdash; and the seal breaks at the stall angle or when air is injected.
The air path is seeded on <em>separation at the waterline station</em> rather than
on pressure, because with a negative image the loading goes to zero at the free
surface while the cavitation number is also zero there, so a pressure-seeded flood
fill reads &ldquo;zero below zero&rdquo; exactly where the air must enter and may
never start.</p>

<p>Persistence asks only whether ventilation-ready flow remains, never again
whether the seal is broken. That asymmetry between formation and persistence
<em>is</em> the hysteresis: it is structural, not a threshold with a dead band.
Elimination is decided by the closure-angle criterion, and its scaling is compared
with the semi-theoretical washout boundary, equation&nbsp;(5) below, which the
reference derives from the elliptic loading, the unity-slope closure condition and
the long-cavity relation.</p>

<p>Finally, the cavity front advances at a finite speed, and the limit bounds the
cavity that the <em>load</em> sees rather than only the next step&rsquo;s starting
point. Without it the extent reaches its equilibrium in one step, the load is a step
function of time, and the structural response measures the time step rather than the
flow.</p>
</section>

<section>
<h2 class="sec"><span class="n">3</span>The sectional relations, and their limits</h2>
<div class="chips">
  <span class="chip exact">two analytic limits</span>
  <span class="chip pct">1.1&thinsp;% against Acosta in &Psi;</span>
</div>

<p>The two analytic limits of the cavity lift slope are recovered to machine
precision, <span class="num">a<sub>0</sub>(0) = 6.283185307180</span> against
<span class="num">2&pi;</span> and <span class="num">a<sub>0</sub>(&infin;) =
1.570796326795</span> against <span class="num">&pi;/2</span>, which is the test
that catches a plausible misreading of the constant in the denominator of that
relation. Against Acosta&rsquo;s partial cavity the rational fit is within
1.1&thinsp;% in the cavity parameter over
<span class="num">0.2 &le; L &le; 0.4</span> and 4.0&thinsp;% at
<span class="num">L = 0.5</span>, while in the length itself the same fit is
8.7&thinsp;% low at <span class="num">L = 0.5</span> and 33.5&thinsp;% low at
<span class="num">L = 0.05</span>. The comparison that means anything is the one in
the parameter, because its derivative is large near the branch join; a test written
on the length reads as the failure of a fit that is correct.</p>

<p>Two properties of the fits are recorded because they look like defects and are
not. The lift slope is not monotone: it peaks at <span class="num">7.2022</span> at
<span class="num">L = 0.511</span>, above <span class="num">2&pi;</span>, and
undershoots its own limit by 0.93&thinsp;% near <span class="num">L = 10</span>
before approaching <span class="num">&pi;/2</span> from below. And the length fit
saturates at <span class="num">100.54</span> rather than diverging, because its
denominator has no positive real root. The blend used here across the hodograph
singularity that both classical solutions share is monotone, its largest increment
along increasing parameter being
<span class="num">&minus;4.3&times;10<sup>&minus;4</sup></span>.</p>
__FIG_SECTIONAL__
</section>

<section>
<h2 class="sec"><span class="n">4</span>The free surface is exact</h2>
<div class="chips"><span class="chip exact">all eleven rows</span></div>

<p>The quantities in table&nbsp;1 are identities of the construction rather than
agreements between models. The mesh symmetry residual is exactly zero. The image
doublet strength cancels the mirror of the real one to fourteen decimal places
although it is solved for and not imposed. The potential on the free-surface plane
is <span class="num">1.7&times;10<sup>&minus;16</sup></span> of the largest doublet
strength, and what remains carries no spatial structure &mdash; which is what
figure&nbsp;1b shows. Flipping the sign in equation&nbsp;(1) reproduces the
underlying rigid-wall solve bit-for-bit, which converts the hazard above from a
warning into an asserted property. And the pressure floor never lowers a pressure:
a cavity <em>raises</em> the suction-side pressure, which is why lift is lost, and
asserting the sign on pressures rather than on forces avoids folding in the sign of
the normal.</p>

<div class="tablewrap">
<table>
  <caption><span class="fignum">Table&nbsp;1</span>Quantities verified at
    round-off, at &alpha;&nbsp;=&nbsp;14&deg;, Fn<sub>h</sub>&nbsp;=&nbsp;2.5,
    AR<sub>h</sub>&nbsp;=&nbsp;1, on 160 immersed panels.</caption>
  <thead><tr><th>quantity</th><th>error</th></tr></thead>
  <tbody>
    <tr><td>mirror symmetry of the doubled strut</td><td class="zero">0.0</td></tr>
    <tr><td>image doublet strength plus mirror of the real one</td><td>2.36&times;10<sup>&minus;14</sup></td></tr>
    <tr><td>image source strength plus mirror of the real one</td><td>6.63&times;10<sup>&minus;15</sup></td></tr>
    <tr><td>image wake strength plus mirror of the real one</td><td>2.73&times;10<sup>&minus;14</sup></td></tr>
    <tr><td>potential on the free-surface plane, relative</td><td>1.70&times;10<sup>&minus;16</sup></td></tr>
    <tr><td>unflipped image minus the underlying solve</td><td class="zero">0.0</td></tr>
    <tr><td>total force through the load lumping</td><td>1.51&times;10<sup>&minus;16</sup></td></tr>
    <tr><td>total force through the interface transfer</td><td>1.13&times;10<sup>&minus;16</sup></td></tr>
    <tr><td>total moment through the load lumping</td><td>9.29&times;10<sup>&minus;17</sup></td></tr>
    <tr><td>virtual work across the interface</td><td class="zero">0.0</td></tr>
    <tr><td>largest pressure the cavity lowered</td><td class="zero">0.0</td></tr>
  </tbody>
</table>
</div>

<p>The near cancellation warned of above is quantified here: the immersed-half lift
coefficient is <span class="num">0.17428</span>, while the whole doubled mesh
returns <span class="num">0.00407</span>. The doubled value is not exactly zero
because the pressure is quadratic in a velocity whose tangential part does not
simply mirror; scaling it by two to recover a plausible lift would be a real error
wearing a plausible face.</p>

<h3>The wetted baseline, and what vanishes at the waterline</h3>
<div class="chips">
  <span class="chip pct">10&thinsp;% against Helmbold</span>
  <span class="chip reported">the waterline force does not vanish</span>
</div>

<p>The contrast between a free surface and a rigid wall is the comparison that a
sign error passes every other test. The negative image drives the sectional loading
towards zero at the waterline, the shallowest strip carrying
<span class="num">0.434</span> of the peak; the positive image places the maximum on
the plane, the shallowest strip carrying all of it, and the peak itself being
<span class="num">1.98</span> times the free-surface one. A sign error therefore
roughly doubles the lift while leaving every other diagnostic intact.</p>

<p>Refinement separates two statements that are easy to conflate. The circulation
of the shallowest strip falls as
<span class="num">&Delta;y<sup>0.74</sup></span>, from <span class="num">0.549</span>
to <span class="num">0.118</span> of its maximum as the immersed half is refined
from four to thirty-two strips, so the loading vanishes at the waterline in the
limit as the exact condition requires. The strip&rsquo;s pressure-integrated
sectional force, in contrast, converges to <span class="num">0.42</span> of the peak
and does <em>not</em> vanish: with an even station count no panel straddles the
plane, and the strip retains a non-circulatory contribution from the large depthwise
perturbation velocity there. The exact statement is the one carried by the
potential; the panel loading vanishes only in the limit.</p>

<p>The measured lift slope of the wetted strut is <span class="num">1.3453</span>,
which is <span class="num">0.900</span> of Helmbold&rsquo;s value at the immersed
aspect ratio and <span class="num">0.506</span> of the value at twice it. The
effective aspect ratio of a surface-piercing strut is the immersed one, because the
negative image makes the waterline behave as a tip; using twice it is the
rigid-wall image and moves the lift slope by tens of per cent while looking
plausible.</p>
__FIG_DEPTH__
</section>

<section>
<h2 class="sec"><span class="n">5</span>The cavity: planform, thickness, pressure</h2>
<div class="chips">
  <span class="chip pct">dynamic condition, 2&times;10<sup>&minus;3</sup> median</span>
  <span class="chip reported">thickness field</span>
</div>

<p>The cavity is longest at the waterline and shortest at the tip, reaching the
trailing edge at the surface and <span class="num">0.09c</span> at the deepest
station, because the cavitation number is stratified. Both sectional reference
models give the same shape and a longer cavity near the surface &mdash; the lifting
line reaching <span class="num">35.5c</span> and the Helmbold chain
<span class="num">7.6c</span> where the panel method saturates at the trailing edge.
Near the waterline the sectional length is formally indeterminate, since the
cavitation number and the sectional incidence vanish together, and the panel
method&rsquo;s saturation is what it can legitimately say.</p>

<div class="limit">
  <span class="label">The thickness is qualitative, and reported as such</span>
  <p>The thickness vanishes at detachment and grows aft, reaching
  <span class="num">0.15c</span> at the shallowest of the three stations plotted,
  and the cavity encloses <span class="num">4.3&times;10<sup>&minus;2</sup></span>
  chords cubed over the immersed half. But a third of the cavity panels carry a
  slightly negative thickness, the smallest being
  <span class="num">&minus;0.012c</span>, which a linearised transpiration integral
  does not forbid but a cavity does; and the closure residual reaches
  <span class="num">0.87</span> of the local scale rather than vanishing, which is
  why the pressure rule and not the thickness rule sets the extent by default. The
  thickness is used here to show that the cavity has a displacement effect at all
  &mdash; which a prescribed-pressure sectional cavity does not &mdash; and not as a
  predicted cavity shape.</p>
</div>
__FIG_PLANFORM__

<p>The chordwise pressure is the end-to-end test of the cavity path. At each of
three depths the wetted suction peak of <span class="num">&minus;3.05</span> is
replaced by a plateau at the local cavity pressure, ending in the pressure recovery
at closure, and the ventilated and wetted distributions coincide aft of closure. The
plateau is stratified with depth, which is the test that catches using the mean
immersion in place of the local depth. On interior cavity panels the median
departure from the cavity pressure is
<span class="num">2&times;10<sup>&minus;3</sup></span> in the pressure coefficient
and the largest is <span class="num">0.18</span>, the largest values occurring on
the panel adjacent to detachment where the surface-gradient stencil reaches across
the leading edge into wetted flow. That departure is a property of a
surface-gradient pressure evaluation and does not reach the load, which imposes the
cavity pressure directly.</p>
__FIG_PRESSURE__
</section>

<section>
<h2 class="sec"><span class="n">6</span>Loads, and the centre of pressure</h2>
<div class="chips">
  <span class="chip pct">3c/16 at 14&ndash;18&deg;</span>
  <span class="chip reported">magnitude of the lift loss</span>
  <span class="chip reported">drag continuity</span>
</div>

<p>Ventilation removes about 44&thinsp;% of the lift over the whole incidence range
examined. The sign, the ordering and the growth of the absolute gap with incidence
reproduce the measured behaviour, and the magnitude is bounded above by the
70&thinsp;% loss reported in the literature; the magnitude itself is reported and
not verified, because the mechanism this model lacks &mdash; the closure region,
where the experiment&rsquo;s re-entrant jet removes suction over an area a
prescribed-pressure cavity cannot place &mdash; acts in the direction of the
residual difference.</p>

<div class="tablewrap">
<table>
  <caption><span class="fignum">Table&nbsp;2</span>Wetted and fully ventilated
    loads at Fn<sub>h</sub>&nbsp;=&nbsp;2.5, AR<sub>h</sub>&nbsp;=&nbsp;1, on 128
    immersed panels. The centre of pressure <em>e</em> is measured forward of
    mid-chord.</caption>
  <thead><tr>
    <th>&alpha; [deg]</th><th>C<sub>L</sub> wet</th><th>C<sub>L</sub> vent</th>
    <th>ratio</th><th>C<sub>D</sub> ratio</th><th>e wet</th><th>e vent</th>
    <th>L<sub>c</sub>/c</th><th>D/h</th><th>&Phi;&#772; [deg]</th>
  </tr></thead>
  <tbody>
    <tr><td>6</td><td>0.1391</td><td>0.0757</td><td>0.544</td><td>2.778</td><td>0.342</td><td>0.084</td><td>0.904</td><td>0.94</td><td>51.6</td></tr>
    <tr><td>10</td><td>0.2300</td><td>0.1322</td><td>0.575</td><td>2.078</td><td>0.339</td><td>0.144</td><td>0.976</td><td>0.94</td><td>50.4</td></tr>
    <tr><td>14</td><td>0.3183</td><td>0.1779</td><td>0.559</td><td>1.628</td><td>0.335</td><td>0.181</td><td>1.000</td><td>0.94</td><td>49.9</td></tr>
    <tr><td>18</td><td>0.4029</td><td>0.2191</td><td>0.544</td><td>1.361</td><td>0.329</td><td>0.186</td><td>1.000</td><td>0.94</td><td>49.9</td></tr>
  </tbody>
</table>
</div>

<p>The panel method&rsquo;s wetted lift lies below both sectional models, and the
comparison quantifies the classical failure of lifting-line theory at unit aspect
ratio: the lifting line exceeds the Helmbold-corrected chain by a factor
<span class="num">1.38</span> to <span class="num">1.41</span>, which is precisely
why the reference applies the small-aspect-ratio rescale.</p>

<p>The centre of pressure moves forward from <span class="num">0.34c</span> wetted
to <span class="num">0.19c</span> ventilated, against the quarter chord and
<span class="num">3c/16 = 0.1875</span> that the sectional relation gives in those
two limits; at 14&deg; and 18&deg; the ventilated value is
<span class="num">0.181</span> and <span class="num">0.186</span>, landing on
<span class="num">3c/16</span>. The wetted value sits 36&thinsp;% further forward
than the sectional quarter chord, and the mechanism is named rather than fitted:
the loading of a low-aspect-ratio surface-piercing strut is concentrated towards
the leading edge in a way a two-dimensional thin-foil result cannot express.</p>

<p>The scatter about the sectional curve is itself a result. An incidence sweep
barely moves the mean cavity length, which stays between
<span class="num">0.51</span> and <span class="num">0.56c</span> while the centre of
pressure ranges from <span class="num">0.084</span> to
<span class="num">0.206c</span>; a Froude sweep at 14&deg; moves the mean length
from <span class="num">0.21</span> to <span class="num">0.72c</span> and the centre
of pressure from <span class="num">0.32</span> to <span class="num">0.09c</span>,
crossing the sectional curve rather than following it and passing forward of the
supercavitating limit no sectional relation can go beyond. The centre of pressure is
therefore not a function of the mean cavity length alone: it depends on how the
cavity is distributed in depth, which is precisely the three-dimensional information
a sectional blend has no place to hold.</p>
__FIG_LOADS__
</section>

<section>
<h2 class="sec"><span class="n">7</span>The closure angle, computed rather than measured</h2>
<div class="chips">
  <span class="chip pct">4&thinsp;% at the published condition</span>
  <span class="chip reported">the aspect-ratio trend</span>
</div>

<p>Because the cavity length here is solved rather than prescribed, the mean
closure angle is an output, and comparing it with the value measured at washout is a
real comparison and not a circularity. The computed closure line, with the local
re-entrant-jet direction superimposed, is the same construction used to measure the
angle photographically: the cavity closes further aft as the Froude number rises,
and the jet sweeps from upstream towards the tip as the closure line tapers.</p>

<p>At the condition of the published measurement &mdash; 20&deg;,
Fn<sub>h</sub>&nbsp;=&nbsp;1.5, unit aspect ratio &mdash; the computed angle is
39.1&deg; to 45.8&deg; across six to sixteen spanwise stations and ten to fourteen
chordwise panels, centred on about 42.5&deg; and insensitive to the chordwise
resolution to 0.4&deg;. The measured mean is 40.75&deg; and the proposed criterion
is 45&deg;, so the agreement is 4&thinsp;% at the published condition.</p>

<div class="limit">
  <span class="label">The mean agrees; the trend does not</span>
  <p>Evaluated at each case&rsquo;s own predicted washout Froude number &mdash;
  which is where the angle was measured &mdash; the mean over four incidences and
  three aspect ratios is 41.08&deg; against the measured 40.75&deg;, agreement to
  0.8&thinsp;%. That conceals a disagreement that matters more. The computed angle
  is 42.2&deg; to 46.6&deg; at unit aspect ratio, but 19.9&deg; to 21.4&deg; at
  half and 57.4&deg; to 59.9&deg; at one and a half &mdash; a factor of three across
  the range, where the measurement found the angle tightly clustered for every
  incidence and aspect ratio.</p>
  <p>The mechanism is geometric and can be named exactly. The angle is taken between
  the closure line and the flow, so its tangent is the ratio of the cavity&rsquo;s
  depth extent to its chordwise spread; the depth extent is the immersion, while the
  chordwise spread is set by the cavitation number and the sectional incidence,
  neither of which depends on the immersion at fixed Froude number and incidence.
  The angle therefore inherits the aspect ratio directly. That the measured angle
  does not implies the physical cavity length scales with immersion, which this
  model does not reproduce, and the 45&deg; criterion is consequently trustworthy
  here only near the aspect ratio at which it was calibrated.</p>
</div>
__FIG_CLOSURE__
</section>

<section>
<h2 class="sec"><span class="n">8</span>Washout, the regime map, and bi-stability</h2>
<div class="chips">
  <span class="chip exact">boundary against its own derivation</span>
  <span class="chip reported">placement in the incidence&ndash;Froude plane</span>
  <span class="chip reported">width of the bi-stable band</span>
</div>

<div class="eq">
  <div class="body">Fn<sub>h</sub> = (&pi;/4)&thinsp;&radic;[ (&pi;/2&thinsp;AR<sup>4</sup>
    &minus; 4AR<sup>3</sup> + 18AR<sup>2</sup> + 8AR) /
    (2.31&thinsp;C<sub>L</sub>(&pi;AR<sup>3</sup> &minus; 2&pi;AR<sup>2</sup>
    + 3&pi;AR + 4)) ]</div>
  <div class="tag">(5)</div>
</div>

<p>The washout boundary is recovered from the chain of relations it was derived
from to a largest relative difference of
<span class="num">2.1&times;10<sup>&minus;16</sup></span> over aspect ratios from a
half to three and lift coefficients from <span class="num">0.2</span> to
<span class="num">0.8</span>. That verifies the transcription and the algebra of
five relations at once, and it verifies nothing about the physics; the wording that
follows is that the boundary is verified against the relations it was derived from,
never that the washout boundary is verified. The older empirical boundary bounds it
from above everywhere, by a factor <span class="num">2.86</span> at unit aspect
ratio and <span class="num">C<sub>L</sub> = 0.5</span>, which is the published
finding that it bounds all the data from above.</p>

<div class="limit">
  <span class="label">What the regime label can and cannot say</span>
  <p>The boundary between wetted and ventilated flow is the stall band and is
  independent of the Froude number, reproducing the vertical boundary of the
  published regime map; it is vertical here because inception is gated by the
  modelled surface seal, and that gate is a function of incidence alone. The
  Froude-dependent boundary of the published map is the washout boundary, and this
  model expresses it through the closure angle and equation&nbsp;(5) rather than
  through the regime label &mdash; because the label does not reach the fully
  ventilated state at these conditions. That classification requires the cavity to
  reach the tip exactly, and on a mesh whose immersed tip is pinched the cavity
  covers 0.92 to 0.94 of the immersion, so a state in which nine tenths is
  ventilated and half the lift is gone is still labelled partially ventilated.
  Fully ventilated results elsewhere are obtained by imposing that branch, which is
  legitimate precisely because the branches are bi-stable and the caller
  chooses.</p>
</div>
__FIG_WASHOUT__

<p>Sweeping the incidence up from a wetted state and back down from a ventilated
one, carrying the state from one point to the next, gives a closed loop. The
ascending branch stays wetted to 14&deg; and ventilates at 16&deg;, one step above
the stall angle that gates the seal; the descending branch stays ventilated all the
way back to 4&deg;. The two branches differ over ten degrees and the loop encloses
an area of <span class="num">1.12</span> in lift coefficient times degrees. At
identical incidence, Froude number and mesh the lift is <span class="num">0.144</span>,
<span class="num">0.238</span> and <span class="num">0.307</span> on the wetted
branch against <span class="num">0.078</span>, <span class="num">0.134</span> and
<span class="num">0.165</span> on the ventilated one; only the history differs. The
bi-stability is structural, not a threshold with a dead band, because inception asks
for an air path and a broken seal while persistence asks only for
ventilation-ready flow.</p>
__FIG_HYSTERESIS__
</section>

<section>
<h2 class="sec"><span class="n">9</span>The coupled response</h2>
<div class="chips">
  <span class="chip reported">the transient</span>
  <span class="chip reported">energy balance during growth</span>
</div>

<p>The strut is soft enough for a transition to matter: the dry natural frequencies
are 6.62, 16.24 and 37.36&nbsp;Hz and the added-mass ratio on a frozen cavity is
<span class="num">4.05</span>; the wet fundamental period is resolved by 7.6 steps
at the time step used, above the resolution constraint but not comfortably.
Marching the coupled system from a wetted static equilibrium through an inception
event triggered by injecting air at the junction of the leading edge and the free
surface &mdash; the perturbation route of the experiments, which fires below the
stall angle &mdash; the cavity depth reaches the tip within one step while the
length grows at the imposed rate limit. The lift falls from its wetted equilibrium
of <span class="num">0.2187</span> to a mean of <span class="num">0.1264</span> over
the last third of the march, a ratio of <span class="num">0.578</span> against the
<span class="num">0.575</span> the steady solve gives at the same incidence, so the
marched and steady load losses agree to half a per cent. The tip deflection grows
from <span class="num">1.21</span> to a peak of
<span class="num">2.19&times;10<sup>&minus;3</sup></span> chords, an overshoot of
<span class="num">1.81</span>.</p>
__FIG_TRANSIENT__

<div class="limit">
  <span class="label">Two results about the numerics, not the flow</span>
  <p>The energy balance does not close while the cavity is growing, the largest
  relative residual being of order unity, because entrained air displaces water and
  does work this model does not account for: there is no gas equation of state and
  no entrainment energy. Once the cavity stops growing the balance recovers.</p>
  <p>The ringing separates into a physical part and an open one. The lift oscillates
  at the wet fundamental, resolved by 7.6 steps, and does not decay because the
  model carries no structural damping; its amplitude scales with the sharpness of
  the load step, the standard deviation falling from <span class="num">0.329</span>
  at a growth rate of 0.2 chords of cavity per chord of travel to
  <span class="num">0.038</span> at 0.1 and <span class="num">0.030</span> at 0.05.
  The figure uses the fastest of the three, because only there does the cavity reach
  its equilibrium extent within the march; the price is the visible ringing and two
  isolated steps at which the lift drops out and recovers within one step.
  So far this is ordinary. What is not is that the same oscillation survives making
  the structure rigid: at a modulus of 2&times;10<sup>12</sup>&nbsp;Pa the tip
  deflection falls to <span class="num">6.5&times;10<sup>&minus;7</sup></span>
  chords and yet at a growth rate of 1.0 the lift still ranges over
  <span class="num">&minus;1.50</span> to <span class="num">+0.55</span>. It is not
  the added-mass term, since a quasi-steady evaluation oscillates identically; not
  the steady dependence of the load on the cavity length, which at frozen extents
  falls smoothly; and not the marched wetted case, which is smooth to
  <span class="num">5.7&times;10<sup>&minus;3</sup></span>. The remaining suspect is
  the unsteady pressure of the cavity&rsquo;s own prescribed doublet strength on a
  strip whose cavity has reached the trailing edge. It is recorded as an open item,
  and it bounds the transient results to the growth of the cavity, the depth it
  reaches, and structural responses at a resolved growth rate.</p>
</div>

<h3>Compliance moves the operating point</h3>
<div class="chips"><span class="chip reported">sign and magnitude of the shift</span></div>

<p>This is the result no sectional model can express. At 16&deg;,
Fn<sub>h</sub>&nbsp;=&nbsp;1.23 and unit aspect ratio, reducing Young&rsquo;s
modulus from 2&times;10<sup>10</sup> to 3&times;10<sup>6</sup>&nbsp;Pa increases the
tip deflection from below <span class="num">10<sup>&minus;4</sup></span> to
<span class="num">0.155</span> chords, increases the wetted lift coefficient from
<span class="num">0.3983</span> to <span class="num">0.4481</span>, and deepens the
smallest suction margin from <span class="num">&minus;1.640</span> to
<span class="num">&minus;2.220</span>: the deformation makes the flow 35&thinsp;%
more ventilation-prone by that measure. Because the washout Froude number falls as
the lift rises, the signed distance from the washout boundary changes sign, from
<span class="num">&minus;0.008</span> on the stiff strut to
<span class="num">+0.052</span> on the soft one. A compliant strut sustains a fully
ventilated cavity at a condition where a stiffer one of the same geometry cannot;
the absolute placement of that boundary inherits the uncertainty of
equation&nbsp;(5), so the reported result is the sign and the magnitude of the
shift, not the location of the crossing.</p>
__FIG_TWIST__
</section>

<section>
<h2 class="sec"><span class="n">10</span>What the coupled model adds</h2>

<p>Each row below is a quantity a structural designer needs and a sectional
lifting-line model has no place to put. The list is not an inventory of
features.</p>

<div class="tablewrap">
<table>
  <caption><span class="fignum">Table&nbsp;3</span>Quantities the coupled model
    supplies that a sectional lifting-line model cannot.</caption>
  <thead><tr>
    <th>quantity</th><th>why it is out of reach sectionally</th><th>where</th>
  </tr></thead>
  <tbody>
    <tr><td>free-surface condition verifiable as an identity</td>
        <td>imposed on a spanwise loading, not on a field</td><td>table&nbsp;1</td></tr>
    <tr><td>solved cavity thickness and its displacement effect on the circulation</td>
        <td>the cavity enters as a pressure condition with no thickness</td><td>fig.&nbsp;4c</td></tr>
    <tr><td>chordwise pressure distribution at each depth</td>
        <td>the sectional model gives a length and a slope, not a distribution</td><td>fig.&nbsp;5</td></tr>
    <tr><td>mean closure angle as an output</td>
        <td>measured photographically and supplied to the model as an input</td><td>fig.&nbsp;7</td></tr>
    <tr><td>regime selected by the model, with hysteresis</td>
        <td>the branch is chosen by the analyst</td><td>fig.&nbsp;9</td></tr>
    <tr><td>finite cavity growth rate, unsteady wake memory and added mass</td>
        <td>the model has no time coordinate</td><td>fig.&nbsp;10</td></tr>
    <tr><td>two-way coupling: deformation changes the regime</td>
        <td>the hydrofoil is rigid</td><td>fig.&nbsp;11</td></tr>
    <tr><td>cavity area, volume and entrainment rate</td>
        <td>no cavity volume exists</td><td>&sect;5</td></tr>
    <tr><td>the discarded wave elevation, quantified</td>
        <td>the free surface is not represented as a field</td><td>&sect;2</td></tr>
  </tbody>
</table>
</div>
</section>

<section>
<h2 class="sec"><span class="n">11</span>Conclusions</h2>

<p>Atmospheric ventilation has been added to a partitioned fluid&ndash;structure
solver, and the question asked was whether it can be predicted there at a fidelity
that keeps the coupling well posed. It can. The free surface is an exact negative
image, so the linearised high-Froude condition is an identity of the construction
and holds to round-off rather than to a tolerance; the antisymmetry of the doublet
strength, which is solved for and not imposed, holds to fourteen decimal places and
is the test that the image is right. The cavity is a boundary-value problem in which
the pressure condition prescribes the doublet strength and the source strength is
solved, so the cavity acquires a thickness and the pressure computed on it is a
result. The three flow regimes and the four transitions between them are carried by
a state that advances only when a time level is committed, which is what keeps the
coupled load a function of the deformation and therefore keeps quasi-Newton
acceleration applicable to a bi-stable system.</p>

<p>The model reproduces what the published sectional theory says, to measured
accuracy, and it reproduces the scalars that were measured. Ventilation removes a
little under half of the lift, bounded above by the largest loss reported, and it
moves the centre of pressure to three sixteenths of the chord, the supercavitating
value. The computed mean angle of the cavity closure line is within four per cent
of the angle measured at the instant of washout &mdash; a genuine comparison,
because the cavity length here is solved rather than assumed &mdash; though the
angle inherits the immersed aspect ratio directly where the measurement found it
almost independent of it, so the stability criterion is trustworthy near the aspect
ratio at which it was calibrated and not far from it. Wetted and ventilated states
coexist over ten degrees of incidence at identical conditions, differing only in
history.</p>

<p>Beyond that, the model supplies what a sectional theory cannot: a chordwise
pressure distribution at every depth, a cavity thickness and its displacement
effect, a closure angle as an output rather than an input, a regime chosen by the
model rather than by the analyst, unsteady wake memory and added mass, a finite rate
of cavity growth, an entrainment rate, and &mdash; the result that motivated the
work &mdash; a two-way coupling in which the deformation changes the regime.
Softening the strut increases its loading, deepens its suction margin by a third,
and moves the operating point across the boundary beyond which a ventilated cavity
cannot be sustained. A rigid analysis of the same geometry places it on the other
side.</p>

<p>Four limitations bound these statements. The free surface does not deform, so the
model is a high-Froude linearisation, and the elevation the wake projection discards
is reported precisely so a reader can tell when it is being pushed too far. There is
no boundary layer, so the stall angle and the pressure recovery that decide
inception are inputs, printed as inputs. The energy balance does not close while the
cavity grows. And the lift oscillates at the step scale while a cavity is growing on
a rigid structure, in a way that remains undiagnosed, so transient results are
bounded to the growth of the cavity and the depth it reaches. The immersed tip is
rounded rather than blunt, so base ventilation and the aerated tip-vortex route to
inception are outside the scope.</p>

<p>The significance is that the load limit of a surface-piercing lifting surface can
now be assessed together with its structural response, in one model, at a cost of
the same order as a wetted coupled computation. For the designer of a sailing
hydrofoil, a surface-effect craft or a tidal-turbine support strut, the question is
not whether ventilation occurs at a nominal incidence but whether it occurs at the
incidence the structure actually reaches under load, and whether the ventilated
state that follows can be sustained. Those two questions are coupled, and answering
them requires the two-way model demonstrated here.</p>
</section>

<footer class="colophon">
  <p>Every number on this page is produced by the repository it documents: the
  figures by <code>docs/paper/make_figures.py</code>, the tabulated scalars by
  <code>verify_vent.py</code>, whose output <code>docs/VENTILATION.md</code> also
  tabulates. No experimental data have been digitised; the values attributed to
  Harwood, Young &amp; Ceccio (2016) are those stated in the text of that paper.
  The submission manuscript is <code>docs/paper/ventilation.tex</code>; this page
  is built from it by <code>docs/paper/build_artifact.py</code>.</p>
</footer>
"""

FIGURES = {
    "__FIG_MESH__": (1, "mesh",
        "The doubled surface-piercing strut and the potential it produces on the "
        "free-surface plane, at 14&deg;, Fn<sub>h</sub>&nbsp;=&nbsp;2.5: "
        "(a) the immersed half in black, its image in grey and the plane shaded, "
        "with the span drawn vertically because the span is the depth; (b) the "
        "magnitude of the perturbation potential on the plane, at a level that is "
        "round-off and carries no structure."),
    "__FIG_SECTIONAL__": (2, "sectional",
        "The sectional cavity model against the closed forms it is built from: "
        "(a) cavity length versus the cavity parameter, with Acosta's partial "
        "cavity, Tulin's supercavity, the two rational fits and the blend used "
        "here; (b) the cavity lift slope versus cavity length, with Acosta's slope "
        "and the two analytic limits."),
    "__FIG_DEPTH__": (3, "depth_loading",
        "Sectional loading against depth for the wetted strut at 10&deg;, "
        "Fn<sub>h</sub>&nbsp;=&nbsp;2.5: (a) the free-surface image against a "
        "rigid-wall image and the elliptic shape; (b) the normalised shape at three "
        "spanwise resolutions; (c) the waterline strip's circulation and sectional "
        "force against the strip width."),
    "__FIG_PLANFORM__": (4, "cavity_planform",
        "The solved cavity at 16&deg;, Fn<sub>h</sub>&nbsp;=&nbsp;2.0: (a) the "
        "cavity planform against the two sectional reference models, on a "
        "logarithmic chordwise axis because the sectional cavity leaves the body; "
        "(b) the cavitation number, the cavity lift slope and the sectional "
        "incidence against depth; (c) the solved cavity thickness at three "
        "depths."),
    "__FIG_PRESSURE__": (5, "pressure",
        "Chordwise pressure coefficient on the suction side at 14&deg;, "
        "Fn<sub>h</sub>&nbsp;=&nbsp;2.5, wetted and fully ventilated, with the "
        "local cavity pressure: (a) z&prime;/h&nbsp;=&nbsp;0.19, "
        "(b) 0.56, (c) 0.81."),
    "__FIG_LOADS__": (6, "loads",
        "Loads at Fn<sub>h</sub>&nbsp;=&nbsp;2.5: (a) lift coefficient against "
        "incidence for both regimes, with the two sectional reference models; "
        "(b) the ratio of ventilated to wetted lift, with the largest loss "
        "reported in the literature; (c) the centre of pressure against the mean "
        "cavity length, with the sectional blend and its two limits."),
    "__FIG_CLOSURE__": (7, "closure_angle",
        "The cavity closure line and its mean angle from the horizontal: (a) the "
        "computed closure line at 20&deg; for Fn<sub>h</sub> from 1.15 to 2, with "
        "the affine fit and the local jet direction; (b) the mean closure angle "
        "against incidence for three immersed aspect ratios, each point at its own "
        "predicted washout Froude number, with the proposed criterion and the value "
        "measured at washout."),
    "__FIG_WASHOUT__": (8, "washout",
        "Elimination of the cavity: (a) the washout Froude number against lift "
        "coefficient for three immersed aspect ratios, with the boundary solved "
        "numerically from the relations it was derived from and the older empirical "
        "boundary; (b) the regime reached at each condition, shaded by the fraction "
        "of the immersion the cavity covers, with the stall band and the washout "
        "boundary through this model's own lift."),
    "__FIG_HYSTERESIS__": (9, "hysteresis",
        "Lift coefficient against incidence at Fn<sub>h</sub>&nbsp;=&nbsp;2.5, "
        "swept up from a wetted state and back down from a ventilated one, with the "
        "bi-stable band and the stall band shaded."),
    "__FIG_TRANSIENT__": (10, "transient",
        "The coupled response to an inception event at 10&deg;, "
        "Fn<sub>h</sub>&nbsp;=&nbsp;2.5, with air injected from the first step: "
        "(a) lift coefficient, with the wetted equilibrium; (b) the largest cavity "
        "length and the cavity depth; (c) the tip deflection, against distance "
        "travelled in chords."),
    "__FIG_TWIST__": (11, "twist",
        "The effect of structural compliance at 16&deg;, "
        "Fn<sub>h</sub>&nbsp;=&nbsp;1.23: (a) the tip deflection and the wetted "
        "lift coefficient against Young's modulus; (b) the signed distance from the "
        "washout boundary and the smallest suction margin, against Young's "
        "modulus."),
}


def main():
    body = BODY
    for token, (number, name, caption) in FIGURES.items():
        body = body.replace(token, figure(number, name, caption))
    html = (f"<title>Ventilated cavities on a flexible hydrofoil</title>\n"
            f"<style>{CSS}</style>\n"
            f'<div class="page">{body}</div>\n')
    with open(OUT, "w") as handle:
        handle.write(html)
    print(f"wrote {os.path.relpath(OUT, os.path.dirname(HERE))} "
          f"({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
