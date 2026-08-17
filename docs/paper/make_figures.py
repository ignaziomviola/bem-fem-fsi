"""Figures of docs/paper/ventilation.tex, all computed by this script.

    MPLBACKEND=Agg python3 docs/paper/make_figures.py
    MPLBACKEND=Agg python3 docs/paper/make_figures.py sectional mesh

Writes docs/paper/figures/*.pdf and *.png and prints, under each figure, the
scalars the text quotes. Nothing else in this repository writes a file, so this
script is the one exception and it is confined to the paper.

It contains one thing the library does not: a discrete nonlinear lifting line,
which rebuilds the model of Harwood, Young & Ceccio (2016) section 1.2.2 from the
sectional relations of `vent_section` alone. It lives here rather than in the
library so that the two models being compared share only those closed forms. The
free surface enters it as a negative image, and with a negative image the bound
circulation vanishes at the waterline exactly as it does at a tip, so the immersed
strut is a wing of span h with two tips and its aspect ratio is AR_h = h/c, not
2 AR_h. Getting that wrong shifts the lift slope by tens of per cent and nothing
else in the model notices.

The scalar anchors taken from that paper are values stated in ITS TEXT - the stall
angle, the mean closure angle at washout, the two centre-of-pressure limits, the
largest reported lift loss and the Breslin & Skalak boundary. No experimental data
are digitised anywhere in this script, and no figure of that paper was read to
obtain a number.
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

import numpy as np
import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt

import thick_panel_wing as tpw
import freewake_kernels as fk
import fem_materials as fmat
import fem_solid as fes
import fsi_driver as fsd
import vent_cavity as vcv
import vent_loads as vnl
import vent_mesh as vm
import vent_section as vs
import vent_solve as vsl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
RHO = 1000.0
STRUCT = dict(e_mod=2.0e8, nu=0.3, rho=1200.0)

# values STATED in the text of Harwood, Young & Ceccio (2016); no figure of that
# paper was digitised to obtain any of them
H_STALL = (14.0, 15.0)          # stall angle, degrees, section 5
H_PHI_WASHOUT = 40.75           # mean closure angle at washout, degrees, fig. 8
H_E_WET, H_E_SUPER = 0.25, 3.0 / 16.0     # centre of pressure, section 3.3
H_CL_LOSS = 0.70                # largest reported lift loss, section 1.1
H_FIG8 = dict(alpha=20.0, fn_h=1.5, ar_h=1.0)     # the condition of its fig. 8(a)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.linewidth": 0.6,
    "lines.linewidth": 1.0, "figure.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf and .png")


def note(text):
    print(f"    {text}")


# ------------------------------------------------------- the panel-method case

def strut_case(alpha=16.0, fn_h=2.5, n_c=12, nspan_half=8, h=1.0, chord=1.0,
               regime="FW", nwake=6, unsteady=False, e_mod=None, image=True,
               **vent_kw):
    """A ventilated surface-piercing strut. -> (fluid, sstate, transfer, vent, mesh)"""
    points = vm.strut_mesh(n_c=n_c, nspan_half=nspan_half, h=h, chord=chord,
                           alpha_deg=alpha, image=image)
    maps = vm.mirror_maps(points.shape, image=image)
    mesh = vm.strut_solid_mesh(vm.half_points(points, maps), n_thick=2,
                               min_half_thickness=0.02)
    material = dict(STRUCT)
    if e_mod is not None:
        material["e_mod"] = e_mod
    model = fes.build_model(mesh["nodes"], mesh["elements"],
                            fmat.IsotropicElastic(**material))
    onset = fk.make_onset(np.array([-10.0, 10.0]), np.array([1.0, 1.0]))
    fluid, sstate, transfer, vent = fsd.init_vent_fsi(
        points, onset, model, h=h, chord=chord, u_ref=1.0, rho=RHO,
        alpha_deg=alpha, image=image, unsteady=unsteady,
        config={"lifting": True, "nwake": nwake}, fn_h=fn_h, regime=regime,
        **vent_kw)
    fes.clamp(sstate, mesh["node_sets"]["tip_max"])
    fes.assemble_operators(sstate)
    return fluid, sstate, transfer, vent, mesh


def solved(**kw):
    """One converged ventilated solve. -> (fluid, vent, cav, loads)"""
    fluid, _, _, vent, _ = strut_case(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fluid, cav, loads = vsl.solve_cavity(fluid, vent, rho=RHO)
    return fluid, vent, cav, loads


def settled(max_advance=6, **kw):
    """Solve and commit until the regime stops changing. -> (fluid, vent, cav, loads)

    A single commit can only take a fully wetted state to partial ventilation:
    inception is FW to PV and stabilisation is PV to FV, and each is one time
    level, because the regime advances only on commit. A steady regime map has to
    advance the state until it stops moving, which is the discrete analogue of
    holding the condition and waiting.
    """
    fluid, _, _, vent, _ = strut_case(**kw)
    cav = loads = None
    for _ in range(max_advance):
        was = vent["regime"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fluid, cav, loads = vsl.solve_cavity(fluid, vent, rho=RHO)
        vcv.commit_vent(vent, cav, 0.0, None)
        if vent["regime"] == was:
            break
    return fluid, vent, cav, loads


def strip_depth(pan, vent):
    """Mean depth of each spanwise strip. -> (nspan,)"""
    j_of = np.tile(np.arange(pan["nspan"]), pan["nwrap"])
    depth = vcv.panel_depth(pan, vent["y_fs"])
    return np.array([depth[j_of == j].mean() for j in range(pan["nspan"])])


# ------------------------------------- Harwood's nonlinear lifting-line model

def lifting_line(alpha_deg, fn_h, ar_h, n=16, regime="FV", omega=1.0,
                 max_iter=120, tol=1e-11, length_model="fit"):
    """Discrete nonlinear lifting line with a negative free-surface image.

    The circulation is a Glauert sine series over the immersed span,
    Gamma = 2 h u sum A_m sin(m theta) with z' = (h/2)(1 - cos theta), which
    vanishes at both ends of the interval. That IS the negative image: with the
    potential antisymmetric about the free surface the bound circulation is odd,
    so the waterline carries no circulation and behaves as a tip. The sectional
    lift coefficient follows the cavity length through the lift slope (1.9) of
    `vent_section`, and regime='FW' replaces the cavity length by zero and the
    lift slope by 2 pi, which is what that paper does for wetted flow.

    Solved by a damped Newton iteration on the mode amplitudes, not by successive
    substitution: the induced-angle operator has a gain that grows with the mode
    number, so a Picard iteration diverges on the highest mode and returns an
    alternating distribution that looks like a converged answer with every other
    station empty.

    -> dict with kappa, alpha_2d, length, a0, cl, CL, sigma_c, iterations
    """
    alpha = np.radians(alpha_deg)
    modes = np.arange(1, n + 1)
    theta = modes * np.pi / (n + 1)                 # collocation, both ends free
    kappa = 0.5 * (1.0 - np.cos(theta))             # z'/h, zero at the waterline
    sin_m = np.sin(np.outer(theta, modes))          # (n, n)
    down = sin_m * modes / np.sin(theta)[:, None]   # induced angle operator
    bound = 4.0 * ar_h * sin_m
    sigma_c = vs.sigma_cavity(kappa, 1.0, fn_h) if regime == "FV" \
        else np.zeros(n)

    def sectional(amp):
        """(alpha_2d, cavity length, lift slope, sectional lift) of amplitudes."""
        alpha_2d = np.clip(alpha - down @ amp, 1e-6, 0.5 * np.pi)
        if regime == "FV":
            leng = np.asarray(vs.cavity_length(vs.psi(sigma_c, alpha_2d),
                                               length_model), dtype=float)
            slope = vs.lift_slope(leng)
        else:
            leng, slope = np.zeros(n), np.full(n, 2.0 * np.pi)
        return alpha_2d, leng, slope, slope * np.sin(alpha_2d)

    def residual(amp):
        return bound @ amp - sectional(amp)[3]

    amp = np.zeros(n)
    amp[0] = alpha / (1.0 + 2.0 * ar_h / np.pi)     # elliptic first guess
    res = residual(amp)
    it = 0
    for it in range(1, max_iter + 1):
        jac = np.empty((n, n))
        eps = 1e-7
        for m in range(n):
            probe = amp.copy()
            probe[m] += eps
            jac[:, m] = (residual(probe) - res) / eps
        try:
            step = np.linalg.solve(jac, -res)
        except np.linalg.LinAlgError:
            break
        scale = omega
        for _ in range(30):                         # backtracking line search
            trial = amp + scale * step
            new = residual(trial)
            if np.abs(new).max() < np.abs(res).max():
                break
            scale *= 0.5
        amp, res = trial, new
        if float(np.abs(step).max()) < tol:
            break
    alpha_2d, length, a_0, c_l = sectional(amp)
    return {"kappa": kappa, "alpha_2d": alpha_2d, "length": length, "a0": a_0,
            "cl": c_l, "CL": float(np.pi * ar_h * amp[0]), "sigma_c": sigma_c,
            "iterations": it, "residual": float(np.abs(res).max())}


def harwood_lift(alpha_deg, fn_h, ar_h, regime="FV", n=200, max_iter=600,
                 tol=1e-10, length_model="fit"):
    """Harwood's own three-dimensional lift: the Helmbold chain (1.20), (4.1)-(4.3).

    His model does not integrate a lifting line for the lift. It takes the
    elliptic shape (4.7) for the spanwise distribution, infers the sectional
    incidence from it through the cavity lift slope (1.9), reads the cavity length
    off the sectional relation at the local cavitation number, forms the
    lift-weighted mean slope a0* of (4.6) and (4.8), and rescales that single
    slope to three dimensions with Helmbold's small-aspect-ratio formula at AR_h.
    Every piece is a `vent_section` primitive, so this is that paper's model with
    no reimplemented physics; the loop closes it self-consistently, which the
    paper does by hand.

    -> dict with kappa, length, a0, a_star, alpha_2d, CL, iterations
    """
    alpha = np.radians(alpha_deg)
    kappa = np.linspace(0.0, 1.0, int(n) + 1)
    sigma_c = vs.sigma_cavity(np.clip(kappa, 1e-9, None), 1.0, fn_h) \
        if regime == "FV" else np.zeros_like(kappa)
    length = np.zeros_like(kappa)
    a_0 = np.full_like(kappa, 2.0 * np.pi)
    a_star = 2.0 * np.pi
    c_l = float(vs.helmbold(a_star, ar_h)) * np.sin(alpha)
    alpha_2d = np.full_like(kappa, alpha)
    it = 0
    for it in range(1, max_iter + 1):
        cl_2d = c_l * vs.elliptic_shape(kappa)               # (4.2)
        alpha_2d = np.arcsin(np.clip(cl_2d / a_0, -0.99, 0.99))     # (4.3)
        if regime == "FV":
            length = np.asarray(
                vs.cavity_length(vs.psi(sigma_c, np.maximum(alpha_2d, 1e-9)),
                                 length_model), dtype=float)
            a_0 = vs.lift_slope(length)
            # (4.8): the lift-weighted mean, E of (4.7) having unit integral
            a_star = float(np.trapezoid(a_0 * vs.elliptic_shape(kappa), kappa))
        new = float(vs.helmbold(a_star, ar_h)) * np.sin(alpha)
        if abs(new - c_l) < tol:
            c_l = new
            break
        c_l = 0.5 * (c_l + new)
    return {"kappa": kappa, "length": length, "a0": a_0, "a_star": a_star,
            "alpha_2d": alpha_2d, "CL": float(c_l), "iterations": it}


# ------------------------------------------------------------------- figure 1

def fig_sectional():
    """Cavity length and lift slope against the classical closed forms."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax = axes[0]
    length = np.linspace(0.02, 0.5, 200)
    ax.plot(vs.acosta_psi(length), length, "k-", lw=1.6, label="Acosta (1955)")
    l_s = np.linspace(1.26, 12.0, 200)
    ax.plot(vs.tulin_psi(l_s), l_s, "k--", lw=1.6, label="Tulin (1953)")
    psi = np.logspace(np.log10(0.25), np.log10(30.0), 400)
    ax.plot(psi, vs.harwood_length(psi), color="0.45", ls="-", lw=1.0,
            label="Harwood (1.7)")
    ax.plot(psi, vs.harwood_length_low_order(psi), color="0.45", ls=":", lw=1.0,
            label="Harwood (1.8)")
    ax.plot(psi, vs.cavity_length(psi, "exact"), "k-", lw=0.9, marker="o",
            markevery=40, ms=3, mfc="none", label="present, blended")
    ax.set_xscale("log")
    ax.set_xlim(0.25, 30.0)
    ax.set_ylim(0.0, 6.0)
    ax.set_xlabel(r"$\Psi$")
    ax.set_ylabel(r"$L_{\mathrm{c}}/c$")
    ax.set_yticks([0, 2, 4, 6])
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.03, 0.93, "(a)", transform=ax.transAxes)

    ax = axes[1]
    lg = np.linspace(0.0, 6.0, 400)
    slope = vs.lift_slope(lg)
    ax.plot(lg, slope, "k-", lw=1.6, label="present, (1.9)")
    la = np.linspace(0.0, 0.5, 100)
    ax.plot(la, vs.acosta_lift_slope(la), color="0.45", ls="--", lw=1.4,
            label="Acosta (1955)")
    ax.axhline(2.0 * np.pi, color="0.7", ls=":", lw=0.8)
    ax.axhline(np.pi / 2.0, color="0.7", ls=":", lw=0.8)
    ax.annotate(r"$2\pi$", (3.4, 2.0 * np.pi + 0.3), fontsize=8)
    ax.annotate(r"$\pi/2$", (3.4, np.pi / 2.0 + 0.3), fontsize=8)
    ax.set_xlim(0.0, 6.0)
    ax.set_ylim(0.0, 9.0)
    ax.set_xlabel(r"$L_{\mathrm{c}}/c$")
    ax.set_ylabel(r"$a_0$")
    ax.set_yticks([0, 3, 6, 9])
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.03, 0.93, "(b)", transform=ax.transAxes)
    fig.tight_layout()
    save(fig, "sectional")
    blend = np.asarray(vs.cavity_length(psi, "exact"), dtype=float)
    note("blend monotone in Psi: largest increment %.1e"
         % float(np.diff(blend).max()))
    note("a0 peaks at %.4f at L = %.3f; a0(6) = %.4f against pi/2 = %.4f"
         % (slope.max(), lg[int(np.argmax(slope))], slope[-1], np.pi / 2.0))
    note("(1.7) at Acosta's L = 0.5: L = %.3f, Psi error %.2f%%"
         % (float(vs.harwood_length(vs.acosta_psi(0.5))),
            100.0 * abs(vs.acosta_psi(float(vs.harwood_length(vs.acosta_psi(0.5))))
                        / vs.acosta_psi(0.5) - 1.0)))


# ------------------------------------------------------------------- figure 2

def fig_mesh():
    """The doubled strut, and the potential on the free-surface plane."""
    points = vm.strut_mesh(n_c=10, nspan_half=8, h=1.0, chord=1.0, alpha_deg=14.0)
    maps = vm.mirror_maps(points.shape)
    fig = plt.figure(figsize=(6.6, 2.9))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    nh = maps["n_half"]
    # the span is the DEPTH, so it is drawn on the vertical axis of the view and
    # the free surface is a horizontal plane
    for j in range(points.shape[1]):
        col, wid = ("k", 0.5) if j <= nh else ("0.72", 0.4)
        ax.plot(points[:, j, 0], points[:, j, 2], points[:, j, 1], color=col,
                lw=wid)
    for i in range(0, points.shape[0], 2):
        ax.plot(points[i, :nh + 1, 0], points[i, :nh + 1, 2],
                points[i, :nh + 1, 1], "k-", lw=0.5)
        ax.plot(points[i, nh:, 0], points[i, nh:, 2], points[i, nh:, 1],
                color="0.72", lw=0.4)
    xg, zg = np.meshgrid(np.linspace(-0.7, 1.6, 2), np.linspace(-0.8, 0.8, 2))
    ax.plot_surface(xg, zg, np.zeros_like(xg), color="0.5", alpha=0.3,
                    linewidth=0)
    ax.set_xlabel(r"$x/c$", labelpad=-9)
    ax.set_ylabel(r"$z/c$", labelpad=-9)
    ax.set_zlabel(r"$y/h$", labelpad=-9)
    ax.set_xticks([0, 1]); ax.set_yticks([0]); ax.set_zticks([-1, 0, 1])
    ax.tick_params(pad=-3)
    ax.view_init(elev=16, azim=-64)
    ax.set_box_aspect((1.5, 1.0, 2.1))
    ax.text2D(0.02, 0.95, "(a)", transform=ax.transAxes)

    fluid, vent, cav, loads = solved(alpha=14.0, n_c=10, nspan_half=8)
    pan = fluid["panels"]
    gx, gz = np.meshgrid(np.linspace(-1.0, 2.0, 46), np.linspace(-1.0, 1.0, 34))
    probe = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1)
    d_blk = tpw.doublet_potential_matrix(probe, pan["corners"])
    s_blk = tpw.source_potential_matrix(probe, pan["corners"])
    ref = max(float(np.abs(fluid["mu"]).max()), 1e-30)
    phi = (d_blk @ fluid["mu"] + s_blk @ fluid["sigma"]) / ref
    ax = fig.add_subplot(1, 2, 2)
    scale = float(np.abs(phi).max())
    con = ax.contourf(gx, gz, np.abs(phi).reshape(gx.shape) / scale,
                      levels=np.linspace(0.0, 1.0, 11), cmap="Greys")
    bar = fig.colorbar(con, ax=ax, ticks=[0.0, 0.5, 1.0])
    bar.set_label(r"$|\phi| / \phi_{\mathrm{ref}}$, $\phi_{\mathrm{ref}} = "
                  r"%.1f\times10^{-16}$" % (scale * 1e16))
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$z/c$")
    ax.set_xticks([-1, 0, 1, 2]); ax.set_yticks([-1, 0, 1])
    ax.text(0.03, 0.92, "(b)", transform=ax.transAxes)
    fig.tight_layout()
    save(fig, "mesh")
    res = vsl.antisymmetry_residual(fluid, vent)
    note("phi on the plane: %.2e of max|mu|; mesh symmetry residual %.1e"
         % (scale, vm.symmetry_residual(np.array(pan["points"]),
                                        vent["y_fs"])))
    note("antisymmetry: " + ", ".join(f"{k} = {v:.2e}" for k, v in
                                      sorted(res.items())))
    return scale


# ------------------------------------------------------------------- figure 3

def fig_depth_loading():
    """Sectional loading against depth: free surface, rigid wall, refinement."""
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.6))
    ax = axes[0]
    peak = {}
    for sign, name, col, ls, mk in ((-1.0, "free surface", "k", "-", "o"),
                                    (+1.0, "rigid wall", "0.5", "--", "s")):
        fluid, _, _, vent, _ = strut_case(alpha=10.0, n_c=10, nspan_half=12)
        fluid = vsl.solve_wetted(fluid, vent, wake="frozen", sign=sign)
        fields = vnl.vent_pressure(fluid["panels"], fluid["mu"], fluid["u_rel"],
                                   vent["u_ref"], vent,
                                   np.zeros(len(fluid["mu"])), rho=RHO)
        depth, cl = vnl.depth_loading(fluid["panels"], vent, fields["p_gauge"],
                                      RHO)
        ax.plot(cl, depth, color=col, ls=ls, marker=mk, ms=3, mfc="none",
                label=name)
        peak[name] = (float(cl[0]), float(cl.max()))
    kappa = np.linspace(0.0, 1.0, 200)
    ax.plot(0.30 * vs.elliptic_shape(kappa), kappa, color="0.7", ls=":",
            label="elliptic (4.7)")
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(1.0, 0.0)
    ax.set_xlabel(r"$C_{l}$")
    ax.set_ylabel(r"$z'/h$")
    ax.set_xticks([0.0, 0.4, 0.8]); ax.set_yticks([0.0, 0.5, 1.0])
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.04, 0.06, "(a)", transform=ax.transAxes)

    ax = axes[1]
    for k, nspan_half in enumerate((4, 8, 16)):
        fluid, vent, cav, _ = solved(alpha=10.0, n_c=8, nspan_half=nspan_half)
        depth, cl = vnl.depth_loading(fluid["panels"], vent, cav["p_gauge"], RHO)
        ax.plot(cl / cl.max(), depth, color=str(0.6 - 0.3 * k), ls="-",
                marker="ov^"[k], ms=3, mfc="none",
                label=r"$n_{\mathrm{s}} = %d$" % nspan_half)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.15)
    ax.set_ylim(1.0, 0.0)
    ax.set_xlabel(r"$C_{l} / C_{l,\max}$")
    ax.set_ylabel(r"$z'/h$")
    ax.set_xticks([0.0, 0.5, 1.0]); ax.set_yticks([0.0, 0.5, 1.0])
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.9, 0.06, "(b)", transform=ax.transAxes)

    ax = axes[2]
    dy, force, circ = [], [], []
    for nspan_half in (4, 6, 8, 12, 16, 24, 32):
        fluid, _, _, vent, _ = strut_case(alpha=10.0, n_c=8,
                                          nspan_half=nspan_half)
        fluid = vsl.solve_wetted(fluid, vent, wake="frozen", sign=-1.0)
        fields = vnl.vent_pressure(fluid["panels"], fluid["mu"], fluid["u_rel"],
                                   vent["u_ref"], vent,
                                   np.zeros(len(fluid["mu"])), rho=RHO)
        depth, cl = vnl.depth_loading(fluid["panels"], vent, fields["p_gauge"],
                                      RHO)
        pan = fluid["panels"]
        mu = np.asarray(fluid["mu"]).reshape(pan["nwrap"], pan["nspan"])
        gamma = np.abs(mu[0, :] - mu[-1, :])[:nspan_half]
        dy.append(vent["h"] / nspan_half)
        force.append(float(cl[0]) / float(cl.max()))
        circ.append(float(gamma[-1]) / float(gamma.max()))
    dy = np.array(dy); force = np.array(force); circ = np.array(circ)
    ax.loglog(dy, circ, "k-", marker="o", ms=4,
              label=r"$\Gamma/\Gamma_{\max}$")
    ax.loglog(dy, force, color="0.5", ls="-", marker="s", ms=4, mfc="none",
              label=r"$C_{l}/C_{l,\max}$")
    ax.loglog(dy, circ[0] * dy / dy[0], color="0.7", ls="--",
              label=r"$O(\Delta y)$")
    ax.set_xlabel(r"$\Delta y/h$")
    ax.set_ylabel("waterline strip")
    ax.set_ylim(0.05, 2.0)
    ax.set_xticks([0.03, 0.1, 0.3])
    ax.set_xticklabels(["0.03", "0.1", "0.3"])
    ax.set_yticks([0.1, 0.3, 1.0])
    ax.set_yticklabels(["0.1", "0.3", "1"])
    ax.minorticks_off()
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.04, 0.9, "(c)", transform=ax.transAxes)
    fig.tight_layout()
    save(fig, "depth_loading")
    note("waterline circulation falls as dy^%.2f: %s"
         % (np.polyfit(np.log(dy), np.log(circ), 1)[0], np.round(circ, 4)))
    note("waterline sectional force does NOT vanish: %s of peak"
         % np.round(force, 4))
    note("free surface first strip %.3f of peak against rigid wall %.3f"
         % (peak["free surface"][0] / peak["free surface"][1],
            peak["rigid wall"][0] / peak["rigid wall"][1]))
    note("rigid-wall peak loading %.3f is %.2f times the free-surface peak %.3f"
         % (peak["rigid wall"][1],
            peak["rigid wall"][1] / peak["free surface"][1],
            peak["free surface"][1]))


# ------------------------------------------------------------------- figure 4

def fig_cavity_planform():
    """Cavity planform, sectional distributions and thickness."""
    alpha, fn_h = 16.0, 2.0
    fluid, vent, cav, loads = solved(alpha=alpha, fn_h=fn_h, n_c=14,
                                     nspan_half=10, regime="FV")
    pan = fluid["panels"]
    nh = vent["n_half"]
    depth = strip_depth(pan, vent)[:nh]
    order = np.argsort(depth)
    length = cav["l_c"][:nh][order]
    detach = cav["detach"][:nh][order]
    ll = lifting_line(alpha, fn_h, vent["ar"], regime="FV")
    hw = harwood_lift(alpha, fn_h, vent["ar"], regime="FV")

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.6))
    ax = axes[0]
    ax.fill_betweenx(depth[order], detach, detach + length, color="0.82",
                     edgecolor="k", lw=0.8, label="present, panel method")
    ax.plot(ll["length"], ll["kappa"], color="0.3", ls="--", lw=1.4,
            label="lifting line")
    ax.plot(hw["length"], hw["kappa"], color="0.55", ls="-.", lw=1.2,
            label="Helmbold chain")
    ax.axvline(1.0, color="0.6", ls="-", lw=0.8)
    ax.annotate("trailing edge", (1.15, 0.55), fontsize=7, rotation=90)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlim(0.03, 100.0)
    ax.set_ylim(1.05, -0.05)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$z'/h$")
    ax.set_xticks([0.1, 1.0, 10.0, 100.0])
    ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    ax.text(0.04, 0.94, "(a)", transform=ax.transAxes)

    ax = axes[1]
    ax.plot(vs.sigma_cavity(depth[order], vent["h"], fn_h), depth[order], "k-",
            marker="o", ms=3, mfc="none", label=r"$\sigma_{\mathrm{c}}$")
    ax.plot(ll["a0"] / (2.0 * np.pi), ll["kappa"], color="0.35", ls="--",
            label=r"$a_0/2\pi$")
    ax.plot(np.degrees(ll["alpha_2d"]) / 10.0, ll["kappa"], color="0.6",
            ls="-.", label=r"$\alpha_{\mathrm{2D}}/10^\circ$")
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.6)
    ax.set_ylim(1.0, 0.0)
    ax.set_xlabel("nondimensional")
    ax.set_ylabel(r"$z'/h$")
    ax.set_xticks([0.0, 0.8, 1.6]); ax.set_yticks([0.0, 0.5, 1.0])
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.04, 0.06, "(b)", transform=ax.transAxes)

    ax = axes[2]
    frame = cav["frame"]
    j_of = np.tile(np.arange(pan["nspan"]), pan["nwrap"])
    thick = np.abs(cav["thickness"]) / vent["chord"]
    picks = [int(np.argmin(np.abs(strip_depth(pan, vent)[:nh] - d)))
             for d in (0.25, 0.55, 0.85)]
    for k, j in enumerate(picks):
        here = (j_of == j) & (cav["weight"] > 0.0)
        if not np.any(here):
            continue
        xi = frame["xi"][here]
        srt = np.argsort(xi)
        ax.plot(xi[srt], thick[here][srt], color=str(0.6 - 0.3 * k), ls="-",
                marker="ov^"[k], ms=3, mfc="none",
                label=r"$z'/h = %.2f$" % strip_depth(pan, vent)[j])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$t_{\mathrm{c}}/c$")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.9, 0.06, "(c)", transform=ax.transAxes)
    fig.tight_layout()
    save(fig, "cavity_planform")
    rep = vcv.cavity_report(pan, frame, vent, cav)
    note("cavity length %.2f c at the waterline to %.2f c at the tip; "
         "lifting line %.2f to %.2f"
         % (length[0], length[-1], ll["length"][0], ll["length"][-1]))
    note("Helmbold chain length %.2f to %.2f c" % (hw["length"][0],
                                                   hw["length"][-1]))
    note("cavity volume %.3e c^3, thickness up to %.3f c, closure residual %.1e"
         % (rep["cavity_volume"], thick[cav["weight"] > 0.0].max(),
            rep["closure_residual"]))
    note("entrainment %.3e, iterations %d, coverage %.3f"
         % (float(np.sum(cav["entrainment"])), rep["iterations"],
            rep["coverage"]))


# ------------------------------------------------------------------- figure 5

def fig_pressure():
    """Chordwise pressure at three depths, wetted and ventilated."""
    alpha, fn_h = 14.0, 2.5
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5), sharey=True)
    out = {r: solved(alpha=alpha, fn_h=fn_h, n_c=16, nspan_half=8, regime=r)
           for r in ("FW", "FV")}
    fluid, vent, cav, _ = out["FV"]
    pan = fluid["panels"]
    frame = cav["frame"]
    nh = vent["n_half"]
    j_of = np.tile(np.arange(pan["nspan"]), pan["nwrap"])
    depth_strip = strip_depth(pan, vent)
    picks = [int(np.argmin(np.abs(depth_strip[:nh] - d)))
             for d in (0.15, 0.5, 0.85)]
    suction = vcv.suction_side(pan, vent["alpha_rad"])
    err = []
    for k, (ax, j) in enumerate(zip(axes, picks)):
        here = (j_of == j) & suction
        xi = frame["xi"][here]
        srt = np.argsort(xi)
        ax.plot(xi[srt], out["FW"][2]["cp_solved"][here][srt], "k-", marker="o",
                ms=3, mfc="none", label="fully wetted")
        ax.plot(xi[srt], cav["cp"][here][srt], color="0.35", ls="-", marker="s",
                ms=3, label="ventilated")
        ax.plot(xi[srt], cav["cp_cav"][here][srt], color="0.6", ls="--",
                label=r"$-\sigma_{\mathrm{c}}$")
        ax.axhline(0.0, color="0.85", lw=0.6)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1.0, -2.5)
        ax.set_xlabel(r"$x/c$")
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_title(r"$z'/h = %.2f$" % depth_strip[j])
        ax.text(0.04, 0.06, "(%s)" % "abc"[k], transform=ax.transAxes)
        deep = here & np.asarray(cav["interior"], dtype=bool)
        if np.any(deep):
            err.append(float(np.abs(cav["cp"][deep]
                                    - cav["cp_cav"][deep]).max()))
    axes[0].set_ylabel(r"$C_p$")
    axes[0].set_yticks([1.0, 0.0, -1.0, -2.0])
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "pressure")
    full = cav["weight"] > 0.99
    inner = np.asarray(cav["interior"], dtype=bool)
    note("dynamic condition: max |Cp + sigma_c| = %.1e over %d INTERIOR cavity "
         "panels, %.1e over all %d covered panels"
         % (float(np.abs(cav["cp"][inner] - cav["cp_cav"][inner]).max()),
            int(inner.sum()),
            float(np.abs(cav["cp"][full] - cav["cp_cav"][full]).max()),
            int(full.sum())))
    note("cavity length at the three stations: %s"
         % np.round([cav["l_c"][j] for j in picks], 3))
    note("suction peak removed: wetted %.3f, ventilated %.3f"
         % (float(out["FW"][2]["cp_solved"][suction].min()),
            float(cav["cp"][suction].min())))


# ------------------------------------------------------------------- figure 6

def fig_loads():
    """Lift, lift loss and centre of pressure."""
    alphas = np.array([6.0, 10.0, 14.0, 18.0, 22.0])
    fn_h = 2.5
    res = {"FW": [], "FV": []}
    for alpha in alphas:
        for regime in ("FW", "FV"):
            _, vent, cav, loads = solved(alpha=alpha, fn_h=fn_h, n_c=12,
                                         nspan_half=8, regime=regime)
            r = loads["resultants"]
            mean_l = float(np.mean(cav["l_c"][:vent["n_half"]]))
            res[regime].append((r["CL"], r["CM"] * vent["chord"] / r["CL"],
                                mean_l, r["CD_pressure"]))
    ll = {r: np.array([lifting_line(a, fn_h, 1.0, regime=r)["CL"]
                       for a in alphas]) for r in ("FW", "FV")}
    hw = {r: np.array([harwood_lift(a, fn_h, 1.0, regime=r)["CL"]
                       for a in alphas]) for r in ("FW", "FV")}
    cl_fw = np.array([v[0] for v in res["FW"]])
    cl_fv = np.array([v[0] for v in res["FV"]])

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.6))
    ax = axes[0]
    ax.axvspan(H_STALL[0], H_STALL[1], color="0.9", zorder=0)
    ax.plot(alphas, cl_fw, "k-", marker="o", ms=4, label="present, wetted")
    ax.plot(alphas, cl_fv, "k-", marker="s", ms=4, mfc="none",
            label="present, ventilated")
    ax.plot(alphas, hw["FW"], color="0.5", ls="--",
            label="Helmbold chain, wetted")
    ax.plot(alphas, hw["FV"], color="0.5", ls=":",
            label="Helmbold chain, ventilated")
    ax.annotate(r"$\alpha_{\mathrm{s}}$", (14.3, 0.03), fontsize=8)
    ax.set_xlim(0.0, 22.0)
    ax.set_ylim(0.0, 0.8)
    ax.set_xlabel(r"$\alpha$ [deg]")
    ax.set_ylabel(r"$C_L$")
    ax.set_xticks([0, 11, 22]); ax.set_yticks([0.0, 0.4, 0.8])
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.04, 0.06, "(a)", transform=ax.transAxes)

    ax = axes[1]
    ax.axhline(1.0 - H_CL_LOSS, color="0.6", ls="--", lw=1.0)
    ax.annotate("largest loss reported, 70%", (4.4, 1.0 - H_CL_LOSS + 0.03),
                fontsize=7)
    ax.plot(alphas, cl_fv / cl_fw, "k-", marker="s", ms=4, mfc="none",
            label="panel method")
    ax.plot(alphas, hw["FV"] / hw["FW"], color="0.5", ls="--",
            label="Helmbold chain")
    ax.plot(alphas, ll["FV"] / ll["FW"], color="0.5", ls=":",
            label="lifting line")
    ax.set_xlim(0.0, 22.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$\alpha$ [deg]")
    ax.set_ylabel(r"$C_{L,\mathrm{FV}} / C_{L,\mathrm{FW}}$")
    ax.set_xticks([0, 11, 22]); ax.set_yticks([0.0, 0.5, 1.0])
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.04, 0.9, "(b)", transform=ax.transAxes)

    ax = axes[2]
    lg = np.linspace(0.0, 2.0, 200)
    ax.plot(lg, vs.centre_of_pressure(lg), "k-", lw=1.4, label="Harwood (3.4)")
    ax.axhline(H_E_WET, color="0.7", ls=":", lw=0.9)
    ax.axhline(H_E_SUPER, color="0.7", ls="--", lw=0.9)
    ax.annotate(r"$c/4$", (1.55, H_E_WET + 0.008), fontsize=8)
    ax.annotate(r"$3c/16$", (1.45, H_E_SUPER - 0.022), fontsize=8)
    ax.plot([v[2] for v in res["FW"]], [v[1] for v in res["FW"]], "ko", ms=4,
            ls="none", label=r"present, wetted")
    ax.plot([v[2] for v in res["FV"]], [v[1] for v in res["FV"]], "ks", ms=4,
            mfc="none", ls="none", label=r"ventilated, $\alpha$ sweep")
    # a Froude sweep spreads the ventilated points along the cavity length, which
    # an incidence sweep does not: sigma_c depends on Fn_h and barely on alpha
    swept = []
    for fn in (1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
        _, vent, cav, loads = solved(alpha=14.0, fn_h=fn, n_c=12, nspan_half=8,
                                     regime="FV")
        r = loads["resultants"]
        swept.append((float(np.mean(cav["l_c"][:vent["n_half"]])),
                      r["CM"] * vent["chord"] / r["CL"], fn))
    ax.plot([v[0] for v in swept], [v[1] for v in swept], "k^", ms=4, mfc="0.7",
            ls="none", label=r"ventilated, $Fn_h$ sweep")
    ax.set_xlim(-0.05, 2.0)
    ax.set_ylim(0.1, 0.4)
    ax.set_xlabel(r"$\overline{L}_{\mathrm{c}}/c$")
    ax.set_ylabel(r"$e = x_{\mathrm{cp}}/c$")
    ax.set_xticks([0.0, 1.0, 2.0]); ax.set_yticks([0.1, 0.25, 0.4])
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.04, 0.06, "(c)", transform=ax.transAxes)
    fig.tight_layout()
    save(fig, "loads")
    note("lift ratio over incidence: %s (mean %.3f)"
         % (np.round(cl_fv / cl_fw, 3), float(np.mean(cl_fv / cl_fw))))
    note("centre of pressure: wetted %s, ventilated %s"
         % (np.round([v[1] for v in res["FW"]], 3),
            np.round([v[1] for v in res["FV"]], 3)))
    note("mean cavity length: %s" % np.round([v[2] for v in res["FV"]], 3))
    note("Froude sweep at alpha = 14 deg, (mean Lc/c, e, Fn_h): %s"
         % [tuple(np.round(v, 3)) for v in swept])
    note("drag ratio: %s"
         % np.round([v[3] / w[3] for v, w in zip(res["FV"], res["FW"])], 3))
    note("lifting line CL wetted %s ventilated %s"
         % (np.round(ll["FW"], 3), np.round(ll["FV"], 3)))
    note("Helmbold chain CL wetted %s ventilated %s"
         % (np.round(hw["FW"], 3), np.round(hw["FV"], 3)))
    note("lifting line exceeds the Helmbold chain in wetted lift by %s"
         % np.round(ll["FW"] / hw["FW"], 3))
    return res


# ------------------------------------------------------------------- figure 7

def fig_closure_angle():
    """The cavity closure line and its mean angle against the criterion."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax = axes[0]
    fns = (1.15, 1.5, 2.0)
    for k, fn_h in enumerate(fns):
        fluid, vent, cav, _ = solved(alpha=H_FIG8["alpha"], fn_h=fn_h, n_c=12,
                                     nspan_half=10, regime="FV")
        geo = cav["closure"]
        col = str(0.6 - 0.28 * k)
        ax.plot(geo["x_closure"], geo["depth"], color=col, ls="-",
                marker="ov^"[k], ms=3.5, mfc="none",
                label=r"$Fn_h = %.2f$" % fn_h)
        fit = np.polyfit(geo["x_closure"], geo["depth"], 1)
        span = np.array([geo["x_closure"].min(), geo["x_closure"].max()])
        ax.plot(span, np.polyval(fit, span), color=col, ls=":", lw=0.9)
        # the re-entrant jet leaves the closure line at 2*Phi from the horizontal
        for i in range(1, len(geo["depth"]) - 1,
                       max((len(geo["depth"]) - 2) // 3, 1)):
            two = 2.0 * geo["phi_local"][i]
            ax.arrow(geo["x_closure"][i], geo["depth"][i],
                     -0.13 * np.cos(two), 0.13 * np.sin(two), color=col,
                     width=0.002, head_width=0.018, length_includes_head=True)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.15)
    ax.set_ylim(1.05, -0.05)
    ax.set_xlabel(r"$x_{\mathrm{closure}}/c$")
    ax.set_ylabel(r"$z'/h$")
    ax.set_xticks([0.0, 0.5, 1.0]); ax.set_yticks([0.0, 0.5, 1.0])
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.04, 0.94, "(a)", transform=ax.transAxes)

    ax = axes[1]
    alphas = np.array([12.0, 16.0, 20.0, 24.0])
    got = {}
    for k, ar_h in enumerate((0.5, 1.0, 1.5)):
        phi, fn_w = [], []
        for alpha in alphas:
            # evaluated at the case's OWN washout Froude number, which is where
            # the angle was measured: a common Froude number would compare three
            # aspect ratios at three different distances from their boundaries
            fn_h = H_FIG8["fn_h"]
            for _ in range(3):
                _, vent, cav, loads = solved(alpha=alpha, fn_h=fn_h, h=ar_h,
                                             chord=1.0, n_c=12, nspan_half=10,
                                             regime="FV")
                cl = abs(loads["resultants"]["CL"])
                new_fn = float(vs.washout_froude(max(cl, 1e-3), ar_h))
                if abs(new_fn - fn_h) < 1e-3:
                    fn_h = new_fn
                    break
                fn_h = new_fn
            phi.append(np.degrees(cav["phi_bar"]))
            fn_w.append(fn_h)
        got[ar_h] = (np.array(phi), np.array(fn_w))
        ax.plot(alphas, phi, color=str(0.6 - 0.28 * k), ls="none",
                marker="ov^"[k], ms=5, mfc="none",
                label=r"$AR_h = %.1f$" % ar_h)
    ax.axhline(45.0, color="k", ls="--", lw=1.2)
    ax.axhline(H_PHI_WASHOUT, color="0.5", ls=":", lw=1.2)
    ax.annotate(r"criterion, $\bar\Phi = 45^\circ$", (12.3, 46.5), fontsize=7)
    ax.annotate("measured at washout, 40.75", (12.3, 35.5), fontsize=7)
    ax.set_xlim(10.0, 26.0)
    ax.set_ylim(20.0, 70.0)
    ax.set_xlabel(r"$\alpha$ [deg]")
    ax.set_ylabel(r"$\bar\Phi$ [deg]")
    ax.set_xticks([12, 18, 24]); ax.set_yticks([20, 35, 45, 60])
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.04, 0.06, "(b)", transform=ax.transAxes)
    fig.tight_layout()
    save(fig, "closure_angle")
    every = np.concatenate([got[k][0] for k in got])
    note("phi_bar at each case's own washout Froude number: mean %.2f deg, "
         "range %.2f to %.2f, against the measured %.2f"
         % (every.mean(), every.min(), every.max(), H_PHI_WASHOUT))
    for ar_h in got:
        note("  AR_h = %.1f: phi_bar %s at Fn_h %s"
             % (ar_h, np.round(got[ar_h][0], 2), np.round(got[ar_h][1], 3)))
    conv = []
    for nspan_half in (6, 8, 10, 12, 16):
        _, _, cav, _ = solved(alpha=H_FIG8["alpha"], fn_h=H_FIG8["fn_h"],
                              n_c=12, nspan_half=nspan_half, regime="FV")
        conv.append(np.degrees(cav["phi_bar"]))
    note("spanwise refinement at its figure 8(a) condition (alpha 20, Fn_h 1.5)"
         ": %s" % np.round(conv, 2))
    return got


# ------------------------------------------------------------------- figure 8

def fig_washout():
    """The washout boundary, and the regime map it bounds."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax = axes[0]
    cl = np.linspace(0.1, 1.4, 200)
    for k, ar in enumerate((0.5, 1.0, 2.0)):
        ax.plot(cl, vs.washout_froude(cl, ar), color=str(0.6 - 0.28 * k),
                ls="-", label=r"$AR_h = %.1f$" % ar)
    ax.plot(cl, vs.breslin_skalak(cl), "k--", lw=1.4,
            label="Breslin & Skalak (1.2)")
    ax.plot(cl[::20], vs.washout_froude_chain(cl[::20], 1.0), "ko", ms=3,
            mfc="none", label="(4.1)-(4.4) solved")
    ax.set_xlim(0.0, 1.4)
    ax.set_ylim(0.0, 6.0)
    ax.set_xlabel(r"$C_L$")
    ax.set_ylabel(r"$Fn_h$")
    ax.set_xticks([0.0, 0.7, 1.4]); ax.set_yticks([0, 3, 6])
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.04, 0.06, "(a)", transform=ax.transAxes)

    ax = axes[1]
    alphas = np.array([4.0, 8.0, 12.0, 16.0, 20.0])
    fns = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    marks = {"FW": "o", "PV": "s", "FV": "^"}
    seen, grid, boundary, depth_of = set(), {}, [], {}
    for alpha in alphas:
        cl_ref = None
        for fn_h in fns:
            _, vent, cav, loads = settled(alpha=alpha, fn_h=fn_h, n_c=10,
                                          nspan_half=6, regime="FW")
            grid[(alpha, fn_h)] = vent["regime"]
            depth_of[(alpha, fn_h)] = cav["d_cav"] / vent["h"]
            ax.scatter([alpha], [fn_h], marker=marks[vent["regime"]],
                       c=[depth_of[(alpha, fn_h)]], vmin=0.0, vmax=1.0,
                       cmap="Greys", edgecolors="k", linewidths=0.5, s=48,
                       label=vent["regime"] if vent["regime"] not in seen
                       else None, zorder=3)
            seen.add(vent["regime"])
            if cl_ref is None:
                cl_ref = abs(loads["resultants"]["CL"])
        boundary.append(float(vs.washout_froude(max(cl_ref, 1e-3), 1.0)))
    ax.plot(alphas, boundary, "k-", lw=1.2, label="washout (4.5)")
    ax.axvspan(H_STALL[0], H_STALL[1], color="0.9", zorder=0)
    ax.annotate(r"$\alpha_{\mathrm{s}}$", (14.2, 3.15), fontsize=8)
    bar = fig.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(0.0, 1.0), cmap="Greys"), ax=ax, ticks=[0.0, 0.5, 1.0])
    bar.set_label(r"$D/h$")
    ax.set_xlim(0.0, 22.0)
    ax.set_ylim(0.5, 3.5)
    ax.set_xlabel(r"$\alpha$ [deg]")
    ax.set_ylabel(r"$Fn_h$")
    ax.set_xticks([0, 11, 22]); ax.set_yticks([1, 2, 3])
    ax.legend(frameon=False, loc="lower left", numpoints=1, scatterpoints=1)
    ax.text(0.04, 0.9, "(b)", transform=ax.transAxes)
    fig.tight_layout()
    save(fig, "washout")
    err = float(np.abs(vs.washout_froude(cl, 1.0)
                       - vs.washout_froude_chain(cl, 1.0)).max())
    note("(4.5) against its own chain (4.1)-(4.4): max difference %.1e" % err)
    note("Breslin & Skalak exceeds (4.5) by a factor %.2f at AR_h = 1, CL = 0.5"
         % float(vs.breslin_skalak(0.5) / vs.washout_froude(0.5, 1.0)))
    note("(4.5) anchor at AR_h = 1, CL = 0.5: Fn_h = %.4f"
         % float(vs.washout_froude(0.5, 1.0)))
    for alpha in alphas:
        note("  alpha = %4.1f: %s  D/h %s  washout Fn_h = %.2f"
             % (alpha, [grid[(alpha, f)] for f in fns],
                np.round([depth_of[(alpha, f)] for f in fns], 2),
                boundary[int(np.where(alphas == alpha)[0][0])]))
    return grid


# ------------------------------------------------------------------- figure 9

def fig_hysteresis():
    """The bi-stable loop, swept up and back down in incidence."""
    alphas = np.array([4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    fn_h = 2.5
    branch = {}
    for name, order in (("up", alphas), ("down", alphas[::-1])):
        regime, cl, reg = "FW", [], []
        for alpha in order:
            _, vent, cav, loads = settled(alpha=alpha, fn_h=fn_h, n_c=10,
                                          nspan_half=6, regime=regime)
            regime = vent["regime"]                  # carried to the next point
            cl.append(loads["resultants"]["CL"])
            reg.append(regime)
        branch[name] = (np.asarray(order), np.asarray(cl), reg)

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    a_up, cl_up, r_up = branch["up"]
    a_dn, cl_dn, r_dn = branch["down"]
    band = [a for a, ru, rd in zip(a_up, r_up, r_dn[::-1]) if ru != rd]
    if band:
        ax.axvspan(min(band) - 1.0, max(band) + 1.0, color="0.93", zorder=0,
                   label="bi-stable")
    ax.axvspan(H_STALL[0], H_STALL[1], color="0.8", zorder=0)
    ax.plot(a_up, cl_up, "k-", marker="o", ms=4,
            label=r"increasing $\alpha$")
    ax.plot(a_dn, cl_dn, color="0.45", ls="--", marker="s", ms=4, mfc="none",
            label=r"decreasing $\alpha$")
    ax.annotate(r"$\alpha_{\mathrm{s}}$", (14.2, 0.02), fontsize=8)
    ax.set_xlim(0.0, 22.0)
    ax.set_ylim(0.0, 0.5)
    ax.set_xlabel(r"$\alpha$ [deg]")
    ax.set_ylabel(r"$C_L$")
    ax.set_xticks([0, 11, 22]); ax.set_yticks([0.0, 0.25, 0.5])
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "hysteresis")
    area = float(np.abs(np.trapezoid(cl_up, a_up) - np.trapezoid(cl_dn[::-1], a_dn[::-1])))
    note("up branch:   %s" % list(zip(a_up.astype(int), r_up)))
    note("down branch: %s" % list(zip(a_dn.astype(int), r_dn)))
    note("enclosed area %.4f CL-degrees; bi-stable over alpha = %s"
         % (area, band))
    return branch


# ------------------------------------------------------------------ figure 10

def fig_transient():
    """The coupled response to an inception event."""
    alpha, dt, nsteps, growth = 10.0, 0.02, 70, 0.1
    fluid, sstate, transfer, vent, mesh = strut_case(
        alpha=alpha, n_c=8, nspan_half=6, nwake=24, unsteady=True,
        growth_chords=growth, regime="FW")
    # air injected at the junction of the leading edge and the free surface, which
    # is the paper's perturbation route to inception. The incidence is BELOW the
    # stall angle, so without the injection this case stays wetted indefinitely,
    # and the march therefore starts from a wetted static equilibrium and steps
    # the load once, at the first commit.
    vent = vcv.inject(vent, active=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fluid, sstate, _ = fsd.static_aeroelastic(fluid, sstate, transfer,
                                                 rho=RHO, max_iter=10, vent=vent)
        cl_wet = fsd.ventilation_margin(fluid, vent, rho=RHO)["CL_wetted"]
        hist, fluid, sstate = fsd.time_march_fsi(fluid, sstate, transfer, dt=dt,
                                                nsteps=nsteps, rho=RHO,
                                                rho_inf=0.5, max_sub=25,
                                                vent=vent)
    s = np.asarray(hist["s"])
    cl = np.asarray(hist["CL"])
    tip = 1e3 * np.asarray(hist["tip"])
    fig, axes = plt.subplots(3, 1, figsize=(3.6, 4.8), sharex=True)
    axes[0].plot(s, cl, "k-", marker="o", ms=2.5)
    axes[0].axhline(cl_wet, color="0.6", ls="--", lw=0.9)
    axes[0].annotate("wetted equilibrium", (0.06, cl_wet + 0.03), fontsize=7)
    axes[0].set_ylabel(r"$C_L$")
    axes[1].plot(s, np.asarray(hist["l_c_max"]), "k-",
                 label=r"$L_{\mathrm{c}}/c$")
    axes[1].plot(s, np.asarray(hist["d_cav"]) / vent["h"], color="0.5", ls="--",
                 label=r"$D/h$")
    axes[1].set_ylabel("cavity")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_yticks([0.0, 0.5, 1.0])
    axes[1].legend(frameon=False, loc="center right")
    axes[2].plot(s, tip, "k-", marker="o", ms=2.5)
    axes[2].set_ylabel(r"$10^3\,\delta_{\mathrm{tip}}/c$")
    axes[2].set_xlabel(r"$s = 2 u t / c$")
    for k, ax in enumerate(axes):
        ax.text(0.03, 0.86, "(%s)" % "abc"[k], transform=ax.transAxes)
    axes[2].set_xlim(0.0, s.max())
    fig.tight_layout()
    save(fig, "transient")
    period = 2.0 / max(fsd.steps_per_period(sstate, dt), 1e-30) * dt
    note("transitions: %s" % vent["transitions"])
    note("wetted equilibrium CL %.4f; ventilated mean over the last third %.4f, "
         "ratio %.3f" % (cl_wet, float(cl[-nsteps // 3:].mean()),
                         float(cl[-nsteps // 3:].mean()) / cl_wet))
    note("tip: static %.3e c, peak %.3e c, overshoot %.2f"
         % (tip[0] * 1e-3, tip.max() * 1e-3, tip.max() / max(tip[0], 1e-30)))
    note("steps per period %.1f, so the wet fundamental is resolved by %.0f "
         "steps; ringing period in s = %.2f"
         % (fsd.steps_per_period(sstate, dt), fsd.steps_per_period(sstate, dt),
            period))
    note("subiterations per step: median %.1f, max %d"
         % (float(np.median(hist["sub"])), int(np.max(hist["sub"]))))
    note("growth rate %.2f chords of cavity per chord of travel, %.4f c per step"
         % (growth, growth * dt / vent["chord"]))
    return s, cl, vent


# ------------------------------------------------------------------ figure 11

def fig_twist():
    """Structural compliance moves the operating point across the boundary."""
    moduli = np.array([2.0e10, 1.0e8, 3.0e7, 1.0e7, 5.0e6, 3.0e6])
    alpha, fn_h = 16.0, 1.23
    tip, cl, margin, washout = [], [], [], []
    for e_mod in moduli:
        fluid, sstate, transfer, vent, mesh = strut_case(
            alpha=alpha, fn_h=fn_h, n_c=8, nspan_half=6, e_mod=e_mod)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fluid, sstate, _ = fsd.static_aeroelastic(
                fluid, sstate, transfer, rho=RHO, max_iter=12, vent=vent)
            rep = fsd.ventilation_margin(fluid, vent, rho=RHO)
        tip.append(float(np.abs(sstate["u"]).max()) / vent["chord"])
        cl.append(rep["CL_wetted"])
        margin.append(rep["margin"])
        washout.append(rep["washout_margin"])
    tip = np.array(tip); cl = np.array(cl)
    margin = np.array(margin); washout = np.array(washout)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax = axes[0]
    ax.semilogx(moduli, tip, "k-", marker="o", ms=4,
                label=r"$\delta_{\mathrm{tip}}/c$")
    ax.set_xlabel(r"$\hat{E}$ [Pa]")
    ax.set_ylabel(r"$\delta_{\mathrm{tip}}/c$")
    ax.invert_xaxis()
    twin = ax.twinx()
    twin.semilogx(moduli, cl, color="0.5", ls="--", marker="s", ms=4,
                  mfc="none", label=r"$C_L$")
    twin.set_ylabel(r"$C_L$", color="0.4")
    twin.tick_params(axis="y", colors="0.4")
    twin.invert_xaxis()
    ax.legend(frameon=False, loc="center left",
              handles=ax.get_lines() + twin.get_lines())
    ax.text(0.5, 0.9, "(a)", transform=ax.transAxes)

    ax = axes[1]
    ax.semilogx(moduli, washout, "k-", marker="o", ms=4,
                label=r"$(Fn_h - Fn_{h,\mathrm{w}})/Fn_{h,\mathrm{w}}$")
    ax.semilogx(moduli, margin / 10.0, color="0.5", ls="--", marker="s", ms=4,
                mfc="none", label=r"$\min(C_p + \sigma_{\mathrm{c}})/10$")
    ax.axhline(0.0, color="k", ls=":", lw=1.0)
    ax.annotate("fully ventilated flow sustainable", (1.4e10, 0.012),
                fontsize=7)
    ax.set_xlabel(r"$\hat{E}$ [Pa]")
    ax.set_ylabel("margin")
    ax.invert_xaxis()
    ax.legend(frameon=False, loc="lower left")
    ax.text(0.04, 0.9, "(b)", transform=ax.transAxes)
    fig.tight_layout()
    save(fig, "twist")
    note("alpha = %.0f deg, Fn_h = %.2f, AR_h = 1" % (alpha, fn_h))
    note("E from %.1e to %.1e Pa: tip %.4f to %.4f c, CL %.4f to %.4f"
         % (moduli[0], moduli[-1], tip[0], tip[-1], cl[0], cl[-1]))
    note("washout margin %+.4f to %+.4f; suction margin %.4f to %.4f"
         % (washout[0], washout[-1], margin[0], margin[-1]))
    return moduli, tip, cl, margin, washout


# ------------------------------------------------------------------------ run

FIGURES = {
    "sectional": fig_sectional,
    "mesh": fig_mesh,
    "depth_loading": fig_depth_loading,
    "cavity_planform": fig_cavity_planform,
    "pressure": fig_pressure,
    "loads": fig_loads,
    "closure_angle": fig_closure_angle,
    "washout": fig_washout,
    "hysteresis": fig_hysteresis,
    "transient": fig_transient,
    "twist": fig_twist,
}


def main():
    os.makedirs(OUT, exist_ok=True)
    wanted = sys.argv[1:] or list(FIGURES)
    print("Figures of docs/paper/ventilation.tex")
    for name in wanted:
        if name not in FIGURES:
            raise SystemExit(f"unknown figure {name!r}; "
                             f"one of {sorted(FIGURES)}")
        FIGURES[name]()
    print("Done.")


if __name__ == "__main__":
    main()
