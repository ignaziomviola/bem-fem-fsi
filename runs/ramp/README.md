# A rigid surface-piercing wing in a pitch ramp

Produced by `vent_ramp.py`, on the branch that carries it:

```
MPLBACKEND=Agg python3 vent_ramp.py --nc 10 --nspan 10 --dt 0.05 --steps 400 \
                                    --nwake 120 --growth 0.2 --out runs/ramp
```

## The case

Nondimensional on the water density, the chord and the free-stream speed, so
rho = c = U = 1 and one convective time is one chord of travel.

| quantity | value |
| --- | --- |
| immersed span h/c | 4 |
| section | NACA 0012, no camber |
| incidence | 0 deg at t = 0, linear to 20 deg at t = 20 |
| depth Froude number U/sqrt(gh) | 10 |
| stall angle (the free-surface seal gate) | 15 deg |
| cavity pressure | atmospheric, dsigma = 0 |
| structure | rigid |
| panels, doubled mesh | 20 wrap x 20 spanwise (10 chordwise per surface, 10 spanwise immersed) |
| spanwise spacing | cosine within each half: fine at the immersed tip AND at the waterline |
| chordwise spacing | the vendored cosine wrap |
| time step | 0.05 c/U, 400 steps |
| wake | free, truncated at 120 rows (6 c) |
| cavity front speed | 0.2 chords of cavity per chord of travel |

## What the run does

| t | alpha (deg) | regime | CL | CD | dry area | max L_c/c |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 5 | FW | +0.374 | 0.015 | 0 % | 0 |
| 10 | 10 | FW | +0.683 | 0.050 | 0 % | 0 |
| 15 (inception) | 15 | FW -> PV | +0.980 | 0.103 | 0 % | 0 |
| 16 | 16 | PV | +0.710 | 0.207 | 10.2 % | 0.20 |
| 18 | 18 | PV | +0.458 | 0.145 | 30.0 % | 0.60 |
| 20 | 20 | PV | -0.176 | -0.083 | 49.7 % | 1.00 |

The wetted lift slope over 5 to 14 degrees is 0.061 per degree on s_ref = h*c.
Ventilation is inhibited until the incidence reaches the stall angle, which is
the seal gate; the transition is then immediate, and the cavity front advances
at the rate limit, reaching the trailing edge at t = 20. Half the immersed
surface is dry by then, the cavity covering the whole suction side, and the
lift has collapsed from its wetted value of about 1.3 at 20 degrees to roughly
zero. The linearised wave elevation that the image projection discards stays
below 0.018 c throughout, so the undeformed-plane image is consistent.

## Two things to read carefully

**The spikes after inception are a discretisation artefact of the moving
front, not a physical oscillation.** They occur at max L_c/c = 0.05, 0.15,
0.27, 0.42, 0.57, 0.72, 0.84 - one for each chordwise panel whose collocation
point the cavity front crosses. Crossing switches that panel to the Dirichlet
cavity condition, which steps its doublet strength, and the step reaches the
pressure through dmu/dt. The running median in `history.png` is the trend; a
finer chordwise mesh reduces the amplitude, since the doublet step per crossing
falls with the panel size.

**The cavity front must be resolved in time.** At the model default of 1.0
chords per chord of travel - 0.05 c per step at this dt - the same case ran
away: the cavity reached the trailing edge in twenty steps, CL swung to -1.55,
and the march diverged outright by t = 19. That is the open item recorded in
`docs/VENTILATION.md`, and here it is fatal rather than merely inaccurate.

## Files

- `mesh.png` - the doubled mesh, one section, and the spanwise spacing
- `wake.png` - the free wake at t = 20, from the side and from above
- `history.png` - CL, CD, CM, dry and wetted area, cavity extent, regime, washout margin
- `cavity.png` - the cavity footprint, the pressure it imposes, and the depth loading at t = 20
- `history.npz` - every recorded quantity, one entry per time level
