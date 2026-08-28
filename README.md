# atmogen

`atmogen` is a deterministic, standalone vertical atmosphere/material solver for
procedural planets. It owns local hydrostatics, compact thermochemical equilibrium,
surface gas/liquid/solid partitioning, multicomponent liquid activity/stability,
local stiff reaction kinetics, spectral photolysis, vertical eddy transport and
quench diagnostics, cloud/aerosol particle mechanics and optics, and reduced-order
spectral radiation. It does not import or depend on a world generator.

All public physical inputs use SI units. Run `atmogen --demo earth` for a complete
representative column.

## Implemented physics

Version 0.7 implements a logarithmic-pressure vertical grid; ideal-gas isothermal
hydrostatics; element-constrained ideal-mixture Gibbs minimization; IAPWS-form water
saturation; explicit estimated vapor-pressure fallbacks for several condensables;
mass-conserving finite gas/liquid/solid surface partitioning; ideal/NRTL liquid
activity backends and liquid-liquid Gibbs stability; spectral Rayleigh and
Beer-Lambert shortwave transfer; semi-gray longwave balance; Planck thermal spectra;
CIE-derived visible colour; complex-index absorption; Lorentz-Lorenz mixing; Fresnel
reflection; and robust homogeneous-sphere Lorenz-Mie scattering.

The chemistry layer provides provenance-carrying reaction records, elemental-balance
validation, Arrhenius mass-action kinetics, externally supplied first-order
photolysis rates, and stiff BDF/Radau integration. Photolysis utilities convert
spectral irradiance to photon flux and evaluate

`J = integral sigma(lambda) q(lambda) Phi(lambda) dlambda`.

Vertical transport implements the conservative finite-volume eddy flux

`Phi_i = -Kzz * n * d f_i / dz`

with zero-flux top/bottom boundaries, nonuniform density/Kzz profiles, BDF/Radau
integration, column-inventory checks, and derived `tau_mix=L^2/Kzz` quench
crossings rather than a configured quench altitude.

Version 0.7 also introduces a reusable cloud/aerosol particle layer. A
`ParticlePopulation` represents a spherical lognormal radius distribution with
composition, number concentration, median radius, geometric standard deviation and
particle density. Analytical lognormal moments provide mass concentration and
effective radius. Settling uses Stokes-Cunningham only when the resulting Reynolds
number is in the creeping-flow regime; larger particles solve terminal force balance
with the Schiller-Naumann sphere drag relation. Sedimentation mass flux is integrated
over the radius distribution with Gauss-Hermite quadrature.

The same particle distribution can be integrated through the Mie backend to return
wavelength-resolved extinction, scattering, absorption, single-scattering albedo
and asymmetry parameter. Optical material indices remain caller/data supplied, so
this capability does not manufacture cloud colours or refractive indices when
laboratory data are absent.

`precipitation_step` provides a conservative operator-split downward routing step.
Settling condensate can cross at most one represented atmospheric interface per
step, preventing numerical teleportation through a coarse column. Optional
receiving-layer evaporation timescales allow falling material to re-evaporate before
reaching the surface. The result separately exposes remaining condensate,
downward transfer, re-evaporated mass, surface precipitation and mass-closure error.

## Scientific data and limits

Every bundled species datum carries provenance class, source and validity text.
Thermochemistry is a compact NIST-derived constant-heat-capacity validation set.
Water saturation follows the IAPWS release form. The bundled database does not
fabricate NRTL interaction parameters or a planetary reaction network; those
backends remain useful with custom/sourced data, while automatic fallbacks are
reported explicitly. Synthetic coefficients appear only in tests where an analytic
solution or numerical invariant is being validated.

Kzz is supplied rather than inferred from a universal planet formula because
effective eddy mixing is circulation/regime dependent. Likewise, precipitation
re-evaporation currently accepts an externally derived evaporation timescale rather
than inventing one universal droplet law for arbitrary condensates. These interfaces
are intended for subsequent coupling to phase equilibrium, gas transport and energy
balance.

The current liquid-density model remains ideal-volume additive. Surface pressure is
a prescribed boundary rather than a solved total-reservoir pressure. The FAST
longwave opacity coefficients, non-water vapor-pressure relations and current
top-level bulk cloud suspension remain reduced-order approximations.

The new Mie particle optics and resolved sedimentation are not yet automatically
substituted for the old bulk-cloud term inside `solve_planet`, because the bundled
material database still lacks sufficiently sourced wavelength-dependent complex
refractive-index spectra and a sourced nucleation/growth model for all supported
condensates. The kinetics/photolysis/transport primitives likewise remain explicit
operator backends until a sourced reaction network, UV actinic-transfer state and
Kzz profile are selected. The implementation does not claim capabilities merely
because the interfaces now exist.

Version 0.7 does **not** yet claim a bundled planetary kinetic network, molecular
vertical diffusion/gravitational separation, nucleation/coagulation, a fully coupled
latent-heat precipitation cycle, correlated-k or line-by-line transfer, or a full
radiative-convective/3-D climate solution. `HIGH` and `REFERENCE` increase vertical
resolution but do not falsely activate unavailable high-fidelity backends.

Primary references include NIST Chemistry WebBook thermochemistry; IAPWS saturation
releases; SciPy implicit BDF/Radau integration; planetary Kzz flux-gradient and
chemical/mixing-timescale literature; Davies/Cunningham slip correction; Stokes and
Schiller-Naumann sphere settling; Bohren & Huffman and Wiscombe Mie scattering; CIE
1931 colour matching; and Wyman et al. analytic CIE approximations.

## Reproducibility

Results and scientific-data objects expose package/API/data schema versions,
database hashes, solver modes, provenance/fallback information and numerical
residuals. Reaction-network data participate in the database revision hash.
Standalone kinetics, transport, particle and precipitation calculations expose their
conservation or integration diagnostics. No global random number generator is used.
