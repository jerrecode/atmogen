# atmogen

`atmogen` is a deterministic, standalone vertical atmosphere/material solver for
procedural planets. It owns local hydrostatics, compact thermochemical equilibrium,
surface gas/liquid/solid partitioning, multicomponent liquid activity/stability,
local stiff reaction kinetics, spectral photolysis, vertical eddy transport and
quench diagnostics, cloud/aerosol particle mechanics and optics, precipitation
routing, reduced-order radiative-convective structure, and spectral radiation. It
does not import or depend on a world generator.

All public physical inputs use SI units. Run `atmogen --demo earth` for a complete
representative column.

## Implemented physics

Version 0.9 implements a logarithmic-pressure vertical grid; finite-volume ideal-gas
hydrostatics for isothermal or layer-varying temperature profiles; element-constrained
ideal-mixture Gibbs minimization; IAPWS-form water saturation; explicit estimated
vapor-pressure fallbacks for several condensables; mass-conserving finite
gas/liquid/solid surface partitioning; ideal/NRTL liquid activity backends and
liquid-liquid Gibbs stability; spectral Rayleigh and Beer-Lambert shortwave transfer;
semi-gray longwave balance; Planck thermal spectra; CIE-derived visible colour;
complex-index absorption; Lorentz-Lorenz mixing; Fresnel reflection; and robust
homogeneous-sphere Lorenz-Mie scattering.

FAST fidelity deliberately retains the historical isothermal vertical thermal
profile. STANDARD, HIGH and REFERENCE use a reduced-order dry gray
radiative-convective profile. The gray radiative shape uses

`tau(P) = tau_s * (P/P_s)^2`

and `T^4 proportional to (tau + 2/3)`, normalized to preserve the public first-layer
surface-temperature proxy. Superadiabatic layers are limited to the ideal dry
adiabatic logarithmic pressure gradient

`nabla_ad = R / Cp_molar`.

This is a dry static-stability/radiative-structure model, not a moist adiabat and not
a layer-by-layer radiative-flux equilibrium solver. The converged surface energy
balance remains the single authority for surface temperature.

The chemistry layer provides provenance-carrying reaction records, elemental-balance
validation, Arrhenius mass-action kinetics, externally supplied first-order
photolysis rates, and stiff BDF/Radau integration. Photolysis utilities convert
spectral irradiance to photon flux and evaluate

`J = integral sigma(lambda) q(lambda) Phi(lambda) dlambda`.

Vertical transport implements the conservative finite-volume eddy flux

`Phi_i = -Kzz * n * d f_i / dz`

with zero-flux top/bottom boundaries, nonuniform density/Kzz profiles, BDF/Radau
integration, column-inventory checks, and derived `tau_mix=L^2/Kzz` quench crossings
rather than a configured quench altitude.

A `ParticlePopulation` represents a spherical lognormal radius distribution with
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

`solve_planet` now owns the resolved vertical cloud/transport state. With
`cloud_mode="lognormal_sedimentation"`, it performs the bounded suspended-condensate,
settling and precipitation operator. When a complex particle refractive index is
provided, resolved Mie extinction participates in the shortwave calculation. When
optical material data are absent, the solver records that fact and uses its bounded
bulk-gray cloud optical fallback rather than inventing refractive-index data.

`precipitation_step` provides a conservative operator-split downward routing step.
Settling condensate can cross at most one represented atmospheric interface per
step, preventing numerical teleportation through a coarse column. Optional
receiving-layer evaporation timescales allow falling material to re-evaporate before
reaching the surface. The result separately exposes remaining condensate, downward
transfer, re-evaporated mass, surface precipitation and mass-closure error. Surface
precipitation is explicitly an amount per configured microphysics operator step; it
is not mislabeled as an annual climatological precipitation rate.

`solve_columns` supports deterministic exact-state de-duplication and an explicit
per-column stellar-flux multiplier, allowing a horizontal host model to preserve
geographic/seasonal forcing differences while reusing identical vertical states.

## Fidelity semantics

`FAST` prioritizes compatibility and speed: the vertical temperature profile is
isothermal and several opacity/phase couplings remain deliberately reduced order.

`STANDARD`, `HIGH` and `REFERENCE` activate the dry gray radiative-convective
vertical profile in addition to their progressively larger default layer counts.
They do not silently activate nonexistent chemistry networks, correlated-k data,
moist convection, or molecular diffusion. Higher fidelity therefore means more
resolved use of implemented physics, not fabricated backends.

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
than inventing one universal droplet law for arbitrary condensates.

The current liquid-density model remains ideal-volume additive. Surface pressure is
a prescribed boundary rather than a solved total-reservoir pressure. Non-water
vapor-pressure relations and the semi-gray longwave coefficients remain reduced-order
approximations. Cloud source/nucleation/growth physics is still reduced order even
though settling, size-distribution optics and precipitation routing are resolved.

The dry radiative-convective profile does not conserve layer enthalpy through a
prognostic convective mixing calculation; it imposes static stability on a gray
radiative shape anchored to the independently solved surface energy balance. It does
not include latent heat in the adiabat, wavelength-resolved longwave heating rates,
or layerwise flux convergence.

Version 0.9 does **not** claim a bundled planetary kinetic network, molecular
vertical diffusion/gravitational separation, nucleation/coagulation, a fully coupled
latent-heat precipitation cycle, correlated-k or line-by-line transfer, a moist
radiative-convective solver, or a full 3-D climate solution. The kinetics,
photolysis and eddy-transport primitives remain explicit backends until a sourced
reaction network, UV actinic-transfer state and physically selected Kzz profile are
provided.

Primary references include NIST Chemistry WebBook thermochemistry; IAPWS saturation
releases; SciPy implicit BDF/Radau integration; planetary Kzz flux-gradient and
chemical/mixing-timescale literature; Davies/Cunningham slip correction; Stokes and
Schiller-Naumann sphere settling; Bohren & Huffman and Wiscombe Mie scattering; gray
Eddington atmosphere relations and dry ideal-gas adiabatic stability; CIE 1931 colour
matching; and Wyman et al. analytic CIE approximations.

## Reproducibility

Results and scientific-data objects expose package/API/data schema versions,
database hashes, solver modes, thermal-profile model, provenance/fallback information
and numerical residuals. Reaction-network data participate in the database revision
hash. Standalone kinetics, transport, particle and precipitation calculations expose
their conservation or integration diagnostics. No global random number generator is
used.
