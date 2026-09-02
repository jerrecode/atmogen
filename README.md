# atmogen

`atmogen` is a deterministic standalone atmosphere/material-column engine for
procedural planets. It owns local hydrostatics, compact thermochemical equilibrium,
surface gas/liquid/solid partitioning, multicomponent liquid activity/stability,
local stiff reaction kinetics, spectral photolysis, vertical eddy-transport
primitives, cloud/aerosol particle mechanics and optics, precipitation routing,
reduced-order radiative-convective structure, and spectral radiation. It does not
import or depend on a world generator.

All public physical inputs use SI units. Run `atmogen --demo earth` for a complete
representative column.

## Implemented physics

Version 0.10 implements a logarithmic-pressure vertical grid; finite-volume ideal-gas
hydrostatics for isothermal or layer-varying temperature profiles; element-constrained
ideal-mixture Gibbs minimization; sourced H2O saturation over liquid and ice;
explicit estimated vapor-pressure fallbacks for several other condensables;
mass-conserving finite gas/liquid/solid surface partitioning; ideal/NRTL liquid
activity backends and liquid-liquid Gibbs stability; spectral Rayleigh and
Beer-Lambert shortwave transfer; semi-gray longwave balance; Planck thermal spectra;
CIE-derived visible colour; complex-index absorption; Lorentz-Lorenz mixing; Fresnel
reflection; and robust homogeneous-sphere Lorenz-Mie scattering.

Water saturation is piecewise and provenance-explicit. From 273.16 K through the
critical point the solver uses the IAPWS saturation release form. From 110 K up to
the triple point it uses the Murphy & Koop (2005) equilibrium vapor-pressure relation
over ice,

`ln(p_ice/Pa) = 9.550426 - 5723.265/T + 3.53068 ln(T) - 0.00728332 T`.

The solver does not silently extrapolate that relation outside its published domain.

## Vertical thermal profiles

`SolverSettings.temperature_profile_mode` is explicit:

- `auto`
- `isothermal`
- `dry_radiative_convective`
- `dilute_saturated`

With `auto`, FAST retains the historical isothermal thermal profile, STANDARD uses a
dry gray radiative-convective profile, and HIGH/REFERENCE attempt the bounded
single-condensable saturated backend and fall back to the dry profile when its
physical/data requirements are not met.

The gray radiative shape uses

`tau(P) = tau_s * (P/P_s)^n`

with configurable `gray_optical_depth_pressure_exponent` (default 2) and
`T^4 proportional to (tau + 2/3)`. It is normalized to preserve the public
first-layer surface-temperature proxy. The converged surface energy balance remains
the single authority for surface temperature.

Dry superadiabatic layers are constrained by the ideal-gas logarithmic pressure
gradient

`nabla_d = R / Cp_molar`.

For one eligible saturated condensable, the solver can instead use the standard
dilute approximate saturated/pseudoadiabatic lapse-rate relation

`Gamma_m = g/c_pd * (1 + L*r_s/(R_d*T)) / (1 + epsilon*L^2*r_s/(c_pd*R_d*T^2))`

and its hydrostatic pressure-coordinate equivalent. The saturation mixing ratio is
computed from the condensable vapor pressure and the non-condensable carrier gas.
The implementation has deliberate validity gates: a condensed reservoir must exist,
the lower boundary must be near saturation, a positive latent heat and a
non-condensable carrier atmosphere must exist, and the saturation mixing ratio must
remain below `moist_max_saturation_mixing_ratio`. Estimated non-water saturation
relations are not used automatically unless `moist_allow_estimated_saturation=true`.

The current database contains one generic latent-heat value per species rather than
separate vaporization and sublimation enthalpies. Therefore the saturated convective
constraint reverts to dry stability after entering a solid-condensate regime instead
of incorrectly substituting vaporization latent heat for sublimation latent heat.
This limitation is explicit in diagnostics.

This is a bounded single-condensable dilute saturated adjustment, not a full
reversible multicomponent moist-convection solver. It does not prognose condensate
loading/enthalpy inside the convective adjustment and does not solve layer-by-layer
radiative heating-rate convergence.

## Chemistry, transport, clouds, and precipitation

The chemistry layer provides provenance-carrying reaction records, elemental-balance
validation, Arrhenius mass-action kinetics, externally supplied first-order
photolysis rates, and stiff BDF/Radau integration. Photolysis utilities convert
spectral irradiance to photon flux and evaluate wavelength-integrated photolysis
rates.

Vertical transport implements a conservative finite-volume eddy-flux backend

`Phi_i = -Kzz * n * d f_i / dz`

with zero-flux top/bottom boundaries, nonuniform density/Kzz profiles, BDF/Radau
integration, column-inventory checks, and derived `tau_mix=L^2/Kzz` quench crossings.
The top-level column result currently exposes resolved Kzz/mixing timescales; fully
coupling a sourced kinetic network through that transport operator remains future
work.

A `ParticlePopulation` represents a spherical lognormal radius distribution with
composition, number concentration, median radius, geometric standard deviation and
particle density. Analytical lognormal moments provide mass concentration and
effective radius. Settling uses Stokes-Cunningham only in the creeping-flow regime;
larger particles use sphere drag force balance. Sedimentation mass flux is integrated
over the size distribution with Gauss-Hermite quadrature.

The same particle population can be integrated through the Mie backend to return
wavelength-resolved extinction, scattering, absorption, single-scattering albedo and
asymmetry parameter. Optical material indices remain caller/data supplied, so the
solver does not manufacture cloud colours or refractive indices when laboratory data
are absent.

`solve_planet` owns the resolved vertical cloud/transport state. With
`cloud_mode="lognormal_sedimentation"`, it performs the bounded suspended-condensate,
settling and precipitation operator. When a complex particle refractive index is
provided, resolved Mie extinction participates in shortwave transfer; otherwise the
solver records and uses its bounded bulk-gray optical fallback.

The conservative precipitation operator allows settling condensate to cross at most
one represented atmospheric interface per step. Optional receiving-layer evaporation
timescales permit re-evaporation before material reaches the surface. Surface
precipitation is explicitly an amount per configured microphysics operator step, not
an annual climatological precipitation rate.

`solve_columns` supports deterministic exact-state de-duplication and an explicit
per-column stellar-flux multiplier, allowing a horizontal host model to preserve
geographic/seasonal forcing differences while reusing identical vertical states.
`column_state_fingerprint` publishes the versioned cache identity used by that
de-duplication, including the solver settings and chemical-database revision.
`solve_columns_with_diagnostics` retains the state-aligned result order while also
reporting unique-state counts, de-duplication, convergence, fallbacks, fingerprints
and per-column provenance for host-side caches and request coalescing.

## Fidelity semantics

`FAST` prioritizes compatibility and speed. Its automatic thermal profile is
isothermal and several opacity/phase couplings remain deliberately reduced order.

`STANDARD` automatically activates the dry gray radiative-convective vertical
profile.

`HIGH` and `REFERENCE` automatically attempt the single-condensable dilute saturated
constraint when its thermodynamic/data gates are satisfied. They otherwise fall back
to the dry gray profile and report the reason. Higher fidelity does not silently
activate nonexistent chemistry networks, correlated-k data, molecular diffusion, or
multicomponent moist convection.

## Scientific data and limits

Every bundled species datum carries provenance class, source and validity text.
Thermochemistry is a compact NIST-derived constant-heat-capacity validation set.
Water saturation uses IAPWS plus Murphy & Koop ice saturation. The bundled database
does not fabricate NRTL interaction parameters or a planetary reaction network;
those backends remain useful with custom/sourced data, while automatic fallbacks are
reported explicitly. Synthetic coefficients appear only in tests where an analytic
solution or numerical invariant is being validated.

Kzz is supplied rather than inferred from a universal planet formula because
effective eddy mixing is circulation/regime dependent. Likewise, precipitation
re-evaporation accepts an externally derived evaporation timescale rather than
inventing one universal droplet law for arbitrary condensates.

The current liquid-density model remains ideal-volume additive. Surface pressure is
a prescribed boundary rather than a solved total-reservoir pressure. Non-water
vapor-pressure relations and the semi-gray longwave coefficients remain reduced-order
approximations. Cloud source/nucleation/growth physics is still reduced order even
though settling, size-distribution optics and precipitation routing are resolved.

Version 0.10 does **not** claim a bundled planetary kinetic network, molecular
vertical diffusion/gravitational separation, nucleation/coagulation, a fully coupled
latent-heat precipitation cycle, multicomponent/non-dilute moist convection,
solid-phase moist convection without sourced sublimation enthalpies, correlated-k or
line-by-line transfer, layerwise radiative-flux convergence, or a full 3-D climate
solution.

Primary references include NIST Chemistry WebBook thermochemistry; IAPWS saturation
releases; Murphy & Koop (2005) ice vapor pressure; American Meteorological Society
adiabatic/pseudoadiabatic definitions; SciPy implicit BDF/Radau integration;
planetary Kzz flux-gradient and chemical/mixing-timescale literature;
Davies/Cunningham slip correction; Stokes and Schiller-Naumann sphere settling;
Bohren & Huffman and Wiscombe Mie scattering; gray Eddington atmosphere relations;
CIE 1931 colour matching; and Wyman et al. analytic CIE approximations.

## Reproducibility

Results and scientific-data objects expose package/API/data schema versions,
database hashes, requested/selected thermal-profile modes, condensable selection,
provenance/fallback information and numerical residuals. Reaction-network data
participate in the database revision hash. Standalone kinetics, transport, particle
and precipitation calculations expose their conservation or integration diagnostics.
No global random number generator is used.
