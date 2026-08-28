# atmogen

`atmogen` is a deterministic, standalone vertical atmosphere/material solver for
procedural planets. It owns local hydrostatics, compact thermochemical equilibrium,
surface gas/liquid/solid partitioning, multicomponent liquid activity/stability,
bulk cloud condensate, and reduced-order spectral radiation. It does not import or
depend on a world generator.

```python
from atmogen import *

species = {"N2": 0.7808, "O2": 0.2095, "Ar": 0.0093, "CO2": 0.0004}
result = solve_planet(
    planet=PlanetPhysicalState(6_371_000, 9.80665, 101325),
    star=blackbody_stellar_spectrum(5772, 1361),
    inventory=ElementInventory(
        species_moles_to_elements(species), species,
        semantics="legacy molecular initial state",
    ),
    surface=SurfaceReservoirs({"H2O": 1.4e21}),
    settings=SolverSettings(chemistry_mode="fixed_species"),
)
```

All public physical inputs use SI units. Run `atmogen --demo earth` for a complete
representative column.

## Implemented physics

Version 0.4 implements a logarithmic-pressure vertical grid; analytic ideal-gas
isothermal hydrostatics; element-constrained ideal-mixture Gibbs minimization;
IAPWS-form water saturation; bounded estimated vapor-pressure fallbacks for several
bundled condensables; mass-conserving finite gas/liquid/solid surface partitioning;
multicomponent liquid activities through an explicit ideal/NRTL backend interface;
liquid-liquid Gibbs phase-stability testing; wavelength-resolved Rayleigh and
Beer-Lambert shortwave transfer; semi-gray longwave balance; Planck thermal spectra;
CIE-1931-fitted sRGB; complex-index absorption; Lorentz-Lorenz effective-medium
mixing; angular/polarized Fresnel reflectance from a non-absorbing incident medium;
and homogeneous-sphere Lorenz-Mie scattering with a Wiscombe-style downward
logarithmic-derivative recurrence. Batch calls de-duplicate identical input states.

The Mie API returns `Q_sca`, `Q_abs`, `Q_ext`, backscatter efficiency, asymmetry
parameter `g`, single-scattering albedo, size parameter and series length. Regression
tests include the Bohren-Huffman/Wiscombe absorbing-sphere cases at size parameters
`x=1` and `x=100`, the Rayleigh small-particle limit, and non-absorbing energy
partition.

## Scientific data and limits

Every bundled species datum carries provenance class, source, and validity text.
Thermochemistry is a compact NIST-derived constant-heat-capacity validation set.
Water saturation follows the IAPWS release form. Rayleigh follows the expected
species-dependent wavelength-to-the-minus-four limit. CIE conversion uses the
Wyman et al. analytic fit to the CIE 1931 2-degree functions.

NRTL is implemented as a solver backend and data schema, but the bundled database
does not fabricate binary interaction coefficients. `auto` therefore falls back to
an ideal liquid model when a complete directed parameter set is unavailable and
records that fallback. Custom/scientific databases may provide sourced NRTL
interaction energies, alpha parameters, provenance and validity metadata.

The current liquid-density calculation remains ideal-volume additive. Surface
pressure remains a prescribed boundary rather than a fully solved total-reservoir
pressure. The square-root-column longwave coefficients, non-water vapor-pressure
relations, and bulk cloud suspension are explicitly reduced-order estimates. FAST
condensable opacity applies a recorded bulk vertical-depletion factor because
surface saturation is not vertically uniform; this is not a substitute for a
resolved moist adiabat and band radiative transfer.

The Mie/Fresnel primitives are not yet automatically applied to clouds or ocean
rendering because the bundled material database does not yet contain a sufficiently
sourced wavelength-dependent complex refractive-index dataset. This is intentional:
the solver exposes the physical machinery without inventing authoritative RGB or
optical constants for missing materials.

Version 0.4 does **not** claim stiff kinetics, photolysis, vertical diffusion,
resolved cloud microphysics, sedimentation/precipitation, correlated-k,
line-by-line transfer, or a full radiative-convective/3-D climate solution. `HIGH`
and `REFERENCE` currently increase vertical resolution but do not falsely activate
unavailable high-fidelity backends.

Primary references:

- NIST Chemistry WebBook, gas thermochemistry and Shomate definitions:
  https://webbook.nist.gov/
- IAPWS thermodynamic/saturation releases: https://iapws.org/documents/release
- Bodhaine et al. (1999), *On Rayleigh Optical Depth Calculations*.
- Bohren & Huffman (1983), *Absorption and Scattering of Light by Small Particles*.
- Wiscombe (1980), improved Mie scattering algorithms and large-size-parameter
  numerical treatment.
- Scott Prahl / OMLC Mie implementation and equation documentation:
  https://omlc.org/software/mie/
- CIE 018:2019, CIE 1931 2-degree colour-matching functions:
  https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer
- Wyman, Sloan & Shirley (2013), analytic approximations to CIE XYZ functions.

## Reproducibility

Results record package version, API schema, material-data schema and database hash,
solver modes, fidelity, stellar-spectrum provenance, liquid-activity model,
fallbacks, conservation residuals, hydrostatic residual, energy imbalance, and
convergence history. No global random number generator is used.
