# atmogen

`atmogen` is a deterministic, standalone vertical atmosphere/material solver for
procedural planets. It owns local hydrostatics, compact thermochemical equilibrium,
surface phase partitioning, bulk cloud condensate, and reduced-order spectral
radiation. It does not import or depend on a world generator.

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

The 0.1 API implements a logarithmic-pressure grid; analytic ideal-gas isothermal
hydrostatics; element-constrained ideal-mixture Gibbs minimization; IAPWS-form water
saturation; bounded estimated vapor-pressure fallbacks for CO2, CH4 and NH3;
mass-conserving gas/liquid/solid surface partitioning; wavelength-resolved
Rayleigh and Beer–Lambert shortwave transfer; semi-gray longwave balance; Planck
thermal spectra; Fresnel normal-incidence reflection; and CIE-1931-fitted sRGB.
Batch calls de-duplicate identical input states.

## Scientific data and limits

Every bundled species datum carries provenance class, source, and validity text.
Thermochemistry is a compact NIST-derived constant-heat-capacity validation set.
Water saturation follows the IAPWS release form. Rayleigh follows the expected
species-dependent wavelength-to-the-minus-four limit. CIE conversion uses the
Wyman et al. analytic fit to the CIE 1931 2-degree functions.

The current longwave coefficients, non-water vapor-pressure relations, and bulk
cloud suspension are explicitly reduced-order estimates. Version 0.1 does **not**
claim non-ideal activity models, liquid-liquid separation, kinetics, photolysis,
vertical diffusion, Mie scattering, correlated-k, line-by-line transfer, or a full
radiative-convective/3-D climate solution. `HIGH` and `REFERENCE` currently increase
vertical resolution but do not falsely activate unavailable high-fidelity backends.

Primary references:

- NIST Chemistry WebBook, gas thermochemistry and Shomate definitions:
  https://webbook.nist.gov/
- IAPWS-95 and saturation-property releases: https://iapws.org/documents/release
- Bodhaine et al. (1999), *On Rayleigh Optical Depth Calculations*.
- CIE 018:2019, CIE 1931 2-degree colour-matching functions:
  https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer
- Wyman, Sloan & Shirley (2013), analytic approximations to CIE XYZ functions.

## Reproducibility

Results record package version, API schema, material-data schema and database hash,
solver modes, fidelity, stellar-spectrum provenance, fallbacks, conservation
residuals, hydrostatic residual, energy imbalance, and convergence history. No
global random number generator is used.
