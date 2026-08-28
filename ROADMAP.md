# atmogen Roadmap

`atmogen` is the atmosphere/material-column authority. It does not own terrain tiling, globe rendering, viewer caches, vector tiles or 3-D export. Those are `artifexian_auto_worldgen` responsibilities. This roadmap records the `atmogen` work required to support the worldgen planetary-LOD phases without duplicating atmospheric chemistry/radiation inside worldgen.

## Cross-repository implementation matrix

The worldgen planetary roadmap tracks nine major workstreams. Their ownership is deliberately explicit:

| Workstream | Primary owner | atmogen responsibility |
|---|---|---|
| Camera/SSE LOD tests and parent/child coverage | worldgen | none; atmospheric results must remain deterministic when requested in different tile order |
| Local hydrology, erosion, deposition, hillslope diffusion | worldgen | expose atmospheric/surface boundary quantities needed by local climate/runoff coupling, without owning geomorphology |
| Hierarchical continental river constraints | worldgen | none directly; preserve deterministic column interfaces used by river/runoff climate coupling |
| Topographic precipitation/wind downscaling | worldgen + atmogen interface | worldgen resolves horizontal/topographic flow; atmogen supplies physically consistent representative/local vertical-column thermodynamics/cloud/radiation where requested |
| High-resolution slope/normal/soil/snow/biome/albedo | worldgen | provide atmospheric radiative/cloud/condensation quantities needed to constrain albedo/snow/cloud coupling |
| Persistent cache quotas/LRU/request coalescing | worldgen | expose stable provenance/cache keys for atmospheric column states so worldgen can safely de-duplicate column requests |
| Vector tiling | worldgen | none |
| Viewer-facing HTTP/service/GUI | worldgen | optional serialization of compact atmospheric diagnostics/spectra for layer inspection |
| Cesium/3D Tiles/standard exporters | worldgen | expose versioned scientific metadata suitable for embedding/referencing in exported tileset metadata |

## Phase A — Stable multiscale column contract

**Status:** partially implemented. `solve_columns` already supports deterministic exact-state de-duplication and per-column stellar-flux scaling.

Deliverables:

1. Formalize a compact, hashable local-column input state for host models, including radius/gravity, local surface pressure/elevation relation, surface temperature guess, albedo, stellar forcing, composition/reservoir state and solver settings.
2. Publish a stable column-state fingerprint suitable for worldgen request coalescing and persistent cache keys.
3. Ensure column batching is order-independent and deterministic across worker count/execution order.
4. Add explicit batch diagnostics: unique-state count, de-duplication ratio, fallback counts, convergence counts and per-column provenance.
5. Add tests that reordered and duplicated column batches return bitwise/strictly equivalent state-aligned outputs where numerically appropriate.

Acceptance gate: worldgen can request thousands of geographically repeated representative columns without inventing its own atmospheric cache identity.

## Phase B — Elevation/pressure/topographic boundary support

**Status:** planned.

Deliverables:

1. Define how a host model supplies local elevation-induced pressure boundary changes without conflating hydrostatic atmosphere thickness with terrain geometry.
2. Support host-provided local surface pressure directly and, optionally, a clearly documented pressure-adjustment helper for downscaling from a parent column.
3. Preserve elemental/species/surface-reservoir accounting when local surface pressure differs from the global representative column.
4. Expose diagnostics identifying whether a local column inherited, adjusted or independently prescribed its surface boundary state.
5. Add regression cases spanning mountain/highland and below-datum terrain conditions.

Acceptance gate: local worldgen tiles can obtain physically explicit atmospheric column boundary states rather than applying undocumented pressure heuristics.

## Phase C — Local condensation/cloud/radiation coupling for worldgen tiles

**Status:** planned on top of the existing cloud/transport/radiation backends.

Deliverables:

1. Allow worldgen to batch representative local columns selected from terrain/climate regimes rather than solving one atmosphere per pixel.
2. Return compact quantities useful to local climate/surface refinement:
   - near-surface thermodynamic state;
   - saturation/condensate diagnostics;
   - cloud optical properties/fraction proxies supported by the selected model;
   - shortwave absorption/reflection diagnostics;
   - longwave/thermal flux diagnostics;
   - visible colour/albedo products where physically supported.
3. Preserve explicit fidelity/fallback semantics for every local column.
4. Avoid claiming horizontal wind/orographic precipitation physics inside `atmogen`; those remain host-model processes.
5. Add host-coupling tests where worldgen-like elevation/temperature/forcing perturbations cause physically sensible monotonic changes in eligible atmospheric diagnostics.

Acceptance gate: worldgen can use `atmogen` to refine vertical thermodynamic/cloud/radiative response in selected high-resolution regions without embedding or forking atmosphere physics.

## Phase D — Coupled transport/chemistry fidelity improvements

**Status:** future scientific work.

Deliverables:

1. Couple sourced kinetic reaction networks through the existing conservative vertical eddy-transport operator.
2. Add molecular diffusion/gravitational separation when required for high-altitude columns.
3. Improve condensate source/nucleation/growth/coagulation models with sourced material data.
4. Add phase-appropriate sublimation/vaporization enthalpies so moist convection can cross solid/liquid regimes correctly.
5. Extend beyond the current dilute single-condensable saturated adjustment where data and numerical validation support it.
6. Improve longwave/shortwave transfer toward correlated-k or other spectrally resolved methods without silently inventing missing spectroscopy.
7. Add layerwise radiative heating/flux convergence for higher-fidelity column solutions.

Acceptance gate: fidelity labels continue to correspond to actually implemented physics, with validation/provenance for each newly activated mechanism.

## Phase E — Performance, cacheability and service integration

**Status:** planned.

Deliverables:

1. Add bounded parallel batch execution hooks or a host-neutral execution interface without introducing nondeterministic result ordering.
2. Make large spectral products optional/lazy so terrain viewers can request compact diagnostics without paying full spectral serialization costs.
3. Add compact serialization schemas for worldgen service/viewer inspection.
4. Version all serialized scientific products independently of the internal Python object representation.
5. Add benchmarks for repeated local-column workloads representative of camera-driven regional refinement.

Acceptance gate: regional worldgen refinement can use atmospheric calculations as a bounded, cacheable backend rather than a latency/memory bottleneck.

## Phase F — Cross-repository integration validation

**Status:** continuous.

Deliverables:

1. Maintain an integration test in worldgen against the pinned compatible `atmogen` revision.
2. Validate STANDARD/HIGH thermal-profile behavior through the worldgen adapter, not only inside `atmogen` tests.
3. Add future tests for local-column batching, elevation/pressure coupling and terrain-selected representative columns.
4. Record `atmogen` package/API/data schema versions, database revision hash and compatible git revision in every worldgen run/tileset provenance payload that depends on atmospheric results.
5. Require an explicit compatibility update when worldgen consumes a new `atmogen` API/schema revision.

The guiding boundary is strict: worldgen owns horizontal geography and multiscale terrain/climate orchestration; `atmogen` owns vertical atmosphere/ocean-chemistry/cloud/radiation physics.