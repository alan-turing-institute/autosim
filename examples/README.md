# Examples

## Spatiotemporal

These examples use stable simulators from `autosim.simulations.spatiotemporal`.
They cover reaction-diffusion systems, fluid transport, conditioned
smoke-flow Navier-Stokes, and Gross-Pitaevskii quantum-fluid dynamics.

### Pattern-formation families

- [Reaction-Diffusion](spatiotemporal/reaction_diffusion.ipynb): A spectral (FFT-based) two-species reaction-diffusion generator that produces diverse spatiotemporal patterns — spirals, spots, and labyrinthine textures — across reaction and diffusion parameter regimes.
- [Gray-Scott](spatiotemporal/gray_scott.ipynb): A spectral ETDRK4 reaction-diffusion generator that spans diverse morphologies (spots, spirals, worms, and maze-like regimes) via feed/kill parameters.

### Weather-like and transport families

- [Advection-Diffusion](spatiotemporal/advection_diffusion.ipynb): A 2D incompressible vorticity–streamfunction solver with spectral Poisson inversion that generates vorticity fields across a range of viscosities and forcing strengths.

### Classical fluid dynamics families

- [Conditioned Incompressible Navier-Stokes 2D (smoke)](spatiotemporal/conditioned_navier_stokes.ipynb): A buoyancy-driven incompressible flow generator with passive scalar transport and controllable forcing and boundary variants.

### Quantum-fluid dynamics families

- [Gross-Pitaevskii Equation 2D](spatiotemporal/gross_pitaevskii.ipynb): A nonlinear Schrödinger quantum-fluid generator with trap geometry, optional stirring, and disorder controls for vortex and interference dynamics.
- [Box Vortex Lattice](spatiotemporal/box_vortex_lattice.py): A script for generating GPE vortex-lattice data from a rotating box-trap setup.

## Experimental

These examples use simulators that are still under `autosim.experimental`.

### Weather-like and transport families

- [Shallow-Water 2D](experimental/01_01_shallow_water_equation.ipynb): A geophysical fluid model that evolves height and horizontal velocity fields to capture wave propagation, rotation effects, and balanced flow structure.

### Classical fluid dynamics families

- [Lattice Boltzmann (D2Q9 channel flow)](experimental/02_01_lattice_boltzmann.ipynb): A mesoscopic fluid generator with obstacle and oscillatory-inlet scenarios that produce velocity, vorticity, and density channels with weak compressibility effects.
- [Incompressible Hydrodynamics 2D](experimental/02_02_incompressible_hydrodynamics.ipynb): A 2D incompressible Navier-Stokes generator that captures structured velocity-pressure evolution.
- [Compressible Fluid](experimental/02_03_compressible_fluid.ipynb): A compressible-flow generator with density-coupled dynamics that complements incompressible solvers.
