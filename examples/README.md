# Examples

## Experimental

These simulation families are organized from foundational pattern dynamics to increasingly realistic fluid systems, with the goal of supporting flexible spatiotemporal data generation:

### Pattern-formation families

- [Gray-Scott](experimental/00_01_gray_scott.ipynb): A spectral ETDRK4 reaction-diffusion generator that spans diverse morphologies (spots, spirals, worms, and maze-like regimes) via feed/kill parameters.

### Weather-like and transport families

- [Shallow-Water 2D](experimental/01_01_shallow_water_equation.ipynb): A geophysical fluid model that evolves height and horizontal velocity fields to capture wave propagation, rotation effects, and balanced flow structure.

### Classical fluid dynamics families

- [Conditioned Incompressible Navier-Stokes 2D (smoke)](experimental/02_00_conditioned_navier_stokes.ipynb): A buoyancy-driven incompressible flow generator with passive scalar transport and controllable forcing and boundary variants.
- [Lattice Boltzmann (D2Q9 channel flow)](experimental/02_01_lattice_boltzmann.ipynb): A mesoscopic fluid generator with obstacle and oscillatory-inlet scenarios that produce velocity, vorticity, and density channels with weak compressibility effects.
- [Incompressible Hydrodynamics 2D](experimental/02_02_incompressible_hydrodynamics.ipynb): A 2D incompressible Navier-Stokes generator that captures structured velocity-pressure evolution.
- [Compressible Fluid](experimental/02_03_compressible_fluid.ipynb): A compressible-flow generator with density-coupled dynamics that complements incompressible solvers.

### Quantum-fluid dynamics families

- [Gross-Pitaevskii Equation 2D](experimental/03_01_gross_pitaevskii.ipynb): A nonlinear Schrödinger quantum-fluid generator with trap geometry, optional stirring, and disorder controls for vortex and interference dynamics.
