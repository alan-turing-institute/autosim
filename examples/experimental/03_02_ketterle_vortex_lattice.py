"""
Description of experiment and references.

Phase 1: Cooling to Ground State in Imaginary Time.
Phase 2: Spinning the trap in Real Time to spawn a vortex lattice.

Reproducing the Core Experimental Mechanics of:
Abo-Shaeer, Raman, Vogels, Ketterle. (2001).
Observation of Vortex Lattices in Bose-Einstein Condensates.
Science. DOI: 10.1126/science.1060182

Two-Phase Ground State Nucleation & Symmetry-Breaking Noise:
To mathematically nucleate the perfect topological ground state during
Phase 1 cooling, the non-unitary rotation operator requires the injection
of microscopic quantum noise (vacuum fluctuations) to break parity symmetry:
- Bao, W. et al. (2006). Dynamics of rotating Bose-Einstein condensates
  and its efficient and accurate numerical computation.
  SIAM J. Appl. Math. DOI: 10.1137/050639050

The Quadrupole Instability (Turbulent Quench):
Without initial noise, the condensate retains zero angular momentum during
cooling. Switching to a rapidly rotating Phase 2 causes the fluid to dynamically
shear along a diagonal and violently emit turbulent vortices instead of a lattice:
- Sinha, S., & Castin, Y. (2001). Dynamic instability of a rotating
  Bose-Einstein condensate.
  Physical Review Letters. DOI: 10.1103/PhysRevLett.87.190402
Modern Flat-Bottom Optical Box Traps:
While early studies used harmonic traps, modern BEC turbulence experiments
use uniform box potentials (wx=0, wy=0) to prevent condensate "squishing"
and study true scale-invariant fluid dynamics.
- Navon, N., Smith, A. L., & Hadzibabic, Z. (2021). Quantum gases in optical boxes.
  Nature Physics. DOI: 10.1038/s41567-021-01403-z
- Kwon, W. J. et al. (2021). Sound emission and annihilations in a programmable quantum
  vortex collider.
  Nature. DOI: 10.1038/s41586-021-04047-4
- Adhikari, S. K. (2019). Vortex-lattice in a uniform Bose-Einstein condensate in a box
  trap.
  J. Phys. Condens. Matter. DOI: 10.1088/1361-648X/ab14c5
"""

import matplotlib.pyplot as plt

from autosim.experimental.simulations import GrossPitaevskiiEquation2D as GPESim


def run_vortex_lattice_experiment():  # noqa: D103
    print("Initializing Ketterle 2001 Vortex Lattice Simulation...")

    # We use the updated GPESim which now supports imaginary_time_steps natively.
    sim = GPESim(
        return_timeseries=True,
        log_level="progress_bar",
        n=128,  # High resolution for sharp vortex cores
        L=10.0,
        T=24.0,  # Duration of Phase 2 (Real Time)
        dt=0.005,
        snapshot_dt=0.2,  # How often to save frames during Phase 2
        parameters_range={
            "g": (150.0, 150.0),  # Strong repulsive interactions
            "wx": (0.0, 0.0),  # Un-squished: REMOVE the harmonic trap entirely!
            "wy": (0.0, 0.0),  # Pure flat-bottomed bucket.
            "Omega": (
                1.2,
                1.2,
            ),  # Spin fast (1.2 rad/s) since there's no harmonic blowout limit!
            "imaginary_time": (0, 0),  # Main simulation is REAL TIME
            "imaginary_time_steps": (
                1000,
                1000,
            ),  # Phase 1: Cool to ground state before Main simulation
            "initial_noise": (
                0.05,
                0.05,
            ),  # Random symmetry-breaking vacuum fluctuations
            "box_param": (
                0.01,
                0.01,
            ),  # Huge box (R_wall = (1/0.01)**0.25 = 3.16 unit radius)
            "box_power": (
                10.0,
                10.0,
            ),  # Steep flat-bottom box walls (power 10) instead of quartic
        },
        random_seed=42,
    )

    print("--- Phase 1 & 2 Execution ---")
    print(
        "The simulation will first run 1000 steps of imaginary time cooling without "
        "rotation,"
    )
    print("then switch to real time and spin the elliptical trap.")

    # Run the simulation (batch size 1)
    res = sim.forward_samples_spatiotemporal(n=1)

    # Data shape is [batch, time, x, y, channels={density, real, imag}]
    density_history = res["data"][0, ..., 0]

    print("Simulation complete! Rendering final frame...")

    # Plot the final frame
    plt.figure(figsize=(6, 6))
    plt.imshow(density_history[-1].cpu().numpy(), cmap="magma", origin="lower")
    plt.colorbar(label="Density")
    plt.title("Vortex Lattice in Rotating Oval Trap (Final Frame)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_vortex_lattice_experiment()
