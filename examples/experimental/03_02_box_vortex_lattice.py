"""
Description of experiment and references.

Phase 1: Cooling to Ground State in Imaginary Time.
Phase 2: Spinning the trap in Real Time to spawn a vortex lattice.

Reproducing the Core Experimental Mechanics of Modern Box Traps:
- Navon, N., Smith, A. L., & Hadzibabic, Z. (2021). Quantum gases in optical boxes.
  Nature Physics. DOI: 10.1038/s41567-021-01403-z
- Adhikari, S. K. (2019). Vortex-lattice in a uniform Bose-Einstein condensate in a box
  trap.
  J. Phys. Condens. Matter. DOI: 10.1088/1361-648X/ab14c5

Historical Context (Ketterle 2001):
The original observation of vortex lattices (Abo-Shaeer et al., Science 2001) used a
magnetic harmonic bowl rather than the modern optical Woods-Saxon box simulated here.
In a harmonic trap, the centrifugal force of rotation fights the trap's inward
restoring force, meaning Omega cannot exceed the harmonic trapping frequency.
In this script, the harmonic trap is explicitly switched off (wx=0, wy=0) and replaced
with a pure Flat-Bottom Box, allowing us to spin safely past the harmonic limits!

For reference, the historical Ketterle harmonic parameters look like this:
    "wx": (1.0, 1.0),                 # Symmetric Base
    "wy": (1.1, 1.1),                 # 10% Elliptical deformation for stirring
    "Omega": (0.75, 0.75),            # Sub-blowout rotation (Omega < wx)
    "box_type": "power",              # Standard quartic background wall

Two-Phase Ground State Nucleation & Symmetry-Breaking Noise:
To mathematically nucleate the perfect topological ground state during
Phase 1 cooling, the non-unitary rotation operator requires the injection
of microscopic quantum noise (vacuum fluctuations) to break parity symmetry:
- Bao, W. et al. (2006). Dynamics of rotating Bose-Einstein condensates.
  SIAM J. Appl. Math. DOI: 10.1137/050639050

The Quadrupole Instability (Turbulent Quench):
Without initial noise, the condensate retains zero angular momentum during
cooling. Switching to a rapidly rotating Phase 2 causes the fluid to dynamically
shear along a diagonal and violently emit turbulent vortices instead of a lattice:
- Sinha, S., & Castin, Y. (2001). Dynamic instability of a rotating
  Bose-Einstein condensate.
"""

import matplotlib.pyplot as plt

from autosim.experimental.simulations import GrossPitaevskiiEquation2D as GPESim


def run_box_vortex_lattice_experiment():  # noqa: D103
    print("Initializing Uniform Box Vortex Lattice Simulation...")

    # We use the updated GPESim which now supports imaginary_time_steps natively.
    sim = GPESim(
        return_timeseries=True,
        log_level="progress_bar",
        n=128,  # High resolution for sharp vortex cores
        L=10.0,
        T=24.0,  # Duration of Phase 2 (Real Time)
        dt=0.005,
        snapshot_dt=0.2,  # How often to save frames during Phase 2
        box_type="woods_saxon",
        parameters_range={
            "g": (500.0, 500.0),  # High g prevents evacuation
            "wx": (0.0, 0.0),  # Un-squished: REMOVE the harmonic trap entirely!
            "wy": (0.0, 0.0),  # Pure flat-bottomed bucket.
            "Omega": (0.85, 0.85),  # Spin fast to spawn lattice
            "imaginary_time": (0, 0),  # Main simulation is REAL TIME
            "imaginary_time_steps": (1000, 1000),  # Phase 1: Cool to ground state
            "initial_noise": (0.05, 0.05),  # Random vacuum fluctuations
            "box_param": (0.0039, 0.0039),  # Wall at R=4.0
            "box_anisotropy": (1.06, 1.06),  # Slight oval Woods-Saxon spoon
            "ws_a": (0.15, 0.15),  # Optical laser diffraction sharpness
            "ws_V0": (200.0, 200.0),  # Laser barrier height
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
    plt.title("Vortex Lattice in Uniform Woods-Saxon Box")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_box_vortex_lattice_experiment()
