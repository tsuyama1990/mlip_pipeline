# Cycle 05 UAT: The Watchdog

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-05-01** | High | **Safety Halt** | Verify that the simulation stops immediately when the extrapolation grade exceeds the threshold. |
| **UAT-05-02** | Medium | **Hybrid Safety** | Verify that atoms do not overlap (nuclear fusion) even when the ACE potential is untrained/random, thanks to the ZBL baseline. |

## 2. Behavior Definitions

### UAT-05-01: Safety Halt

**GIVEN** a trained potential and a `gamma_threshold` of 5.0
**WHEN** an MD simulation is started
**AND** the system encounters a configuration where `gamma` spikes to 10.0
**THEN** LAMMPS should exit with a specific non-zero code or log message ("Fix halt condition met")
**AND** the `OTFManager` should catch this event
**AND** the final structure should be saved as `halt_structure.xyz`

### UAT-05-02: Hybrid Safety

**GIVEN** two atoms placed 0.5 Angstroms apart (unphysically close)
**AND** a `HybridPotential` (ACE + ZBL)
**WHEN** the energy is calculated
**THEN** the energy should be extremely positive (Repulsive)
**AND** this should be true even if the ACE part predicts zero or negative energy (checking that ZBL dominates at short range)
