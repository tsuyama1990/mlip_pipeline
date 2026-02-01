# Architectural Analysis & Refactoring Strategy

## Overview
This document outlines the architectural analysis of the `mlip_autopipec` system, comparing the initial `SPEC.md` with the current codebase implementation. It highlights where pragmatic engineering decisions have superseded rigid specification requirements and defines the strategy for the architectural refactoring.

## 1. Comparative Analysis: Spec vs. Code

### 1.1 Structure Generation
- **Spec Vision**: An "Adaptive Exploration Policy Engine" that uses AI/Decision Models (Policies) to dynamically determine sampling strategies ($R_{MD/MC}$, $T_{schedule}$) based on input features ($E_g$, $B_0$).
- **Current Code**: A "Factory Pattern" implementation (`StructureGenFactory`) that instantiates strategies (`Bulk`, `RandomSlice`, `Defect`, `Strain`) based on static configuration.
- **Decision**: **Prioritize Code/Pragmatism**. Implementing a full AI Policy Engine is premature. The Factory pattern is extensible and sufficient for the current maturity level. The refactoring will focus on ensuring the `StructureGenConfig` is strictly typed and extensible, rather than building the Policy Engine.

### 1.2 The Active Learning Cycle (Halt & Diagnose)
- **Spec Vision**: A "Micro-loop" where a `fix halt` triggers a Python interrupt, which then runs a local optimization loop (Local Candidates -> Local D-Optimality -> Embed -> Train) before *resuming* the MD simulation from the halt point.
- **Current Code**: A "Macro-loop" where the pipeline runs `Explore` (MD), detects Halt/High Uncertainty, stops the current iteration, selects candidates from the trajectory (post-mortem), runs `Refine`, and then starts a *new* iteration.
- **Decision**: **Prioritize Code/Pragmatism**. Resuming LAMMPS simulations after a Python-side retraining loop is complex and prone to state management errors. The Macro-loop (Stop -> Retrain -> Restart) is more robust and statistically sufficient. We will refactor the `Orchestrator` to cleanly separate these phases.

### 1.3 Oracle (DFT)
- **Spec Vision**: "Auto-Magic" self-repair and SSSP pseudopotential discovery.
- **Current Code**: Explicit `DFTConfig` requiring paths to pseudopotentials and specific parameters. Basic retry logic exists in `QERunner`.
- **Decision**: **Prioritize Explicit Config**. Automated external resource discovery (SSSP) adds deployment complexity. We will stick to explicit configuration for reproducibility.

### 1.4 Selection Strategy
- **Spec Vision**: "Local D-Optimality" on generated candidates.
- **Current Code**: Trajectory parsing with Gamma thresholding.
- **Decision**: **Enhance Code**. While full D-Optimality is complex, the current `Orchestrator.select` logic is monolithic. We will extract this into a `SelectionStrategy` (or `Phases` component) to allow future plug-in of D-Optimality without changing the core loop.

## 2. Refactoring Strategy

### 2.1 Domain Models
- **Config**: Consolidate `PotentialConfig` validation logic. Ensure `StructureGenConfig` uses a Discriminated Union for type safety.
- **Potential**: Ensure metadata is sufficient for tracking lineage.

### 2.2 Orchestration
- **Problem**: `Orchestrator.run_pipeline` is becoming a "God Method", mixing high-level flow control with low-level file I/O and component instantiation.
- **Solution**: Decompose `Orchestrator` into a `CyclePhases` component (or set of functions).
    - `PhaseExploration`: Handles StructureGen + MD.
    - `PhaseDetection`: Analyzes results for uncertainty.
    - `PhaseSelection`: Filters trajectories.
    - `PhaseRefinement`: Manages DFT + Training.
    - `PhaseValidation`: Runs the validation suite.
- **Benefit**: The `Orchestrator` becomes a lightweight controller. Testing individual phases becomes easier.

### 2.3 Runners
- **PacemakerRunner**: Ensure `input.yaml` generation is isolated and robust.
- **LammpsRunner**: Ensure `fix halt` and `max_gamma` reporting is reliable.

## 3. Conclusion
The refactoring will align the code structure with the *de facto* "Macro-loop" architecture, cleaning up the schema and decoupling the monolithic Orchestrator. This prepares the system for future extensions (like the AI Policy Engine) without enforcing them now.
