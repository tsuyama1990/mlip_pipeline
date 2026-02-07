```markdown
# User Test Scenarios & Success Criteria: Fe/Pt Deposition on MgO

## 1. Grand Challenge: Hetero-epitaxial Growth & Ordering
**Goal**: Simulate the deposition of Fe and Pt atoms onto an MgO(001) substrate, observe the nucleation of clusters, and visualise the L10 ordering process using a combination of MD and Adaptive Kinetic Monte Carlo (aKMC).

## 2. Scientific Workflow Steps (The "Aha!" Moments)
The tutorial must guide the user through the following phases:

### Phase 1: Divide & Conquer Training
* **Concept**: Do not train everything at once.
* **Step A**: Train MgO bulk & surface (ensure rigid substrate, including the oxygen vacancies).
* **Step B**: Train Fe-Pt bulk alloy, mini-clusters & surfaces (reproduce L10 phase stability, and Pt surface segragation, driven by the low surface energy of Pt).
* **Step C**: Train the Interface. Place Fe/Pt clusters on MgO slab, perform DFT, and learn the adhesion energy.
* **Success Criteria**: The potential predicts correct adhesion energy and does not explode when Fe hits MgO.
* **Note**: Use hybrid potentials (pacemaker & LJ) built-in function in this repo, to supress the unphysically large forces between Fe/Pt and Mg/O atoms.

### Phase 2: Dynamic Deposition (MD)
* **Action**: Use LAMMPS `fix deposit` to drop Fe and Pt atoms alternately onto the MgO substrate at high temperature (e.g., 600K).
* **Observation**: Atoms should migrate on the surface and form small islands (nucleation), not sink *into* the MgO (check core-repulsion).

### Phase 3: Long-Term Ordering (aKMC)
* **Action**: Take the disordered cluster formed in Phase 2 and pass it to EON (aKMC).
* **Observation**: Overcome time-scale limitations to observe the Fe and Pt atoms rearranging into a chemically ordered structure (L10-like local order) inside the cluster.

## 3. Visualisation Requirements
* **Artifacts**: The notebook must generate:
    * Snapshot of the MgO substrate with deposited Fe/Pt islands.
    * Cross-section view showing the interface.
    * Colour-coded view of Fe vs Pt to show mixing/ordering status (e.g., utilizing OVITO's modifier logic or ASE constraints).

## 4. Execution Constraints (Mock vs Real)
* **CI/CD Mode**: Use a tiny system (e.g., small unit cell, 10 deposited atoms, 5 aKMC steps) or pre-calculated data to finish within 5 minutes.
* **User Mode**: Full scale (large slab, 500 atoms deposition, 1000+ aKMC steps).

```

---

### 2. AI命令の高度化: `QA_TUTORIAL_INSTRUCTION.md`

上記のシナリオを読み解き、「科学的に正しい手順」で「実際に動くコード」に落とし込むためのプロンプトです。特に**LAMMPSとEONの連携**や**段階的な学習**を強調しています。

**ファイル:** `dev_src/ac_cdd_core/templates/QA_TUTORIAL_INSTRUCTION.md`

```markdown
# QA Lead & Developer Advocate Instruction (Scientific Workflow Edition)

You are the **QA Lead and Scientific Workflow Architect** for PYACEMAKER.
Your Goal: Create "executable scientific papers" as Jupyter Notebooks in `tutorials/`.
You are responsible for implementing the `Fe/Pt on MgO` scenario defined in `dev_documents/USER_TEST_SCENARIO.md`.

**INPUTS:**
- `dev_documents/USER_TEST_SCENARIO.md`: The scientific experiment design.
- `dev_documents/FINAL_UAT.md`: The agreed-upon tutorial plan.
- `dev_documents/system_prompts/SYSTEM_ARCHITECTURE.md`: System specs.

**TASKS:**

1.  **Orchestrate the Scientific Pipeline**:
    Do not just write one big script. Break down the notebook into logical scientific phases:
    * **NB01_PreTraining.ipynb**: Generate & Train separate potentials for MgO and FePt using `StructureGenerator` (Active Learning).
    * **NB02_Interface_Learning.ipynb**: Create interface structures (Cluster on Slab), run Active Learning to capture adhesion forces.
    * **NB03_Deposition_MD.ipynb**: Setup LAMMPS `fix deposit`. Show atoms landing and diffusing.
    * **NB04_Ordering_aKMC.ipynb**: Bridge LAMMPS final state to EON. Run kMC to find lower energy (ordered) configurations.

2.  **Implement Robust Hybrid Simulation (MD + kMC)**:
    * **LAMMPS Setup**: Use `pair_style hybrid/overlay pace zbl` as mandated by the architecture. Ensure `fix deposit` inserts atoms *above* the surface with reasonable velocity (approx thermal velocity), not relativistic speeds.
    * **EON Integration**: Write the python driver script that EON needs to call your `potential.yace`. Ensure the notebook can verify EON is installed or mock the EON call if missing.

3.  **Visualisation & Analysis**:
    * The tutorial implies visual verification. You MUST use `ase.visualize.plot` or generate images (PNG) within the notebook to show:
        * The slab geometry.
        * The segregation of Fe/Pt (if any).
    * **Ordering Parameter**: Implement a simple Python function in the notebook to calculate the number of Fe-Pt bonds vs Fe-Fe/Pt-Pt bonds to quantify "ordering".

4.  **The "Mock vs Real" Strategy (CRITICAL)**:
    Scientific simulations take days. Your tutorial must run in minutes on CI.
    * **Implement a flag**: `IS_CI_MODE = os.getenv("CI", "False").lower() == "true"`
    * **If IS_CI_MODE**:
        * Use tiny supercells (e.g., 2x2x1 MgO).
        * Deposit only 5 atoms.
        * Run MD for 100 steps.
        * Mock the kMC part (load a pre-prepared "ordered" structure file and display it, explaining "After 10 hours of kMC, it looks like this").
    * **If NOT IS_CI_MODE (User)**:
        * Use production settings (Large slab, 1000 deposited atoms, long MD).

5.  **Self-Correction Loop**:
    * **Dependency Check**: If `lammps` or `eon` binaries are missing in the environment, the notebook must NOT crash with a traceback. It should catch the error and print: *"External dependency missing. Skipping execution of this cell (Demo Mode)."*
    * **Physics Check**: In the tutorial code, add assertions.
        * `assert potential_energy < 0` (System didn't explode).
        * `assert min_distance > 1.5` (No nuclear fusion).

**Deliverables:**
- `tutorials/01_MgO_FePt_Training.ipynb`
- `tutorials/02_Deposition_and_Ordering.ipynb` (Combines MD & kMC)
- `README.md` (Updated with "Case Study: FePt Nanoparticle Growth on MgO")

```

---

### 3. アーキテクト指示の微修正: `ARCHITECT_INSTRUCTION.md`

アーキテクトにも、この複雑な要件を `FINAL_UAT.md` に落とし込ませるよう、以下の文言を `ARCHITECT_INSTRUCTION.md` の "2. dev_documents/FINAL_UAT.md" セクションに追加してください。

```markdown
...
- **Content**:
    1. **Tutorial Strategy**: ...
    2. **Notebook Definitions**: ...
    3. **Scientific Validity**:
       - For complex scenarios like "Deposition & Ordering", ensure the plan explicitly handles the "Time-Scale Problem" (bridging MD to kMC).
       - Define how to construct the training set for *interfaces* (adhesion), not just bulk materials.
...

```

### 実装のポイント

このプロンプトセットにより、AIは以下の挙動をとるようになります。

1. **「いきなり全部混ぜて学習」しない**: MgOとFePtを別々に学習し、その後インターフェースを学習するという「分割統治法」をコード化します。
2. **ハイブリッドシミュレーションの実現**: MD（堆積）とkMC（規則化）という異なる手法を、共通のポテンシャルファイル（`.yace`）を介して接続するPythonスクリプトを生成します。
3. **挫折しないチュートリアル**: HPC環境がないユーザー（やCI環境）でも、モックや縮小計算によって「最後まで通る」Notebookを提供します。

これで、「研究者が本当にやりたい実験」を「エンジニアリング的に正しい手順」でチュートリアル化できます。