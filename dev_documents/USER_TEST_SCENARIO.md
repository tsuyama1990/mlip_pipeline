# pyacemaker ユーザー受け入れテスト (UAT) シナリオ
**対象バージョン:** CYCLE07 (MACE Distillation Architecture)

## テストケース概要: 有機反応の遷移状態（TS）エネルギー障壁の再現

本テストでは、開発した「MACE Knowledge Distillation & Delta Learning」パイプラインを用いて、代表的な有機物の化学反応である **SN2反応（例: $CH_3Cl + OH^- \rightarrow CH_3OH + Cl^-$）** のポテンシャルを自動構築する。

最終的に生成された多項式ポテンシャル（ACE）を用いてNEB（Nudged Elastic Band）計算を行い、第一原理計算（DFT）で求めた真の活性化エネルギー（Ea）を、極小のDFT計算コストでどこまで高精度に再現できるかを検証する。

---

## 1. 前提条件 (Prerequisites)
* `pyacemaker` がインストールされ、VASP（または対象のDFTソルバー）および MACE-MP-0 モデルへのアクセスが設定されていること。
* 初期構造（Reactant: 反応前）と終状態（Product: 反応後）の `.xyz` または `POSCAR` ファイルが用意されていること。

## 2. 実行手順 (Execution Steps)

### Step 1: テスト用コンフィグの作成
以下の `config.yaml` を作業ディレクトリに配置する。

```yaml
system:
  elements: ["C", "H", "O", "Cl"]
  base_directory: "./uat_sn2_reaction"

distillation:
  enable_mace_distillation: true
  
  step1_direct_sampling:
    target_points: 50       # 反応経路周辺の相空間をサンプリング
    objective: "maximize_entropy"
  
  step2_active_learning:
    uncertainty_threshold: 0.1  # MACEの予測分散がこの値以上の構造（TS付近を想定）を抽出
    dft_calculator: "VASP"
  
  step3_mace_finetune:
    base_model: "MACE-MP-0"
    epochs: 30

  step4_surrogate_sampling:
    target_points: 500
    method: "mace_md"       # 高温MDで反応経路周辺の多様な構造を生成

  step7_pacemaker_finetune:
    enable: true
    weight_dft: 10.0        # TS付近のDFT真値データを強力に重み付けしてACEを補正