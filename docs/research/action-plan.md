# Digital Life: Action Plan & Specification

> **Document Role:** Action Plan & Technical Specification (3/3)
> This document was created after considering both the initial proposition and the peer review, combined with an in-depth interview with the lead researcher to resolve open questions and make key architectural decisions.
> - Based on: [`digital-life-project-overview.md`](digital-life-project-overview.md) — the initial project proposition
> - Addresses: [`unified-review.md`](unified-review.md) — all critical/high/medium issues raised in the peer review
> - Revised: 2026-02-10 — incorporates second-round review feedback on validation bias, reproducibility, and statistical design
> - Revised: 2026-02-10 — incorporates Gemini external review: LLM scoped to ablation study, added blind spots (genotype encoding, thermodynamics, debugger), parallel writing strategy
> - Revised: 2026-02-10 — incorporates multi-framework analysis (First Principles, Inversion, Systems Thinking, Six Thinking Hats): functional analogy definition, criterion-ablation as core experiment, Day 1 feasibility spike, Week 4 paper milestone, integration regression tests

## Interview Summary

| 項目 | 決定事項 |
|------|---------|
| 7基準の根拠 | 教科書的定義を採用（理論的正当化は後から補強） |
| ALife立場 | **Weak ALife** — 生命の機能的モデルとして位置づける |
| アーキテクチャ | **ハイブリッド** — swarmエージェントがorganism的上位構造を形成する二層構造 |
| スケール | 小規模: 10-50エージェント/organism、環境に10-50 organisms |
| 投稿先 | **ALIFE 2026 Full Paper (8p)**、締切 **要確認: 公式サイトでは3/30の可能性あり（4/1と齟齬）** |
| 計算環境 | Mac Mini M2 Pro |
| 実装言語 | **Python + Rust ハイブリッド**（コアシミュレーション: Rust、実験管理/分析: Python） |
| エージェントの脳 | **進化的NNコントローラー**（主軸）。ローカルLLMはablation study（1実験）のみ |
| 代謝リソース | **未定** — Phase 1で実験的に決定 |
| 環境 | 連続2D空間 |
| ビジュアル | 中程度（基本的な2D描画、論文用figure品質） |
| 知覚評価 | 非公式（同僚・知人によるフィードバック） |
| 論文焦点 | **システム論文** — 実装したシステムの振る舞いと創発的現象が主 |
| 時間投入 | フルタイム（週40時間以上） |
| 研究体制 | ドメイン専門家の協力あり |
| 現在の進捗 | ゼロ（概要文書のみ） |
| ピボット戦略 | 未検討（本計画で策定する） |

---

## レビュー指摘への対応方針

### C1. 理論的基盤 → Day 1-3で「Functional Analogy」の操作的定義を確立

教科書的7基準を出発点としつつ、**コードを書く前に**以下を文書化する:

**最優先 (Day 1-3): 「Functional Analogy」の操作的定義**

> 計算プロセスが生物学的基準の**機能的アナロジー**であるとは、以下の3条件すべてを満たす場合をいう:
> (a) そのプロセスが持続的リソース消費を要する**動的プロセス**であること
> (b) その除去がorganismの自己維持に**測定可能な劣化**を引き起こすこと（→ criterion-ablation実験で検証）
> (c) 他の少なくとも1つの基準と**フィードバックループ**を形成していること

この定義により「simplified proxy」との差異が明確になる: proxyは条件(b)(c)を満たさない（除去しても他の基準に影響しない独立モジュール）。

**この定義を各Go/No-Goチェックポイントで検証する**: 「この時点の実装について、なぜfunctional analogyでありsimplified proxyではないかを1段落で説明できるか？」

**追加の文書化**:
1. **基準体系の比較表**: 7基準 vs. NASA定義 vs. オートポイエーシス vs. Ruiz-Mirazo条件
2. **7基準を採用する理由**: 「最も広く認知された基準であり、各基準が独立した計算プロセスとして実装可能」
3. **Weak ALifeとしてのframing**: 「生命そのもの」ではなく「生命の機能的性質を再現するモデル」として位置づける
4. **オートポイエーシスとの接続**: active boundary + metabolism = 計算的オートポイエーシス

**担当**: ドメイン専門家と協議の上、論文のIntroduction/Related Workセクションに組み込む

### C2. 評価指標 → calibration/test分離 + 統計計画

閾値決定と本評価を同一データで行う循環バイアスを防ぐため、データを明確に分離する:

**データ分離プロトコル:**
1. **Calibration set** (Phase 1-2, Week 2-4): 異なるランダムシード群（seeds 0-99）で予備実験を実施。閾値・摂動強度・N値をここで決定
2. **Final test set** (Phase 4, Week 7): 未使用のシード群（seeds 100-199）で本評価を実施。calibrationで決定した閾値を固定適用
3. calibrationとtestでシード・初期条件・環境配置を完全に分離し、論文に分離手順を明記

**統計計画:**
- 検定手法: Mann-Whitney U検定（ランダムベースライン vs. 本システム）
- サンプルサイズ: 各条件 n≥30 試行（Cohen's d=0.8, α=0.05, power=0.8 で必要な最小サンプル）
- 効果量: Cohen's d を報告
- 複数比較補正: 7基準を同時検定するため Holm-Bonferroni 法を適用
- 除外基準: シミュレーションが初期化後10ステップ以内にクラッシュした試行は除外し、除外率を報告

### H1. 締切修正 → 即時確認が必要

**要確認**: 公式サイト (https://2026.alife.org) で正確な締切を確認すること。文書間で4/6、4/1、3/30と齟齬あり。本計画は暫定的に **2026-04-01** をベースとするが、3/30の場合は全マイルストーンを2日前倒しする。

### H2. 比較設計 → 文献ベース比較に正式固定

既存システムの定量再実装は8週間では非現実的なため、**文献ベース比較**を正式方針とする。

1. Phase 0で比較対象システムの7基準充足度マトリクスを作成
2. 評価方法: 各システムの基準ごとの実装深度を5段階で評価（**文献の記述に基づく**、再実装は行わない）
3. **ルーブリック定義** (恣意性排除のため):
   - 1: 基準に該当する機能なし
   - 2: 静的パラメータとして存在（例: 固定エネルギー値）
   - 3: 動的だが単一プロセス（例: エネルギー消費のみ）
   - 4: 複数プロセスの相互作用あり（例: 代謝経路の分岐）
   - 5: 自己維持的・創発的プロセス（例: 代謝ネットワークの自律調整）
4. **評価者間一致性 (inter-rater reliability)**: 著者 + ドメイン専門家の2名以上が独立評価し、Cohen's κ を報告。κ < 0.6 の基準は評価基準を再定義

### H3. 統合リスク → ピボット戦略を策定

統合リスクを **4.0/5** に上方修正。以下のピボット戦略を設定:

| 状況 | ピボット |
|------|---------|
| 代謝ネットワークが1000ステップ維持不可 | グラフベース → 連続力学系ベース（ODEベース代謝） |
| ハイブリッド二層構造が不安定 | Swarmを落とし、Agent-basedに単純化 |
| 7基準統合が間に合わない | 論文を「代謝+恒常性+細胞組織化」の3基準に絞り、残りは将来研究 |
| Criterion-ablation実験で一部基準が有意差なし | 有意な基準のみで論文を構成。非有意基準はlimitationとして報告 |

### H4. スケジュール → 8週間計画（下記）

### M1. アーキテクチャ → ハイブリッドに決定（上記）

### M2. オートポイエーシス → Phase 0の文献調査に含める

### M3. 投稿形態 → Full Paper を目指す（ピボットとしてExtended Abstract）

### L1-L3 → 論文執筆時に対処

---

## 7.5週間スケジュール（2026-02-10 → 2026-04-01）

### Week 1 (02/10-02/16): Phase 0 — 理論基盤 + 環境構築

**目標**: 理論的正当化の確立 + 計算実現可能性の検証 + 開発環境の準備

**Day 1 (02/10) — 最優先タスク:**
- [ ] **ALIFE 2026 公式締切の確認** (https://2026.alife.org)。3/30なら全マイルストーン2日前倒し
- [ ] **計算実現可能性スパイク (4時間)**: Rustで2,500エージェント（50 org × 50 agents）+ trivial NN + 空間近接クエリをベンチマーク。目標: >100 timesteps/sec。Rustが未経験なら+2時間でツールチェーンセットアップを含む
- [ ] **論文スケルトン (1ページ)**: タイトル、貢献文（3文）、主張構造、計画figureリスト、セクション構成

**Day 1-3 (02/10-02/12) — 理論基盤:**
- [ ] **「Functional Analogy」の操作的定義** (1ページ、上記C1参照) → ドメイン専門家レビュー
- [ ] 基準体系比較表の作成（7基準 vs. NASA vs. オートポイエーシス vs. Ruiz-Mirazo）
- [ ] **Criterion-ablation実験プロトコルの設計**: 各基準の「無効化」方法を事前定義（下記「Criterion-Ablation」参照）

**Day 4-7 (02/13-02/16) — 環境構築:**
- [ ] Rust + Python プロジェクトセットアップ（PyO3 or maturin）
- [ ] 連続2D環境の基本フレームワーク実装
- [ ] 再現性基盤の構築（下記「再現性仕様」参照）
- [ ] **リアルタイムビジュアルデバッガー**の実装（egui or macroquad。開発効率に直結）
- [ ] **Genotype-Phenotype マッピング**の設計（direct encoding + 冗長領域。**7基準すべてに対応する構造で設計**し、初期は2-3基準のみアクティブ。未使用セグメントはゼロ初期化）
- [ ] 既存システム比較表の作成 + ルーブリック定義（Polyworld, ALIEN, Flow-Lenia, Avida, Lenia, Coralai）

**Go/No-Go (Day 1, スパイク後)**:
- 2,500エージェントが >100 timesteps/sec で動作するか？
- No → アーキテクチャ再検討（エージェント数削減 or GPU活用）

**Go/No-Go (Week 1末)**:
- 進化的NNコントローラーの基本動作確認（50エージェントで実用速度か？）

### Week 2-3 (02/17-03/02): Phase 1A — 代謝 + 細胞組織化 (calibration set)

**目標**: 単一organismが自律的に維持される最小システム

**並行執筆**: Methods セクション（システム設計の記述）を実装と同時に草稿

**代謝**:
- [ ] リソースタイプの決定（予備実験で3種類を比較テスト）
  - 候補A: 抽象エネルギートークン（シンプル、制御しやすい）
  - 候補B: 空間的リソース（光、栄養素の空間分布）
  - 候補C: 情報リソース（環境からの情報を処理して有用な形に変換）
- [ ] グラフベース代謝ネットワーク実装（Rustコア）
- [ ] **熱力学的制約**: エネルギー損失率、廃棄物蓄積、エントロピー増大の組み込み
- [ ] 代謝ネットワークの遺伝的エンコーディング（Week 1で設計したGenotype-Phenotypeマッピング適用）
- [ ] 廃棄物排出メカニズム

**細胞組織化**:
- [ ] swarmエージェントの基本実装（位置、速度、状態）
- [ ] アクティブ境界維持プロセス（代謝コスト消費）
- [ ] 境界崩壊メカニズム（維持コストが払えないと溶解）

**Go/No-Go (Week 2 Day 3 = 02/19)**:
- Swarm境界が安定しているか？
- No → 単一エージェント型の円形メンブレンにフォールバック

**統合リグレッションテスト (Week 2以降、各基準追加後に実行)**:
- [ ] 「organism 1000ステップ生存」テストが全既存基準で引き続きパスするか確認

**Go/No-Go (Week 3末)**:
- 単一organismが1000タイムステップ自律維持できるか？
- functional analogy定義の条件(b): 代謝を無効化 → boundary崩壊が観察されるか？（相互依存性の初期検証）
- No → 代謝をODEベースに切り替え（ピボットA）
- No (代謝+境界の両方が不安定) → 手動設計による安定システムで「System Design論文」にピボット

### Week 4 (03/03-03/09): Phase 1B — 恒常性 + 刺激応答 (calibration set)

**目標**: 環境摂動からの回復能力

**並行執筆**: Experimental Setup セクション（環境条件、摂動プロトコル）を草稿

- [ ] 内部状態ベクトルの実装（代謝スループット、境界完全性、リソースレベル）
- [ ] 進化的NNコントローラーによる恒常性調整
- [ ] 感覚入力の実装（リソース勾配、他organism検出、環境条件）
- [ ] 環境摂動テスト（リソース枯渇、温度変化的イベント）
- [ ] 恒常性メトリクス: 摂動後の回復時間、内部状態の分散

**統合リグレッションテスト**: 恒常性追加後、代謝+境界の1000ステップテストが引き続きパスするか確認

**Week 4 論文マイルストーン（ハード）**:
- [ ] 論文スケルトンを3-4ページに拡張: Methods + Experimental Setupが実質完成していること
- **未達成 → Extended Abstract (2-4p) にピボットを即決定**。判断を先送りしない。

**Go/No-Go (Week 4末)**:
- organismが摂動から回復するか？
- functional analogy条件(c): 恒常性 ↔ 代謝間のフィードバックループが観察されるか？
- No → コントローラーを手動設計し、進化は後回し

### Week 5 (03/10-03/16): Phase 2 — 成長/発生 + 生殖

**目標**: ライフサイクルの完成

**成長/発生**:
- [ ] 最小シード → 成熟体への発生プログラム
- [ ] 代謝ネットワークの段階的拡張（成長に代謝コスト）
- [ ] swarmサイズの成長（新エージェントの追加/分化）

**生殖**:
- [ ] 代謝リソース蓄積による生殖条件
- [ ] ゲノムコピー + 変異 → 子organismの生成
- [ ] 子organismはシードから発生開始
- [ ] 生殖コスト（親のリソース分割）

### Week 6 (03/17-03/23): Phase 3 — 進化 + 統合

**目標**: 集団進化の動作確認

**並行執筆**: Related Work セクション（既存システム比較表を含む）を草稿

- [ ] 複数organism（10-50体）の同時シミュレーション
- [ ] 遺伝的変異（点変異、挿入/欠失、重複）
- [ ] 環境変動（リソース分布の変化、新たな摂動タイプ）
- [ ] 世代交代と自然選択の確認
- [ ] 適応度の世代推移トラッキング

**Go/No-Go (Week 6末)**:
- 7基準すべてが同時に機能しているか？
- No → 機能している基準に絞り、論文のスコープを調整（ピボットC）

### Week 7 (03/24-03/28): Phase 4 — 評価 + ビジュアライゼーション (final test set)

**目標**: Criterion-ablation実験 + 未使用シードによる本評価 + 論文用figure

- [ ] **Criterion-ablation実験** (コア実験): 各基準を個別に無効化し、システム劣化を測定（下記「Criterion-Ablation」参照）。ablation条件 × n≥30シード = 本評価と並行実行
- [ ] **Final test set** (seeds 100-199) による本評価の実施。calibrationで固定した閾値を適用
- [ ] 各基準の統計検定（Mann-Whitney U, Holm-Bonferroni補正, Cohen's d報告）
- [ ] 既存システムとの文献ベース比較表の完成（ルーブリック適用、inter-rater κ算出）
- [ ] **Killer figure作成**: 7生物学的基準 → 7計算的実装の対応マップ
- [ ] 非公式知覚評価の実施（同僚5-10名にデモを見せる）
- [ ] 論文用figure生成（シミュレーションスナップショット、メトリクス推移グラフ、**ablation結果表**）
- [ ] 動画作成（補足資料用）
- [ ] LLM ablation study（ストレッチゴール: 上記が全て完了した場合のみ）

### Week 7.5-8 (03/29-04/01): Phase 5 — 論文仕上げ

**目標**: Full Paper (8p) 完成 + 投稿

並行執筆済みセクション（Methods, Experimental Setup, Related Work）を統合し、残りを執筆:

- [ ] Abstract + Introduction（理論的基盤、C1対応）
- [ ] Results（定量評価結果 + **criterion-ablation結果**、C2対応）
- [ ] Discussion（限界、LLM ablation結果（あれば）、epistemological scope、将来研究）
- [ ] 参考文献の書誌情報完備（L1対応）
- [ ] 全体の統合・推敲
- [ ] ドメイン専門家によるレビュー
- [ ] **04/01 投稿**

**ピボット**: 間に合わない場合 → Extended Abstract (2-4p) に切り替え

---

## 技術アーキテクチャ概要

```
┌─────────────────────────────────────────────┐
│                 Environment                  │
│            (連続2D空間, Rust実装)              │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Organism A│  │Organism B│  │Organism C│   │
│  │┌────────┐│  │          │  │          │   │
│  ││ Swarm  ││  │  ...     │  │  ...     │   │
│  ││Agents  ││  │          │  │          │   │
│  ││(10-50) ││  │          │  │          │   │
│  │├────────┤│  └──────────┘  └──────────┘   │
│  ││Metabol.││                                │
│  ││Network ││  リソース分布                    │
│  │├────────┤│  (光、栄養素等)                  │
│  ││Neural  ││                                │
│  ││Control ││                                │
│  │├────────┤│                                │
│  ││Genome  ││                                │
│  │└────────┘│                                │
│  └──────────┘                                │
└─────────────────────────────────────────────┘

各Organism内部:
- Swarm Agents: 境界維持、感覚、運動を担う個別エージェント群
- Metabolic Network: グラフベースの代謝経路（遺伝的にエンコード）
- Neural Controller: 恒常性調整 + 行動制御（進化的NN。LLMはablation studyのみ）
- Genome: 代謝ネットワーク + 発生プログラム + NNアーキテクチャを符号化
```

---

## リスクマトリクス（更新版）

| リスク | 深刻度 | 確率 | ピボット |
|--------|:------:|:----:|---------|
| 代謝ネットワークが維持不可 | 高 | 中 | ODEベース代謝に切替 |
| ハイブリッド二層が不安定 | 高 | 中 | Agent-basedに単純化 |
| LLM ablation studyが実施不可 | 低 | 中 | 省略（将来研究に記載。criterion-ablationが主軸なので影響小） |
| Swarm境界が不安定 (Week 2 Day 3) | 中 | 中 | 単一エージェント型円形メンブレンにフォールバック |
| 7基準統合が間に合わない | 高 | 中 | 3-5基準に絞る |
| 8週間でFull Paper完成不可 | 中 | 中 | Extended Abstractに切替 |
| M2 Proで集団進化が回らない | 中 | 低 | 集団サイズ縮小 or クラウドGPU |

---

## 成功基準

**論文の主張ライン（claim line）**: 投稿時に以下のいずれかを採択判定の主軸とする。ピボット発生時は下位の主張ラインに切り替え、論文のスコープを一貫させる。

### Tier 1: 最低限（Extended Abstract相当）
- **主張**: 「代謝・恒常性・細胞組織化の3基準について、既存システムより深い機能的アナロジーを実現した」
- 3基準以上が機能的に動作
- 単一organismの自律維持 >1000ステップ
- calibration/test分離による統計的検証

### Tier 2: 目標（Full Paper相当）
- **主張**: 「7基準すべてを統合した初のシステムを構築し、criterion-ablation実験により各基準が機能的に必要かつ相互依存であることを定量的に示した」
- 7基準すべてが同時に機能
- **criterion-ablation**: 各基準の除去がシステム劣化を引き起こすことを統計的に確認
- 集団進化で適応が観察される
- final test setによるランダムベースラインとの統計的有意差（Holm-Bonferroni補正後）
- 既存システムとの文献ベース比較（inter-rater κ ≥ 0.6）

### Tier 3: 理想
- **主張**: Tier 2 + 「システムがopen-ended evolutionの兆候を示した」
- 予期しない創発的現象の観察
- 非公式知覚評価で「生きている」判定
- Open-ended evolutionの兆候

---

## 再現性仕様

実験の再現性を担保するため、以下を固定・記録する:

| 項目 | 仕様 |
|------|------|
| ランダムシード | calibration: 0-99, final test: 100-199。全シードをconfig fileで管理 |
| 進化的NNコントローラー | アーキテクチャ・初期化方法・変異率を論文に記載 |
| ローカルLLM（ablation studyのみ） | モデル名・量子化レベル・temperature・top_p・seedを固定し記録 |
| 推論バックエンド | Ollama バージョン、OS、ハードウェアを記録（ablation実施時のみ） |
| シミュレーション | タイムステップ数、環境サイズ、リソース初期配置を固定 |
| コード | Git commit hashと対応する実験結果を紐付け |

**注意**: 論文の主張は進化的NNコントローラー（シード固定で完全再現可能）のみに依存する。LLMは論文Section 5.3相当のablation studyとして「NNコントローラーを基盤モデルに差し替えた場合の比較」を1実験のみ実施する。時間不足の場合はablationを省略し、将来研究として記載する。LLMの非決定性はlimitationとして明記する。

---

*Document generated: 2026-02-08*
*Revised: 2026-02-10 — second-round review + Gemini external review incorporated*
*Based on: unified-review.md + interview results + second-round review + Gemini review*
