# 統合レビュー: Digital Life Project Overview

> **Document Role:** Peer Review & Risk Analysis (2/3)
> This document reviews [`digital-life-project-overview.md`](digital-life-project-overview.md) from a researcher's perspective, combining two independent reviews into a single prioritized analysis.
> - Reviews: [`digital-life-project-overview.md`](digital-life-project-overview.md) — the initial project proposition
> - Informed: [`action-plan.md`](action-plan.md) — action plan that addresses the issues raised here

2つのレビューを統合し、重複を排除、優先度を再設定した。

---

## 致命的 (Critical) — 論文の成否に直結

### C1. 理論的基盤の欠如
7基準を所与として扱っているが、**なぜこの7基準なのか**が論じられていない。

- NASAの生命定義、オートポイエーシス理論（Maturana & Varela）、FBM条件との関係が未整理
- **「functional analogy」と「simplified proxy」の境界定義**がない。本プロジェクトの代謝ネットワークも抽象トークン操作であり、批判対象の既存システムとの質的差異を哲学的・技術的に定義すべき
- Strong ALife vs. Weak ALife における立場表明がない。これは論文のframing全体に影響する

> `digital-life-project-overview.md:5-7`

### C2. 評価指標が再現不可能
定量メトリクスの閾値に根拠がなく、査読で「恣意的」と判断されるリスクが高い。

| 指標 | 問題 |
|------|------|
| `>80% boundary integrity` | 根拠不明。生物学的対応は？ |
| `>3 interdependent pathways` | なぜ3か。最小構成の理論的裏付けは？ |
| `within N timesteps` | N未定義。摂動強度・試行回数・統計検定も未定義 |
| `>50% offspring survive` | 生態学的に妥当な値か？ |
| `>2x complexity at maturity` | 複雑性の測定方法が未定義 |

知覚評価も、被験者数・統計手法・バイアス制御の計画がなく、このままでは査読に耐えない。

> `digital-life-project-overview.md:280-295`

---

## 高 (High) — 早期に対処しないとプロジェクトが停滞

### H1. ALIFE 2026 締切の不整合
文書では「April 6, 2026」としているが、**公式CFPではFull Papers & Summariesの締切は2026-04-01**。5日のずれは、8週間計画の実装・執筆配分に直接影響する。

> `digital-life-project-overview.md:307`

### H2. 新規性主張に対する比較設計の不足
「No system has implemented all seven criteria as structurally equivalent computational processes」は大胆な主張だが：

- **Polyworld** は5〜6基準を統合的に実装しており、差分が具体的に示されていない
- **ALIEN、Chromaria、Geb** など追加の比較対象が欠落
- 比較計画（L297-299）は「where possible」と曖昧で、**反証可能な比較プロトコルが未確定**
- 各システムの内部構造が異なるため「同じメトリクスを走らせる」のは技術的に困難。公平な比較方法論を示す必要がある

> `digital-life-project-overview.md:7, 297-299`

### H3. 統合リスクの過小評価
個別基準のリスクは2前後だが、統合リスク（3.0/5）は楽観的すぎる。

- 代謝 × 発生 × 恒常性の3者結合だけでパラメータ空間が爆発する
- **計算コスト見積もりが完全に欠如**: グラフベース代謝 + NNコントローラー + 集団進化の組み合わせで、100体・1000体規模のシミュレーションが現実的に回るか
- 進化に必要な最小集団サイズの検討がない
- 統合リスクは **4.0以上** が妥当

> `digital-life-project-overview.md:191`

### H4. スケジュールが研究リスクに対して楽観的
14週間で統合まで到達する計画だが：

- 計算資源見積もり（実験回数、1試行コスト、失敗率）がない
- ボトルネック発生時の遅延吸収バッファがない
- Phase 1 の代謝が「3-6週間」と幅があるが、6週間かかった場合 ALIFE 2026 には間に合わない
- Agent-based vs. Swarm の未決定（L27-29）が全フェーズに波及するリスク

> `digital-life-project-overview.md:238-266`

---

## 中 (Medium) — 論文の質に影響

### M1. Swarm vs. Agent-based の未決定
「Floating Parameter」として扱っているが、これはアーキテクチャの根本選択であり、自由度ではなく**制約**として早期に決めるべき。後のフェーズすべてに波及する。

> `digital-life-project-overview.md:214-218`

### M2. オートポイエーシスとの接続が未記述
Maturana & Varela が参考文献にあるのに、本文で「active boundary + metabolism」とオートポイエーシスの関係が論じられていない。査読者は必ずこの接続を問う。

### M3. ALIFE 2026 投稿形態の再検討
8週間でフルペーパーは現実的に厳しい。ポスター/ワークショップペーパー/Extended Abstractなど、別の投稿形態の検討が必要。あるいはarXivプレプリントで概念フレームワークを先行公開する戦略もある。

---

## 低 (Low) — 修正は容易だが放置すべきでない

### L1. 参考文献の書誌情報が不完全
DOI、venue、volume-page が欠ける項目がある。再現性・追跡性に影響。

> `digital-life-project-overview.md:372-389`

### L2. Fischbach & Walsh (2024) の活用が不明
問題選択フレームワークとして引用されているが、本文中でどう適用したかの説明がない。

### L3. 倫理的考慮の欠如
「デジタル生命」を主張する場合、ALifeコミュニティでは倫理的議論への言及が期待される。

---

## 統合推奨アクション

| 優先度 | アクション | 対応する問題 |
|--------|-----------|-------------|
| **1 (今すぐ)** | 締切を4/1に修正し、逆算でスケジュールを再計算 | H1 |
| **2 (今週中)** | Agent-based vs. Swarm を決定 | M1 |
| **3 (Phase 0内)** | 7基準の選択根拠とfunctional analogyの定義を文書化 | C1 |
| **4 (Phase 0内)** | 既存システム（Polyworld, ALIEN, Flow-Lenia等）の7基準充足度比較表を作成 | H2 |
| **5 (Phase 1開始前)** | 計算コスト・集団サイズの予備見積もりを実施 | H3, H4 |
| **6 (Phase 1中)** | 評価指標のN値・摂動条件・統計手法を具体化し、パイロット実験で閾値を校正 | C2 |
| **7 (Phase 1中)** | オートポイエーシスとの理論的接続を記述 | M2 |
| **8 (論文執筆前)** | ALIFE 2026の投稿形態を決定（フル/ポスター/WS） | M3 |
| **9 (論文執筆時)** | 参考文献の書誌情報を完備、倫理的考慮を追加 | L1, L2, L3 |

---

**総括:** プロジェクトの着眼点——既存ALifeの「チェックボックス的」実装を超える——には明確な価値がある。しかし、**理論的基盤（なぜこの7基準か、functional analogyとは何か）** と **評価の再現可能性** が現状最大の弱点であり、これらが未解決のまま実装に入ると、論文執筆時に根本的な書き直しを迫られるリスクがある。Phase 0の文献調査にこれらの理論的作業を含めることを強く推奨する。
