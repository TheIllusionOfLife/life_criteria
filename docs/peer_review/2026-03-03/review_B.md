# ALife Conference Full Paper Review

## Paper
**Title:** *Searching for an Eighth Criterion of Life: A Falsifiable Framework and Two Null Results*

## Overall Recommendation
**Decision:** Weak Reject (borderline)  
**Reviewer Confidence:** 4/5

## Summary
本稿は、「生命の7基準」に対して第8基準候補をどのように反証可能な形で評価するか、という方法論的問いを扱っています。著者はハイブリッドな swarm-organism シミュレータ上で、Candidate A（生涯内記憶）と Candidate B（kin-sensing）を、baseline / enabled / ablated / sham の4条件、2つの攪乱レジーム（famine, boom-bust）で比較し、両候補とも有意な改善を示さないという null を報告します。さらに、null を「機構が動いていない」ではなく「動いているが効いていない」と診断する分析（EMA収束確認、kin signal の退化診断）を加えている点が主な貢献です。

## Score Breakdown (1–5)
- **Novelty / Originality:** 4
- **Technical Soundness:** 3
- **Empirical Rigor:** 3
- **Clarity / Organization:** 4
- **Significance for ALife community:** 3

## Strengths
- 反証可能性を前面に出した評価プロトコル（ablation + sham + SESOI）は、ALifeにおける「何をもって候補基準を支持するか」を明示しており有益です。
- ネガティブ結果を正面から報告し、かつ機構診断を付した点は、出版バイアス低減の観点で価値があります。
- Candidate A/B いずれも「機構実装」と「生存利得」を切り分けて議論しており、解釈が比較的明瞭です。
- 限界と今後の成功条件（effect size, ablation, sham inequality, mechanism verification）を明示していて、今後の研究計画に接続しやすい構成です。

## Major Concerns
1. **「bounded null」主張の強さが統計的に過大**  
   本文自身が認める通り、各比較の95%CIは広く、正式な同等性検定（TOST）も未実施です。この状況で「SESOI未満に拘束できた」と強く言い切るのは慎重さを欠きます。現状の結論は「中程度以上の効果を排除しきれないが、観測点推定は小さい」が適切です。

2. **提案フレームワークの3正当性テストのうち実証は1つのみ**  
   Orthogonality / Non-reducibility を提案しながら未評価で、実験部は causal necessity 中心です。フレームワーク論文としては、少なくとも縮退版でも2/3を実装した実証が欲しいです。

3. **Candidate B の実験がレジーム設計由来の退化に支配される**  
   著者の診断通り population cap により single-agent 収束が生じ、kin signal が観測不能になります。この場合、候補機構の有効性検証というより、設定依存の可観測性失敗を確認した実験になっており、結論の一般性が限定的です。

4. **可視化・定量報告が不足**  
   結果テーブルは1枚のみで、時系列（生存数、AUC分布、kin_fraction推移、EMA dynamics）図がありません。full paper としては、診断主張を裏づける図表が不足しています。

## Minor Concerns
- Mann–Whitney 検定と Cohen’s d の併用自体は可能ですが、効果量定義の整合（例えば rank-biserial 併記）を明示すると読者に親切です。
- Holm 補正対象となる比較セットの定義をより明確化すると再現性が上がります。
- baseline の絶対値（AUC平均・分散）を本文または付録で示すと、ΔAUCの解釈が容易になります。

## Suggestions for Improvement (for next revision)
- TOST あるいはベイズ的 ROPE を導入し、「bounded null」主張を統計的に厳密化する。
- Orthogonality / Non-reducibility を少なくとも1候補で実施し、提案フレームワークの中核を実証する。
- Candidate B では population cap を緩めた条件、または multi-agent persistence を保証する生態設計で追試する。
- Candidate A では「学習可能構造のある攪乱」（周期・予測可能な再来）を導入し、memory utility の成立条件を検証する。
- 図表を追加（生存曲線、効果量CIフォレスト、kin_fraction/organism-size時系列、EMA収束図）して診断の説得力を高める。

## Final Justification
本稿は、ALifeで見落とされがちな「厳密な null の報告と機構診断」を正面から扱う点で意義があり、着想と文章構成も良好です。一方で、現時点では統計的主張の強さと実証範囲（提案した3テストの未実装）にギャップがあり、full paper 採択にはもう一段の実証強化が必要と判断します。よって現評価は **Weak Reject（境界的）** です。

## 10点満点を狙うための改善案（複数）

### 改善案1: 統計主張を「反証不能」から「同等性証明」へ格上げ
- TOST（SESOI = ±0.5）を各主要比較で実施し、95% CI と併記して bounded null を形式的に主張する。
- 可能なら Bayesian ROPE も併記し、頻度論・ベイズ両面で結論を頑健化する。
- Supplementary に power curve（n=30, 50, 80, 120）を掲載し、必要サンプル数を明示する。

### 改善案2: 提案フレームワークの3本柱を全て実証
- Causal necessity だけでなく、Orthogonality（既存7基準特徴量を統制した予測モデル）を追加する。
- Non-reducibility（7基準側の再チューニング/拡張ベースライン）を実施し、候補機構固有効果を検証する。
- 最低1候補で「3テストすべて通す/落とす」判定例を示し、フレームワーク論文として完成度を上げる。

### 改善案3: Candidate B の可観測性失敗を設計実験で解消
- `max_alive_organisms` を段階的に緩和したアブレーション（例: 100/200/400）を追加する。
- multi-agent persistence を維持する生態制約（最小群サイズ維持コスト等）を導入し、kin signal の情報量を再評価する。
- kin_fraction の時系列と organism-size 分布を図示し、「信号が存在しない」vs「信号を使えない」を厳密に分離する。

### 改善案4: Candidate A を「学習可能レジーム」で再評価
- 単発ショック中心の famine に加え、予測可能周期・部分回復・遅延報酬など学習可能攪乱を導入する。
- memory 依存行動の介在指標（政策変化量、行動予測誤差、memory-controller 相互情報量）を追加する。
- 「EMA は動作するが有用でない」の理由を、タスク不一致か表現力不足かで因果分解する。

### 改善案5: 図表と再現性資料を full paper 水準へ強化
- 本文に最低4図（生存曲線、効果量CIフォレスト、EMA収束、kin信号退化）を追加する。
- 結果表には平均だけでなく分散、CI、効果量定義、補正対象 family を明記する。
- コード・設定・seed・実験スクリプトをアーカイブ化し、1コマンド再現手順を付録に記載する。

### 改善案6: 「方法論論文」としてのメッセージを明確化
- タイトル/導入で「候補の当落」より「評価プロトコルの一般性」を主語に据える。
- Discussion で、他ALife基盤（Lenia系、voxel系等）へ移植可能な最小要件を定義する。
- 最後に「採択可能な成功判定テンプレート（チェックリスト）」を1ページで提示する。

### 改善案7: 採択率を上げる実行順（短期ロードマップ）
- **Phase 1（必須）**: TOST + 図表追加 + 補正定義明確化。
- **Phase 2（重要）**: Orthogonality / Non-reducibility の縮退実装。
- **Phase 3（加点）**: Cap緩和実験と学習可能レジーム実験を各1本ずつ追加。
- 上記を満たせば、Novelty/Technical/Empirical/Significance の各スコアを1段階ずつ押し上げ、総合10点相当を狙える構成になる。