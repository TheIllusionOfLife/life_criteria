# Citation Audit (Primary-Source Web Check)

Date: 2026-03-04
Scope: `paper/main.tex` (current submission manuscript, non-archive)
Method: Claim-by-claim check against primary papers, publisher pages, or official proceedings.

## Summary
- Supported as cited: 6
- Partially supported / wording adjusted: 2
- Unsupported: 0

## Claim-Level Audit

1. Claim
- "Despite this rich landscape, few systems implement all seven criteria as interdependent, individually ablatable processes..." (`paper/main.tex`, Introduction)

Evidence checked
- Representative ALife systems cited in manuscript (Tierra, Avida, Lenia, Flow-Lenia, etc.) do not uniformly report seven-criterion, explicit ablation-style validation in a single framework.
- Tierra: https://doi.org/10.1016/0167-2789(91)90084-4
- Lenia: https://arxiv.org/abs/1812.05433
- Flow-Lenia: https://arxiv.org/abs/2306.09316

Verdict
- Partially supported (broad synthesis claim; not a formal systematic review)

Action taken
- Softened wording to "based on representative systems" in `paper/main.tex`.

2. Claim
- Paired Wilcoxon signed-rank as primary paired nonparametric test (`paper/main.tex`, Statistical pipeline)

Evidence checked
- Wilcoxon, 1945 original test paper: https://doi.org/10.2307/3001968

Verdict
- Supported

Action taken
- No citation change required.

3. Claim
- Holm-Bonferroni correction for family-wise error (`paper/main.tex`, Statistical pipeline)

Evidence checked
- Holm, 1979 original method: https://www.jstor.org/stable/4615733

Verdict
- Supported

Action taken
- No citation change required.

4. Claim
- TOST equivalence testing with SESOI framing (`paper/main.tex`, Equivalence testing)

Evidence checked
- Lakens, 2017: https://doi.org/10.1177/1948550617697177
- Lakens et al., SESOI rationale: https://doi.org/10.1177/2515245918770963

Verdict
- Supported

Action taken
- No citation change required.

5. Claim
- Inclusive fitness motivation for kin-sensing candidate (`paper/main.tex`, Candidate B implementation)

Evidence checked
- Hamilton, 1964 foundational paper: https://doi.org/10.1016/S0022-5193(64)80038-4

Verdict
- Supported

Action taken
- No citation change required.

6. Claim
- Topology evolution as plausible capacity extension (`paper/main.tex`, Limitations)

Evidence checked
- NEAT (Stanley & Miikkulainen, 2002): https://doi.org/10.1162/106365602320169811

Verdict
- Supported

Action taken
- No citation change required.

7. Claim
- Null-result reporting importance (`paper/main.tex`, Introduction)

Evidence checked
- Rosenthal file-drawer framing: https://doi.org/10.1037/0033-2909.86.3.638
- Fanelli on negative results/publication bias: https://doi.org/10.1371/journal.pone.0030576

Verdict
- Supported

Action taken
- No citation change required.

8. Claim
- Flow-Lenia citation year/version alignment (`paper/main.tex`, Introduction refs)

Evidence checked
- Public primary preprint record currently indexed as 2023/2024+ versions depending on archive metadata; canonical URL: https://arxiv.org/abs/2306.09316

Verdict
- Partially supported (possible bib metadata/version drift)

Action taken
- Keep claim scope unchanged; verify exact bibliography year/key consistency in `paper/references.bib` during camera-ready cleanup.
