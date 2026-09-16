# Novelty positioning and claim discipline

This repository deliberately separates **implemented mechanisms** from **claims of novelty**.

## What is established prior art

The following ideas are established and must not be presented as first-of-kind contributions:

- Gradient surgery for conflicting multi-task gradients (PCGrad; NeurIPS 2020).
- Conflict-averse multi-task gradient optimization (CAGrad; NeurIPS 2021).
- Dynamic auxiliary-task weighting using primary-task gradient information (e.g., Adaptive Auxiliary Task Weighting, NeurIPS 2019).
- Module-aware auxiliary-loss weighting/optimization (MAOAL; NeurIPS 2022).
- Validation-driven mitigation of negative transfer in auxiliary-task learning (ForkMerge; NeurIPS 2023).
- Valence-arousal-guided soft contrastive weighting in emotion representation learning (e.g., EMOD; AAAI 2026).
- Multimodal music-emotion systems using cross-attention and/or auxiliary contrastive alignment are already present in the 2026 literature.

## Candidate contribution tested here

The candidate methodological contribution is intentionally narrower:

**Primary-Emotion-Preserving Adaptive Cross-Modal Alignment (CSPA)** dynamically caps the strength of the audio-lyrics alignment gradient using the primary emotion gradient, a relative gradient-norm limit, and a local smoothness-based finite-step constraint. Affect-structured negative weighting is treated as a complementary mechanism, not the primary novelty claim.

The contribution is meaningful only if experiments show that it improves or preserves primary emotion prediction while controlling harmful alignment pressure compared with fixed scalarization, PCGrad, CAGrad, and no-alignment baselines.

## Conditional descent statement

Let the primary emotion loss be `L_e(theta)` with gradient `g_e`, alignment loss gradient `g_a`, combined raw direction

`d = g_e + lambda g_a`,

and an SGD step `theta+ = theta - eta d`.

If `L_e` is beta-smooth in the relevant local region and beta is a valid upper bound, then the descent lemma implies

`L_e(theta+) <= L_e(theta) - eta <g_e,d> + beta eta^2 ||d||^2 / 2`.

CSPA chooses lambda so that the resulting quadratic sufficient condition implies

`L_e(theta+) <= L_e(theta) - kappa eta ||g_e||^2`.

This is a **conditional result for the raw SGD direction**. In the practical default configuration, training uses AdamW and beta is estimated from local secants. Therefore the practical AdamW run is described as an empirical safeguard/diagnostic, not as a certified non-regression theorem.

The implementation logs the estimated beta, gradient cosine, gradient norms, curvature and norm caps, selected lambda, feasibility flag, and first-order directional margin so this distinction can be audited empirically.

## MERGE affect geometry

The MERGE convention is Q1 = positive valence / positive arousal, Q2 = negative valence / positive arousal, Q3 = negative valence / negative arousal, and Q4 = positive valence / negative arousal. The affect-aware coordinate mapping used in the code follows that convention.

## Claim rule

Do not use phrases such as "first", "novel", "provably safe under AdamW", or "certified" in a manuscript until a dedicated literature review and the relevant assumptions/experiments support them. The strongest defensible framing before that audit is "candidate primary-emotion-preserving adaptive alignment mechanism" with explicit comparison to adjacent work.
