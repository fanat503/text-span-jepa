
Research backlog, generated from a literature scan. Directive 1: Cross-Paper SOTA
distillation. Written as a file rather than only in a reply, because a backlog
that exists only in a conversation is lost at the first compaction.

The primary paper is real and correctly identified by the owner:

  RoboJEPA: Scaling Robotic Latent World Models
  arXiv:2610.10515, 7 Oct 2026, FAIR at Meta (Zholus, Beltran-Velez, Yuan,
  Chandar, Nagarajan, Severo, Sinha, Drozdzal, Romero Soriano, Bohg, Ballas,
  Assran). Code: facebookresearch/robo_jepa.

One correction to the owner's framing, recorded because it matters for what we
claim: the paper does NOT "break" or violate scaling laws. It ESTABLISHES them
for JEPA - the first compute-optimal scaling law with verified extrapolation for
a latent world model. L(C) = E + A*C^(alpha - gamma*ln C), a second-order power
law. Claiming we "broke" scaling laws would be a reviewer-detectable error, so
nothing downstream may say that.

WHAT TRANSFERS TO A TEXT-SPAN JEPA
Because RoboJEPA is a video/multiview/action-conditioned world model, a large
part of it does not apply. Porting it wholesale would be cargo cult, and a
reviewer who reads the source would catch it. Stated explicitly so no one
"ports everything" later.

TRANSFERS, ranked by expected value:
1. QK-normalization. The paper states it is "of particular importance for
   training stability when scaling the dataset size and the model size". This is
   a two-line change per attention block and is the single best value-per-risk
   item found.
2. The rollout (autoregressive) half of the objective. RoboJEPA trains
   L = 0.5*L_TF + 0.5*L_AR, where L_AR asks the model to predict the same
   targets from a SHORTER prefix of its own rollouts, using a KV-cache. For a
   span-masked text JEPA this is the natural anti-exposure-bias term: our
   predictor is currently only ever teacher-forced, so it has never seen its own
   errors and multi-step behaviour is untested by training.
3. LayerNorm on the output projection back to encoder space. The paper's stated
   reason is that both input and output features being normalised "strengthens
   the learning signal and lowers the gap between the training and the
   autoregressive generation". This is exactly the gap our repo has never closed.
4. Sliding-window local attention, window 8. Their reason is the one that
   matters: "no query will ever attend to a history longer than the model was
   trained on, including during long autoregressive prediction", which is what
   makes length extrapolation work. Our repo has no length-extrapolation story
   at all.
5. Frozen encoder, scaled predictor. They fix the encoder at V-JEPA 2.1-G and
   scale ONLY the predictor 22M -> 8B. We already freeze a target encoder; this
   says the right way to scale is the predictor, which is a directly citable
   design rationale.
6. Scaling-law infrastructure: fit L(C) = E + A*C^(alpha - gamma*ln C) over the
   scaling ladder we already have, and treat the latent rollout error as the
   PROXY for downstream. That last point is the one a text paper most needs,
   because it is what lets a probe number be predicted before the run is paid
   for.
7. The LR recipe: AdamW wd=0.1, global grad-norm clip 1.0, linear warmup, flat
   peak, linear decay to zero over a short cooldown, with the flat stage run once
   and cooldown applied from intermediate checkpoints. Cheap, and it makes the
   scaling ladder comparable across arms.
8. RMSNorm + pre-norm + GELU as the block recipe.

DOES NOT TRANSFER, and why - recorded so nobody does it anyway:
- Multi-axis 4D RoPE over (t, i, j, v) and the per-layer learnable view bias:
  our modality has no views, only sequence positions.
- Per-embodiment input projections and robot-id conditioning: no embodiments.
- Action conditioning and the state-delta pipeline: text pretraining has no
  actions.
- Timestep-causal masking over interleaved multimodal tokens: our mask is span
  based and already exists.
- CEM planning toward a single goal image, rolling KV-cache at inference, AOT
  Inductor compilation, FP8: deployment machinery, not pretraining.
- The diffusion decoder: reconstruction, which this repo deliberately does not do.

SECONDARY PAPERS WORTH DISTILLING (all found in the same scan):
- lambda-JEPA / SACReg (arXiv:2609.35288): a spectral anti-collapse regulariser
  motivated by lambda-balance across layers. This is the closest published
  neighbour to our sigreg / sta / wsr family and is the most defensible place to
  look for a real improvement.
- Drive vs. Decay (arXiv:2610.02344, NeurIPS 2026): a stability theory giving a
  per-mode ratio mu_i = gamma_i / sigma_i that unifies predictor scaling, masking
  ratio and EMA, plus ResidualPred, a predictor whose attention is biased toward
  the identity at initialisation. Directly applicable: we have EMA, a masking
  ratio and a predictor, and no theory for how they interact.
- DSReg (arXiv:2610.09457): dependency-sparsity regularisation that makes a JEPA
  identifiable up to signed permutation without reconstruction.
- H-JEPA (arXiv:2609.33497): a Bures-Wasserstein isotropic-geometry prior on a
  fixed orthonormal slice, with port-inverse consistency as a parameter-free
  reweighting of prediction error.
- Frozen in a Frame (arXiv:2610.04585): a measured structural blind spot - a
  single-frame JEPA embedding carries no velocity, for ANY encoder including the
  official LeWM weights. Fix is a pose code plus an explicit finite-difference
  motion code. The lesson that transfers: a diagnostic protocol that shows what
  a representation provably cannot encode.

HARD CONSTRAINT ON ALL OF THIS
None of it licenses claiming our model "naturally outperforms standard
baselines". These are techniques to evaluate behind ablations, not results to
assert. The repo already has a results apparatus whose central property is that
it cannot emit an unmeasured number, and any new feature must land as an ablation
arm, not as a claim in the README.
