Run order for the owner
========================

Every command below is one run. None has been executed: this machine has no
CUDA and training is forbidden on it.

Phase 0 -- one cheap run, which also calibrates every wall-clock estimate
---------------------------------------------------------------------
  python -m src.train --fname config/scaling/xsmall_30m.yaml 2>&1 | tee run.log

  src/train.py:1591 prints 'Epoch N avg loss: ... time: Ns' every epoch, so
  run.log alone is enough to turn the whole cost table from symbols into
  arithmetic:

    python scripts/results_apparatus.py cost --log-file run.log --train-seqs N

  This run is also Phase 1's smallest rung -- it is listed once, here.

Phase 1 -- the scaling ladder (width/depth only)
------------------------------------------------
  python -m src.train --fname config/scaling/base_140m.yaml
  python -m src.train --fname config/scaling/large_300m.yaml
  python -m src.train --fname config/scaling/small_100m.yaml

Phase 2 -- the local-GPU baselines (MLM and data2vec)
-----------------------------------------------------
  python -m src.train --fname config/wikitext/data2vec_wikitext_large.yaml
  python -m src.train --fname config/wikitext/data2vec_wikitext_small.yaml
  python -m src.train --fname config/wikitext/data2vec_wikitext_train.yaml
  python -m src.train --fname config/wikitext/data2vec_wikitext_xsmall.yaml
  python -m src.train --fname config/wikitext/mlm_wikitext_base.yaml
  python -m src.train --fname config/wikitext/mlm_wikitext_large.yaml
  python -m src.train --fname config/wikitext/mlm_wikitext_small.yaml
  python -m src.train --fname config/wikitext/mlm_wikitext_xsmall.yaml
  python -m src.train --fname config/wikitext/textspanjepa_wikitext_base.yaml
  python -m src.train --fname config/wikitext/textspanjepa_wikitext_small.yaml

Phase 2b -- the Kaggle T4 arms (need Kaggle, not the local GPU)
---------------------------------------------------------------
  python -m src.train --fname config/kaggle/data2vec_kaggle.yaml
  python -m src.train --fname config/kaggle/mlm_kaggle.yaml
  python -m src.train --fname config/kaggle/textspanjepa_kaggle.yaml

Phase 3a -- the full GWP model (the table's reference arm)
----------------------------------------------------------
  python -m src.train --fname config/ablations/all_core.yaml

Phase 3b -- the leave-one-out grid, one mechanism per run
---------------------------------------------------------
  python -m src.train --fname config/ablations/no_cgn.yaml
  python -m src.train --fname config/ablations/no_cmc.yaml
  python -m src.train --fname config/ablations/no_gac.yaml
  python -m src.train --fname config/ablations/no_jawp.yaml
  python -m src.train --fname config/ablations/no_pcr.yaml
  python -m src.train --fname config/ablations/no_puc.yaml
  python -m src.train --fname config/ablations/no_rdc.yaml
  python -m src.train --fname config/ablations/no_spc.yaml
  python -m src.train --fname config/ablations/no_sta.yaml
  python -m src.train --fname config/ablations/no_swip.yaml
  python -m src.train --fname config/ablations/no_wsd.yaml
  python -m src.train --fname config/ablations/no_wsr.yaml

Phase 3c -- arms that switch off more than one mechanism
--------------------------------------------------------
  python -m src.train --fname config/ablations/none.yaml
  python -m src.train --fname config/ablations/sigreg_only.yaml

Phase 4 -- interaction and 10x-weight arms
------------------------------------------
  python -m src.train --fname config/ablations/cgn_spc.yaml
  python -m src.train --fname config/ablations/cmc_on.yaml
  python -m src.train --fname config/ablations/gac_on.yaml
  python -m src.train --fname config/ablations/jawp_swip.yaml
  python -m src.train --fname config/ablations/puc_on.yaml
  python -m src.train --fname config/ablations/puc_rdc.yaml
  python -m src.train --fname config/ablations/rdc_on.yaml
  python -m src.train --fname config/ablations/spc_on.yaml
  python -m src.train --fname config/ablations/sta_gac.yaml
  python -m src.train --fname config/ablations/sta_on.yaml
  python -m src.train --fname config/ablations/sta_wsr.yaml
  python -m src.train --fname config/ablations/swip_on.yaml
  python -m src.train --fname config/ablations/wsd_on.yaml
  python -m src.train --fname config/ablations/wsr_on.yaml

Phase 5 -- hyperparameter arms
------------------------------
  python -m src.train --fname config/ablations/cgn_on.yaml
  python -m src.train --fname config/ablations/jawp_alpha_0.yaml
  python -m src.train --fname config/ablations/jawp_high_alpha.yaml
  python -m src.train --fname config/ablations/jawp_k_fixed.yaml
  python -m src.train --fname config/ablations/jawp_on.yaml
  python -m src.train --fname config/ablations/jawp_pcr.yaml
  python -m src.train --fname config/ablations/jawp_random_init.yaml
  python -m src.train --fname config/ablations/no_decoder_loss.yaml
  python -m src.train --fname config/ablations/no_future_loss.yaml
  python -m src.train --fname config/ablations/pcr_on.yaml
  python -m src.train --fname config/ablations/predictive_rank_on.yaml

Not in any phase, and not runnable as shipped:

  NOT-APPLICABLE  config/scaling/devices/dev1_bs512.yaml  -- holds the global effective batch at 512 and varies only the per-device micro-batch, which needs N GPUs to mean anything
  NOT-APPLICABLE  config/scaling/devices/dev2_bs256.yaml  -- holds the global effective batch at 512 and varies only the per-device micro-batch, which needs N GPUs to mean anything
  NOT-APPLICABLE  config/scaling/devices/dev4_bs128.yaml  -- holds the global effective batch at 512 and varies only the per-device micro-batch, which needs N GPUs to mean anything
  NOT-APPLICABLE  config/scaling/devices/dev8_bs64.yaml  -- holds the global effective batch at 512 and varies only the per-device micro-batch, which needs N GPUs to mean anything
  NOT-APPLICABLE  config/tinystories/textspanjepa_tinystories.yaml  -- src/train.py:740 hard-codes load_wikitext103, so meta.dataset and data.root_path are inert

Then, with the runs on disk:

  python scripts/results_apparatus.py table     # the rows fill themselves in
  python scripts/results_apparatus.py curves    # the figures appear
  python scripts/results_apparatus.py verify    # the guards still hold

The tables are a pure function of (configs, files on disk). There is no results
file to keep in step by hand, so a stale table is not a state this apparatus can
be in.