"""The set of config keys the code actually reads.

The config is a flat dict passed by reference from the CLI through to the model, and
``--config-json`` merges with ``config.update()``. Nothing checked the keys, so a typo
in a sweep JSON was a silent no-op: the run trained happily on the default and the arm
you thought you were testing never existed. That is not hypothetical -- the 30-cell
seed sweep set ``lr_schedule``, which nothing reads, so all three of its schedule arms
would have been the same hardcoded ReduceLROnPlateau.

KNOWN_KEYS is generated from an AST walk of every ``config.get('k')`` /
``config['k']`` / ``self.hparams.get('k')`` site in the packages, unioned with the
declared default tables. ``tests/shared/nbcore/test_config_keys.py`` re-derives it and fails if a new
call site appears that is not listed here, so the list cannot silently rot.

Renaming a key: add it to ALIASES, add the new name below, and leave the old name out.
Checkpoint ``hyper_parameters`` is a plain dict frozen on disk, so key names are a
persistent contract in a way module paths are not.
"""

KNOWN_KEYS = frozenset({
    'accumulate_grad_batches', 'alpha_lags', 'arm_ev_margin', 'arm_radius',
    'arm_sigma_theta', 'arms_enabled', 'augmented_dataset_paths',
    'augmented_frame_limits', 'augmented_source_frames', 'batch_size', 'blade_ic', 'blade_ic_seed', 'box_size',
    'checkpoint_path', 'colored_noise', 'colored_noise_rank', 'd_enc', 'dataset_path',
    'dataset_paths', 'dec_drift_ln', 'dec_per_species', 'dec_hidden_dim', 'dec_hyper_tokens', 'dec_latent_skip', 'dec_latent_values', 'dec_mlp_layers', 'dec_raw_gain_init', 'dec_use_context',
    'dec_xformer_dim_feedforward', 'dec_xformer_heads', 'dec_xformer_layers',
    'decoder_ar1', 'decoder_noise_type', 'decoder_type', 'disable_torch_compile',
    'dropout', 'dt', 'dynamics_lr_mult', 'ev_force_anneal_epochs', 'ev_force_cap',
    'ev_force_cutoff', 'ev_force_cycle_epochs', 'ev_force_cycle_mult',
    'ev_force_min_frac', 'ev_force_power', 'ev_force_schedule', 'ev_force_strength',
    'ev_loss_form', 'ev_loss_preforce', 'ev_w_cycle_epochs', 'ev_w_cycle_mult',
    'ev_w_min_frac', 'ev_w_overlap_epochs', 'ev_w_ramp_epochs', 'ev_w_schedule',
    'excluded_volume_core', 'excluded_volume_margin', 'excluded_volume_power',
    'freeze_decay', 'freeze_innov_scale', 'geo_embed_type', 'geo_ic_axes', 'geo_ic_once', 'geo_per_species', 'geom_arm', 'geom_blade_enabled', 'gmm_components',
    'gradient_checkpointing', 'gradient_clip_val', 'hg_dec_hidden', 'hg_dec_layers',
    'hg_hidden', 'hg_layers', 'hidden_dim', 'init_decay', 'init_innov_scale',
    'init_noise_corr', 'input_noise_sigma', 'kan_grid_size', 'kan_spline_order',
    'kurtosis_lags', 'lag_lstm_lags', 'lag_lstm_width', 'loss_lag_norm', 'loss_lags', 'lstm_fp32', 'latent_corr_kernel', 'latent_corr_lags',
    'latent_corr_scale', 'latent_corr_target', 'latent_dim', 'latent_mode',
    'latent_type', 'learnable_dt', 'learnable_dynamics', 'learnable_theta',
    'ckpt_keep', 'default_root_dir', 'learning_rate', 'lr_min_factor', 'lr_schedule', 'lr_warmup_epochs',
    'max_epochs', 'mem_lags', 'membrane_augment',
    'membrane_consistency_slope', 'membrane_augment_permute', 'membrane_augment_translate',
    'membrane_edge_latent',
    'membrane_data_path', 'membrane_data_paths',
    'membrane_delta_mode', 'membrane_delta_scale_hyper', 'membrane_delta_scale_node',
    'membrane_depth_mode', 'membrane_loss_units', 'membrane_ou_a_hyper',
    'membrane_ou_a_node', 'membrane_ou_noise', 'membrane_ou_sigma_hyper',
    'membrane_ou_sigma_node',
    'membrane_disc_dim', 'membrane_hidden', 'membrane_hypergraph', 'membrane_layers',
    'membrane_mesh_path', 'membrane_mode', 'membrane_pool_balance', 'membrane_radius',
    'membrane_spacing', 'membrane_standardize_inputs', 'model_type', 'multi_system', 'n_lag_lstms', 'n_particles',
    'n_rollout_samples', 'n_samples', 'noise_rank', 'nsf_bins',
    'nsf_conditioner_hidden', 'nsf_tail_bound', 'nsf_transforms', 'num_heads',
    'num_workers', 'optimizer_type', 'optuna_score_mode', 'ou_dt_init', 'ou_theta',
    'pairwise_tail_power', 'pairwise_tail_weight', 'per_dim_dynamics', 'per_protein_banks',
    'per_species_dynamics',
    'per_species_banks', 'project_name', 'quantile_mode', 'randn_innovations',
    'resume_from_checkpoint', 'samples_per_epoch', 'scheduler_patience',
    'scheduler_threshold', 'seed', 'seq_len_end', 'seq_len_start', 'seq_len_warmup',
    'sigma_layout', 'sophia_rho', 'species_groups', 'splat_grid', 'splat_hidden',
    'splat_out_dim', 'splat_sigma', 'splat_z_slabs', 'squash_asymmetric',
    'squash_core_free', 'squash_cutoff_cores', 'squash_cutoff_frac',
    'squash_edge_hidden', 'squash_enabled', 'squash_hidden', 'squash_init_bias', 'squash_lr_mult',
    'squash_pool', 'squash_use_latent', 'stab_check_interval', 'stab_floor_smooth', 'stab_lag_grid_max', 'stab_lag_grid_step', 'stab_objective_metrics', 'stab_short_lag_weight',
    'stab_rollout_steps', 'stability_floor_json', 'stability_gt_paths', 'stats_path',
    'system_configs', 'transition_head_type', 'uniform_windows',
    'ema_decay', 'ema_decays', 'ema_start_epoch', 'ema_every_n_epochs',
    'early_stop_patience', 'early_stop_min_delta', 'early_stop_start_epoch',
    'fit_objective', 'fit_objective_rollouts', 'fit_rc', 'fit_core', 'ckpt_every_epoch',
    'displacement_signed', 'mem_step_signed', 'geo_pair_dir_shells', 'z_quantile_periodic',
    'z_from_depth', 'predict_z',
    'stab_per_species_objective',
    'w_alpha', 'w_angle',
    'w_arm_excluded_volume', 'w_arm_pairwise', 'w_cohesion', 'w_displacement', 'w_dlogpair_kl', 'w_logpair_kl',
    'w_dlogpair_q', 'w_logpair_q',
    'w_excluded_volume', 'w_heading', 'w_kurtosis', 'w_latent_corr', 'w_mem_autocorr',
    'w_mem_cohesion', 'w_mem_consistency', 'w_mem_coupling', 'w_mem_init', 'w_mem_pairwise',
    'membrane_depth_bias_init',
    'w_mem_step',
    'w_mem_value', 'w_membrane', 'w_odometer', 'w_pairwise', 'w_pairwise_tail', 'w_per_dim_reg',
    'w_rot_step', 'w_squash', 'w_theta_step', 'w_tilt', 'w_vacf', 'w_vacf_mse', 'w_vacf_spatial',
    'w_z_displacement', 'w_z_quantile', 'wandb_group', 'wandb_log_model',
    'wandb_run_name', 'weight_decay', 'z_displacement_signed',
})

# old key -> new key. Applied before validation, with a DeprecationWarning.
ALIASES: dict[str, str] = {}
