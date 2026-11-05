"""Model Registry — maps model_type string to the correct BrownianModel class.

After the consolidation refactor the registry has a single entry: 'multilag'.
Decoder and transition-head variants are selected via config keys
('decoder_type', 'transition_head_type') rather than via registry entries.

Legacy model_type names from the old per-variant file layout still resolve
here for backward compatibility with saved configs; they emit a DeprecationWarning
and rewrite themselves to ('multilag', {extra config keys}).
"""
import importlib
import warnings


MODEL_REGISTRY = {
    'multilag': 'membrane.model_multilag',
    # Membrane-first hypergraph (docs/superpowers/specs/
    # 2026-08-07-membrane-first-hypergraph-design.md). Not a decoder variant of
    # multilag: it has no GAT/GNN geo_embed and no downstream membrane module, so it
    # gets its own entry rather than a config key.
    'hgfirst_7x3d': 'membrane.model_hgfirst',
}

# Architecture defaults per model_type -- only the keys that model_type introduces.
# 'multilag' deliberately has no entry: its defaults have always lived in
# train_brownian_membrane.MULTILAG_DEFAULTS / DECODER_DEFAULTS and moving them would
# change existing runs.
MODEL_ARCH_DEFAULTS = {
    'hgfirst_7x3d': {
        # hypergraph encoder / decoder (design sizing table, 'medium')
        'hg_hidden': 128,
        'hg_layers': 3,
        'hg_dec_hidden': 128,
        'hg_dec_layers': 2,
        # readout -> bank projection, and the bank itself
        'd_enc': 512,
        'lag_lstm_width': 128,
        'n_lag_lstms': 7,
        # 3*(P + E) = 3*(7 + 120); validated in the model, which raises on a mismatch
        'latent_dim': 381,
        'transition_head_type': 'mu_only',
    },
}

# Legacy name -> (model_type, config_overrides_to_apply). Used to translate
# old configs into the generalized multilag form.
LEGACY_ALIASES = {
    'multilag_mlp_7x3d': ('multilag', {'decoder_type': 'mlp', 'transition_head_type': 'mlp_gauss'}),
    'lstm_mlp_7x3d':     ('multilag', {'decoder_type': 'mlp', 'transition_head_type': 'mlp_gauss'}),
    'lstm_xformer_7x3d': ('multilag', {'decoder_type': 'xformer', 'transition_head_type': 'mlp_gauss'}),
    'lstm_kan_7x3d':     ('multilag', {'decoder_type': 'kan', 'transition_head_type': 'mlp_gauss'}),
    'lstm_lstm_7x3d':    ('multilag', {'decoder_type': 'mlp', 'transition_head_type': 'mlp_gauss'}),
    'parallel_xformer_7x3d': ('multilag', {'decoder_type': 'xformer', 'transition_head_type': 'mlp_gauss'}),
}


def _resolve_legacy(model_type: str):
    """Return (canonical_model_type, override_kwargs) for legacy names."""
    if model_type in LEGACY_ALIASES:
        canonical, overrides = LEGACY_ALIASES[model_type]
        warnings.warn(
            f"model_type={model_type!r} is deprecated; use {canonical!r} with "
            f"config overrides {overrides}",
            DeprecationWarning,
            stacklevel=3,
        )
        return canonical, overrides
    return model_type, {}


def get_model_class(model_type: str):
    canonical, _ = _resolve_legacy(model_type)
    if canonical not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_type {model_type!r}. Available: {', '.join(sorted(MODEL_REGISTRY.keys()))}"
        )
    return getattr(importlib.import_module(MODEL_REGISTRY[canonical]), 'BrownianModel')


def resolve_config(config: dict) -> dict:
    """If config['model_type'] is a legacy alias, apply the overrides and normalize."""
    mt = config.get('model_type')
    if mt is None:
        return config
    canonical, overrides = _resolve_legacy(mt)
    if overrides:
        config = {**overrides, **config, 'model_type': canonical}
        # overrides should not clobber explicit user settings; merge lets config win
        for k, v in overrides.items():
            config.setdefault(k, v)
        config['model_type'] = canonical
    return config


def arch_defaults(model_type: str) -> dict:
    """Architecture defaults introduced by a model_type ({} when it has none)."""
    canonical, _ = _resolve_legacy(model_type)
    return dict(MODEL_ARCH_DEFAULTS.get(canonical, {}))


def list_models():
    return sorted(MODEL_REGISTRY.keys())
