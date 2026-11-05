"""Reject config keys nothing reads.

``--config-json`` merges with ``config.update()``, so before this an unrecognised key
was accepted and ignored. Every sweep arm that turned on a misspelled flag silently
trained the baseline instead.
"""
import difflib
import warnings

from nbcore.config.known_keys import ALIASES, KNOWN_KEYS


class UnknownConfigKeys(KeyError):
    """Raised when a config carries keys the code never reads."""

    def __str__(self):                      # KeyError repr()s its arg; we want the text
        return self.args[0]


def validate_config(config, source='<config>', strict=True):
    """Apply ALIASES in place, then reject unknown keys.

    Keys starting with '_' are comments (``_note``, ``_source``, ``_comment``) and are
    always allowed. ``strict=False`` warns instead of raising -- for deliberate
    experiments that add a key before the code that reads it.
    """
    for old, new in ALIASES.items():
        if old in config:
            warnings.warn(f"{source}: config key '{old}' is deprecated, use '{new}'",
                          DeprecationWarning, stacklevel=2)
            config.setdefault(new, config.pop(old))

    unknown = sorted(k for k in config
                     if not k.startswith('_') and k not in KNOWN_KEYS)
    if not unknown:
        return config

    lines = []
    for k in unknown:
        near = difflib.get_close_matches(k, KNOWN_KEYS, n=3, cutoff=0.7)
        lines.append(f"  {k!r}" + (f"   did you mean: {', '.join(near)}?" if near else ""))
    msg = (f"{source}: {len(unknown)} config key(s) that nothing reads:\n"
           + "\n".join(lines)
           + "\nA key no code reads is a silently ignored experiment arm. If it is "
             "intentional, add it to nbcore/config/known_keys.KNOWN_KEYS.")
    if strict:
        raise UnknownConfigKeys(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return config
