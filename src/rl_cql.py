# src/rl_cql.py
import numpy as np
import inspect

# ---- MDPDataset import (name varies across versions) ----
try:
    from d3rlpy.datasets import MDPDataset
except Exception:
    from d3rlpy.dataset import MDPDataset  # fallback


# =========================================================
# 1) Build offline MDP dataset (contextual bandit, 1-step)
# =========================================================
def build_mdp_dataset(X, loan_amnt, int_rate, y_default):
    """
    state  = features X[i]
    action = 1 (approve) for logged data
    reward = +loan_amnt*int_rate if y==0 else -loan_amnt
    terminal = True
    Version-agnostic: only passes kwargs supported by your MDPDataset.
    """
    X         = np.asarray(X, dtype=np.float32)
    loan_amnt = np.asarray(loan_amnt, dtype=np.float32)
    int_rate  = np.asarray(int_rate, dtype=np.float32)
    y_default = np.asarray(y_default, dtype=np.int64)

    n         = X.shape[0]
    actions   = np.ones(n, dtype=np.int64)  # logged policy approved
    rewards   = np.where(y_default == 0, loan_amnt * int_rate, -loan_amnt).astype(np.float32)
    terminals = np.ones(n, dtype=bool)

    # Some versions accept next_observations; some don't.
    next_obs  = np.zeros_like(X, dtype=np.float32)

    sig    = inspect.signature(MDPDataset.__init__)
    params = sig.parameters

    kwargs = {}
    if "observations" in params:        kwargs["observations"] = X
    if "actions" in params:             kwargs["actions"] = actions
    if "rewards" in params:             kwargs["rewards"] = rewards
    if "terminals" in params:           kwargs["terminals"] = terminals
    if "episode_terminals" in params:   kwargs["episode_terminals"] = terminals
    if "next_observations" in params:   kwargs["next_observations"] = next_obs

    return MDPDataset(**kwargs)


# ==========================================
# 2) Construct a DiscreteCQL algo (CPU-only)
# ==========================================
def _make_cql():
    """
    Create DiscreteCQL across d3rlpy 2.x variants.

    Tries:
      - DiscreteCQL(DiscreteCQLConfig(...))
      - DiscreteCQL(DiscreteCQLConfig(...), "cpu", False)
      - DiscreteCQL(learning_rate=..., batch_size=..., target_update_interval=...)
      - DiscreteCQL("cpu", learning_rate=..., ...)
    """
    from d3rlpy.algos import DiscreteCQL

    # Try with config class
    try:
        from d3rlpy.algos import DiscreteCQLConfig
        cfg = DiscreteCQLConfig(
            learning_rate=3e-4,
            batch_size=1024,
            target_update_interval=800,
        )
        try:
            return DiscreteCQL(cfg)                      # common case
        except TypeError:
            return DiscreteCQL(cfg, "cpu", False)        # (config, device, enable_ddp)
    except Exception:
        pass

    # Try passing hyperparams directly
    try:
        return DiscreteCQL(
            learning_rate=3e-4,
            batch_size=1024,
            target_update_interval=800,
        )
    except TypeError:
        # Positional device first (very old variant)
        return DiscreteCQL(
            "cpu",
            learning_rate=3e-4,
            batch_size=1024,
            target_update_interval=800,
        )


# ==========================================
# 3) Train CQL (fit) with flexible signature
# ==========================================
def train_cql(mdp_train, n_steps=200_000):
    """
    Fit CQL on CPU regardless of d3rlpy minor-version differences.
    - No 'verbose' arg (your version doesn't accept it).
    - Handles dataset positional vs keyword forms.
    """
    algo = _make_cql()

    # Some versions expose build_with_dataset; safe to try
    try:
        algo.build_with_dataset(mdp_train)
    except Exception:
        pass

    # Detect fit signature
    fit_sig  = inspect.signature(algo.fit)
    fit_kwds = {k for k in fit_sig.parameters.keys()}

    # Build kwargs that exist
    kwargs = {}
    if "dataset" in fit_kwds:
        kwargs["dataset"] = mdp_train
    if "n_steps" in fit_kwds:
        kwargs["n_steps"] = n_steps
    if "n_steps_per_epoch" in fit_kwds:
        kwargs["n_steps_per_epoch"] = 10_000
    if "scorers" in fit_kwds:
        # no custom scorers; let it run default
        pass
    # DO NOT pass 'verbose' — your version errors on it

    # Call fit; fall back to positional dataset if needed
    try:
        return algo.fit(**kwargs) or algo
    except TypeError:
        # Positional dataset first, then remaining kwargs
        if "dataset" in kwargs:
            ds = kwargs.pop("dataset")
        else:
            ds = mdp_train
        return algo.fit(ds, **kwargs) or algo


# ==========================================
# 4) Policy action helper
# ==========================================
def policy_actions(algo, X):
    X = np.asarray(X, dtype=np.float32)
    return algo.predict(X)


# ==========================================
# 5) FQE helper (optional, version-agnostic)
# ==========================================
def fqe_estimate(algo, mdp_train, mdp_eval, n_steps=100_000):
    """
    Version-agnostic FQE (Estimated Policy Value).
    Tries DiscreteFQE first, then generic FQE.
    """
    # Try import variants
    fqe_cls = None
    cfg_cls = None
    try:
        from d3rlpy.ope import DiscreteFQE as FQE, FQEConfig
        fqe_cls = FQE
        cfg_cls = FQEConfig
    except Exception:
        try:
            from d3rlpy.ope import FQE, FQEConfig
            fqe_cls = FQE
            cfg_cls = FQEConfig
        except Exception as e:
            raise ImportError("FQE not available in this d3rlpy build") from e

    # Build config as simply as possible
    try:
        cfg = cfg_cls()
    except Exception:
        cfg = None

    # Instantiate
    try:
        fqe = fqe_cls(algo, cfg) if cfg is not None else fqe_cls(algo)
    except TypeError:
        # Some versions require only policy
        fqe = fqe_cls(algo)

    # Fit (signature-flexible)
    fit_sig  = inspect.signature(fqe.fit)
    fit_kwds = {k for k in fit_sig.parameters.keys()}
    kwargs   = {}
    if "dataset" in fit_kwds:
        kwargs["dataset"] = mdp_train
    if "n_steps" in fit_kwds:
        kwargs["n_steps"] = n_steps
    if "n_steps_per_epoch" in fit_kwds:
        kwargs["n_steps_per_epoch"] = 10_000

    try:
        fqe.fit(**kwargs)
    except TypeError:
        # positional dataset fallback
        ds = kwargs.pop("dataset", mdp_train)
        fqe.fit(ds, **kwargs)

    # Predict value on eval dataset (one-step bandit => average reward)
    try:
        return float(fqe.predict_value(mdp_eval))
    except TypeError:
        # Some variants expose estimate_value
        return float(fqe.estimate_value(mdp_eval))
