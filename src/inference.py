"""inference.py
Viterbi: fallback en log-space y adaptador robusto para hmmlearn.
"""
import numpy as np

# IMPORT DIRECTO: Se asume que el entorno tiene hmmlearn instalado si se usa infer_with_hmmlearn.
try:
    from hmmlearn import hmm as hmmlearn_hmm
    import hmmlearn
except Exception:
    hmmlearn_hmm = None
    hmmlearn = None

def map_observations_to_indices(obs_seq, observations):
    mapping = {o:i for i,o in enumerate(observations)}
    return np.array([mapping[o] for o in obs_seq], dtype=int)

def viterbi_fallback(pi, A, B, observations_list, obs_seq):
    """
    Implementación en log-space del algoritmo Viterbi.
    """
    if len(obs_seq) == 0:
        return []
    obs_idx = map_observations_to_indices(obs_seq, observations_list)
    T = len(obs_idx)
    N = A.shape[0]
    logA = np.log(A + 1e-12)
    logB = np.log(B + 1e-12)
    logpi = np.log(pi + 1e-12)

    dp = np.full((N, T), -np.inf)
    ptr = np.zeros((N, T), dtype=int)

    dp[:,0] = logpi + logB[:, obs_idx[0]]

    for t in range(1, T):
        for s in range(N):
            probs = dp[:,t-1] + logA[:,s] + logB[s, obs_idx[t]]
            ptr[s,t] = int(np.argmax(probs))
            dp[s,t] = float(np.max(probs))

    states_idx = np.zeros(T, dtype=int)
    states_idx[-1] = int(np.argmax(dp[:,T-1]))
    for t in range(T-1, 0, -1):
        states_idx[t-1] = ptr[states_idx[t], t]

    return states_idx.tolist()

def infer_with_hmmlearn(pi, A, B, observations_list, obs_seq):
    """
    Adapter robusto para hmmlearn.MultinomialHMM.
    Retorna (state_idx_list, route) donde route es 'simple' o 'one-hot'.
    """
    if hmmlearn_hmm is None:
        raise RuntimeError("hmmlearn no está disponible en este entorno.")

    import warnings

    try:
        hl_ver = getattr(hmmlearn, "__version__", "unknown")
    except Exception:
        hl_ver = "unknown"
    print(f"[debug] hmmlearn version: {hl_ver}")

    mapping = {o: i for i, o in enumerate(observations_list)}
    obs_idx = [mapping[o] for o in obs_seq]

    pi_arr = np.asarray(pi, dtype=float).copy()
    A_arr = np.asarray(A, dtype=float).copy()
    B_arr = np.asarray(B, dtype=float).copy()

    n_states = A_arr.shape[0]
    n_symbols = B_arr.shape[1]

    model = hmmlearn_hmm.MultinomialHMM(n_components=n_states, init_params="")
    model.startprob_ = pi_arr
    model.transmat_ = A_arr
    model.emissionprob_ = B_arr

    warnings.filterwarnings("ignore", message="MultinomialHMM has undergone major changes.*")

    e_simple = None
    e_onehot = None

    obs_arr_simple = np.array(obs_idx, dtype=int).reshape(-1, 1)
    T = obs_arr_simple.shape[0]
    try:
        n_trials_simple = np.ones(T, dtype=int)
        try:
            model.n_trials = n_trials_simple
        except Exception:
            pass
        try:
            model.n_trials_ = n_trials_simple
        except Exception:
            pass

        logprob, state_seq_idx = model.decode(obs_arr_simple, algorithm="viterbi")
        print("[debug] infer_with_hmmlearn: decode OK using simple (n,1) input.")
        return state_seq_idx.tolist(), "simple"
    except Exception as exc:
        e_simple = exc
        print(f"[debug] decode with (n,1) failed: {type(e_simple).__name__}: {e_simple}")

    try:
        onehot = np.zeros((T, n_symbols), dtype=int)
        for t, k in enumerate(obs_idx):
            if k < 0 or k >= n_symbols:
                raise ValueError(f"Observación con índice fuera de rango: {k} (n_symbols={n_symbols})")
            onehot[t, k] += 1
        n_trials_counts = onehot.sum(axis=1).astype(int)
        try:
            model.n_trials = n_trials_counts
        except Exception:
            pass
        try:
            model.n_trials_ = n_trials_counts
        except Exception:
            pass

        logprob, state_seq_idx = model.decode(onehot, algorithm="viterbi")
        print("[debug] infer_with_hmmlearn: decode OK using one-hot/counts input.")
        return state_seq_idx.tolist(), "one-hot"
    except Exception as exc2:
        e_onehot = exc2
        print(f"[debug] decode with one-hot failed: {type(e_onehot).__name__}: {e_onehot}")

    msg = (
        "infer_with_hmmlearn: ambos intentos de decode fallaron.\n"
        f"Error simple: {type(e_simple).__name__}: {e_simple}\n"
        f"Error one-hot: {type(e_onehot).__name__}: {e_onehot}\n"
        "Revisa shapes, número de símbolos y versión de hmmlearn."
    )
    raise RuntimeError(msg)

def infer_viterbi(pi, A, B, observations_list, obs_seq):
    """
    Conveniencia: por defecto usa el fallback.
    """
    if len(obs_seq) == 0:
        return []
    return viterbi_fallback(pi, A, B, observations_list, obs_seq)
