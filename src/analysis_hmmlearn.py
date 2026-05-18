"""
analysis_hmmlearn.py: Utilidades para ejecutar y comparar inferencia con hmmlearn y con el fallback.
"""
import numpy as np
import inference

def run_hmmlearn_decode(model):
    """
    Ejecuta decode con hmmlearn (si está disponible).
    Retorna (inferred_states_list, used_backend, message).
    """
    try:
        inferred_idx, route = inference.infer_with_hmmlearn(model.pi, model.A, model.B, model.observations, model.observation_sequence)
        inferred_states = [model.states[i] for i in inferred_idx]
        return (inferred_states, "hmmlearn", f"OK (route={route})")
    except Exception as e:
        inferred_idx = inference.viterbi_fallback(model.pi, model.A, model.B, model.observations, model.observation_sequence)
        inferred_states = [model.states[i] for i in inferred_idx]
        return (inferred_states, "fallback", f"hmmlearn falló: {e}; se usó fallback")

def compare_backends(model):
    """
    Ejecuta ambas inferencias (fallback y hmmlearn si posible) y devuelve resumen.
    """
    results = {}
    fb_idx = inference.viterbi_fallback(model.pi, model.A, model.B, model.observations, model.observation_sequence)
    fb_states = [model.states[i] for i in fb_idx]
    real_states = model.state_sequence[1:1+len(fb_states)]
    mismatches = sum(1 for a,b in zip(real_states, fb_states) if a != b)
    acc = (len(fb_states)-mismatches)/len(fb_states) if fb_states else 0.0
    results['fallback'] = {'states': fb_states, 'idx': fb_idx, 'acc': acc, 'mismatches': mismatches}

    try:
        hl_idx, route = inference.infer_with_hmmlearn(model.pi, model.A, model.B, model.observations, model.observation_sequence)
        hl_states = [model.states[i] for i in hl_idx]
        real_hl = model.state_sequence[1:1+len(hl_states)]
        mism_hl = sum(1 for a,b in zip(real_hl, hl_states) if a != b)
        acc_hl = (len(hl_states)-mism_hl)/len(hl_states) if hl_states else 0.0
        results['hmmlearn'] = {'states': hl_states, 'idx': hl_idx, 'acc': acc_hl, 'mismatches': mism_hl, 'route': route}
        note = f"hmmlearn ejecutado correctamente (route={route})."
    except Exception as e:
        results['hmmlearn'] = None
        note = f"hmmlearn no devolvió resultado: {e}"

    diff_positions = []
    if results.get('hmmlearn'):
        minL = min(len(results['fallback']['idx']), len(results['hmmlearn']['idx']))
        for i in range(minL):
            if results['fallback']['idx'][i] != results['hmmlearn']['idx'][i]:
                diff_positions.append(i)

    results['diff_positions'] = diff_positions
    results['note'] = note
    return results

def compare_and_summary(model):
    return compare_backends(model)
