"""model.py - Simulación HMM, log, métricas y validaciones empíricas.
"""

import time
import numpy as np
import pandas as pd

#MOTOR DE LA SIMULACION:
# Encapsula toda la lógica matemática probabilística del modelo de Markov oculto
# Genera las trayectorias de estados, observaciones y precios
class HMMModel:
    def __init__(self, states, observations, transition_matrix, emission_matrix, initial_distribution): # Valida Parametros de clase HMMapp del Main.py
        self.states = list(states)
        self.observations = list(observations)
        self.n_states = len(self.states)
        self.n_obs = len(self.observations)

        self.A = np.array(transition_matrix, dtype=float) # Matriz de transición 𝐴: almacena las probabilidades de pasar de un estado a otro
        self.B = np.array(emission_matrix, dtype=float) # Matriz de emisión 𝐵: almacena las probabilidades de emisión de observaciones dadas los estados.
        self.pi = np.array(initial_distribution, dtype=float) # Vector de probabilidades: Distribución inicial:

        self._validate_matrices()
        self.reset()

    def _validate_matrices(self):
        if not np.allclose(self.A.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("Cada fila de la matriz de transición A debe sumar 1.") # Se valida que la probabilidad sume 1
        if not np.allclose(self.B.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("Cada fila de la matriz de emisión B debe sumar 1.") # Se valida que la probabilidad sume 1
        if not np.isclose(self.pi.sum(), 1.0, atol=1e-8):
            raise ValueError("La distribución inicial pi debe sumar 1.") # Se valida que la probabilidad sume 1

    def reset(self):
        self.state_sequence = []
        self.observation_sequence = []
        self.price_series = []
        self.log = []

    def simulate(self, periods=50, initial_price=100.0, pre_generate=True):
        """
        Genera una simulación completa. pre_generate=True genera 'periods' pasos.
        """
        self.reset()
        current_state = np.random.choice(self.states, p=self.pi)
        self.state_sequence.append(current_state)
        self.price_series.append(float(initial_price))

        if pre_generate:
            for t in range(periods):
                self._simulate_step_internal(t, current_state, self.price_series[-1])
                current_state = self.state_sequence[-1]

    def simulate_step(self):
        """
        Avanza un paso en la simulación (modo step).
        """
        if not self.state_sequence:
            current_state = np.random.choice(self.states, p=self.pi)
            self.state_sequence.append(current_state)
            self.price_series.append(100.0)
            return True
        t = len(self.state_sequence) - 1
        current_state = self.state_sequence[-1]
        current_price = self.price_series[-1]
        self._simulate_step_internal(t, current_state, current_price)
        return True

    def _simulate_step_internal(self, t, current_state, current_price):
        si = self.states.index(current_state)
        trans_probs = self.A[si]
        next_state = np.random.choice(self.states, p=trans_probs)
        ns_i = self.states.index(next_state)
        emit_probs = self.B[ns_i]
        obs = np.random.choice(self.observations, p=emit_probs)

        price_prev = float(current_price)
        if obs == "Retorno Positivo":
            current_price *= (1 + np.random.uniform(0.01, 0.03))
        elif obs == "Retorno Neutral":
            current_price *= (1 + np.random.uniform(-0.005, 0.005))
        else:
            current_price *= (1 - np.random.uniform(0.01, 0.03))

        self.state_sequence.append(next_state)
        self.observation_sequence.append(obs)
        self.price_series.append(float(current_price))

        self.log.append({
            "period": t,
            "state_prev": current_state,
            "state_next": next_state,
            "trans_probs": trans_probs.tolist(),
            "observation": obs,
            "emit_probs": emit_probs.tolist(),
            "price_prev": round(price_prev, 6),
            "price_next": round(current_price, 6),
            "timestamp": time.time()
        })

    # Export y análisis
    def export_log_csv(self, filename):
        if not self.log:
            raise ValueError("No hay log para exportar. Ejecuta simulate() primero.")
        df = pd.DataFrame(self.log)
        df.to_csv(filename, index=False)
        return filename

    def analyze(self):
        if len(self.price_series) < 2:
            return {}
        prices = np.array(self.price_series)
        returns = np.diff(prices) / prices[:-1]

        durations = {s: [] for s in self.states}
        changes = []
        current = self.state_sequence[0]
        count = 1
        for i in range(1, len(self.state_sequence)):
            s = self.state_sequence[i]
            if s == current:
                count += 1
            else:
                durations[current].append(count)
                changes.append((i-1, current, s))
                current = s
                count = 1
        durations[current].append(count)
        avg_durations = {s: float(np.mean(durations[s])) if durations[s] else 0.0 for s in self.states}

        total_return = float((prices[-1] / prices[0]) - 1.0)

        vol_by_state = {}
        for s in self.states:
            idxs = [i for i, st in enumerate(self.state_sequence[:-1]) if st == s]
            state_returns = returns[idxs] if idxs else np.array([])
            vol_by_state[s] = float(np.std(state_returns)) if state_returns.size > 0 else 0.0

        peak = prices[0]
        max_dd = 0.0
        for p in prices:
            if p > peak:
                peak = p
            dd = (peak - p) / peak
            if dd > max_dd:
                max_dd = dd

        return {
            "avg_durations": avg_durations,
            "total_return": total_return,
            "volatilities": vol_by_state,
            "max_drawdown": max_dd,
            "change_points": changes
        }

    def validate_empirical_transitions(self):
        counts = np.zeros_like(self.A)
        for i in range(len(self.state_sequence)-1):
            a = self.states.index(self.state_sequence[i])
            b = self.states.index(self.state_sequence[i+1])
            counts[a,b] += 1
        totals = counts.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            empirical = np.nan_to_num(counts / totals[:, None])
        diff = np.abs(empirical - self.A)
        return {"empirical": empirical, "theoretical": self.A, "abs_diff": diff, "max_diff": float(np.max(diff))}

    def validate_empirical_emissions(self):
        counts = np.zeros((self.n_states, self.n_obs))
        for i in range(1, len(self.state_sequence)):
            s = self.states.index(self.state_sequence[i])
            o = self.observations.index(self.observation_sequence[i-1])
            counts[s,o] += 1
        totals = counts.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            empirical = np.nan_to_num(counts / totals[:, None])
        diff = np.abs(empirical - self.B)
        return {"empirical": empirical, "theoretical": self.B, "abs_diff": diff, "max_diff": float(np.max(diff))}
