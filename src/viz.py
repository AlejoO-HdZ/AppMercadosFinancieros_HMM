"""
viz.py

Funciones de visualización reutilizables para la aplicación HMM.
Incluye:
- plot_price_with_regimes
- plot_observations_bar
- draw_transition_graph
- plot_state_distribution
- draw_colored_state_graph
- plot_viterbi_comparison
- plot_viterbi_lines
- plot_convergence_curve
- plot_state_evolution_comparison
- plot_state_heatmap

Dependencias: matplotlib, seaborn, networkx, numpy, pandas
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd

sns.set_style("whitegrid")
sns.set_palette("muted")


def plot_price_with_regimes(ax, prices, state_sequence, highlight_index=None, cmap=None, ypad=0.05):
    """
    Dibuja la serie de precios y colorea el fondo según el régimen (state_sequence).
    - ax: Axes matplotlib
    - prices: lista o array de precios (len N)
    - state_sequence: lista de estados (len N o N+1 según alineación)
    - highlight_index: índice para resaltar la transición actual (opcional)
    - cmap: dict estado->color
    - ypad: padding vertical relativo
    """
    if cmap is None:
        cmap = {"Mercado Alcista": "#b7e4c7", "Mercado Lateral": "#fff3b0", "Mercado Bajista": "#ffb4a2"}

    ax.clear()
    if not prices:
        ax.set_ylabel("Precio")
        ax.set_title("Precio (no hay datos)")
        return

    x = np.arange(len(prices))
    ax.plot(x, prices, color="black", linewidth=1.2)

    # colorear intervalos entre t y t+1 según estado en t (si hay alineación)
    n = len(prices)
    for i in range(n - 1):
        st = state_sequence[i] if i < len(state_sequence) else None
        color = cmap.get(st, "#e0e0e0")
        alpha = 0.45 if (highlight_index is not None and highlight_index == i) else 0.18
        ax.axvspan(i, i + 1, color=color, alpha=alpha)

    if highlight_index is not None and 0 <= highlight_index < len(prices):
        ax.plot([highlight_index], [prices[highlight_index]], marker="o", color="black", markersize=6)

    ax.set_ylabel("Precio")
    ax.set_xlabel("Período")
    ymin, ymax = ax.get_ylim()
    dy = ymax - ymin
    ax.set_ylim(ymin - dy * ypad, ymax + dy * ypad)
    ax.set_title("Precio con regímenes")


def plot_observations_bar(ax, observation_sequence):
    """
    Dibuja barras categóricas para observaciones discretas (Positivo/Neutral/Negativo).
    - ax: Axes matplotlib
    - observation_sequence: lista de strings
    """
    ax.clear()
    if not observation_sequence:
        ax.set_ylabel("Observaciones")
        ax.set_title("Observaciones (no hay datos)")
        return

    vals = [1 if o == "Retorno Positivo" else 0 if o == "Retorno Neutral" else -1 for o in observation_sequence]
    colors = ["#2b8cbe" if v == 1 else "#6c757d" if v == 0 else "#de2d26" for v in vals]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_ylabel("Retorno (cat)")
    ax.set_xlabel("Período")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Negativo", "Neutral", "Positivo"])
    ax.set_title("Observaciones por período")


def draw_transition_graph(ax, states, A, highlight_edges=None, show_empirical=None):
    """
    Dibuja un grafo dirigido con pesos de la matriz de transición A.
    - highlight_edges: lista de tuplas (u,v) para resaltar
    - show_empirical: dict {(u,v): value} para mostrar junto al peso teórico
    """
    ax.clear()
    G = nx.DiGraph()
    for s in states:
        G.add_node(s)

    for i, u in enumerate(states):
        for j, v in enumerate(states):
            weight = float(A[i, j])
            if weight > 0:
                G.add_edge(u, v, weight=weight)

    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#8E62A5", edgecolors="gray", node_size=900,alpha=0.7)
    label_pos = {n: (pos[n][0], pos[n][1] + 0.30) for n in pos}
    nx.draw_networkx_labels(G, label_pos, ax=ax, font_size=8)

    edges = list(G.edges(data=True))
    widths = [max(0.5, e[2]["weight"] * 6) for e in edges]
    nx.draw_networkx_edges(G, pos, ax=ax, connectionstyle="arc3,rad=0.12", width=widths, arrowsize=12)

    edge_labels = {}
    for u, v, d in G.edges(data=True):
        label = f"{d['weight']:.2f}"
        if show_empirical and (u, v) in show_empirical:
            label += f" / {show_empirical[(u, v)]:.2f}"
        edge_labels[(u, v)] = label

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax, label_pos=0.5)

    if highlight_edges:
        for (u, v) in highlight_edges:
            if G.has_edge(u, v):
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, edge_color="#ff7f0e", width=4, arrowsize=14)

    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass
    ax.margins(0.18)
    ax.set_axis_off()
    ax.set_title("Grafo de transición (teórico)")


def plot_state_distribution(ax, state_sequence, state_names):
    """
    Dibuja la distribución empírica de estados (frecuencia relativa).
    """
    ax.clear()
    if not state_sequence:
        ax.set_title("Distribución de estados (no hay datos)")
        return

    counts = pd.Series(state_sequence).value_counts().reindex(state_names).fillna(0)
    probs = counts / counts.sum()
    bars = ax.bar(state_names, probs.values, color=["#b7e4c7", "#fff3b0", "#ffb4a2"])
    ax.set_ylim(0, max(probs.values) * 1.25 if probs.values.max() > 0 else 1)
    ax.set_ylabel("Frecuencia relativa")
    ax.set_title("Distribución empírica de estados")
    for rect, val in zip(bars, probs.values):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.01, f"{val:.2%}", ha="center", va="bottom", fontsize=8)


def draw_colored_state_graph(ax, states, A, state_sequence):
    """
    Dibuja el grafo de transición con nodos coloreados según frecuencia empírica.
    - state_sequence: lista de estados simulados (para calcular frecuencias)
    """
    ax.clear()
    counts = pd.Series(state_sequence).value_counts().reindex(states).fillna(0)
    freqs = counts / counts.sum() if counts.sum() > 0 else pd.Series([0] * len(states), index=states)
    norm = (freqs - freqs.min()) / (freqs.max() - freqs.min() + 1e-12)
    cmap = plt.cm.OrRd

    G = nx.DiGraph()
    for s in states:
        G.add_node(s)
    for i, u in enumerate(states):
        for j, v in enumerate(states):
            weight = float(A[i, j])
            if weight > 0:
                G.add_edge(u, v, weight=weight)

    pos = nx.circular_layout(G)
    node_colors = [cmap(norm[s]) for s in states]
    node_sizes = [300 + 1200 * float(freqs[s]) for s in states]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors="k")
    label_pos = {n: (pos[n][0], pos[n][1] + 0.12) for n in pos}
    nx.draw_networkx_labels(G, label_pos, ax=ax, font_size=9)

    edges = list(G.edges(data=True))
    widths = [max(0.5, e[2]["weight"] * 6) for e in edges]
    nx.draw_networkx_edges(G, pos, ax=ax, connectionstyle="arc3,rad=0.12", width=widths, arrowsize=12)

    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax, label_pos=0.5)

    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass
    ax.margins(0.18)
    ax.set_axis_off()
    ax.set_title("Diagrama de estados coloreado por frecuencia empírica")


def plot_viterbi_comparison(ax, real_states, inferred_states, state_names):
    """
    Dibuja la comparación entre secuencia real e inferida.
    Marca mismatches con puntos rojos.
    """
    ax.clear()
    if not real_states and not inferred_states:
        ax.set_title("No hay datos para Viterbi")
        return

    real_states = [str(s) for s in real_states]
    inferred_states = [str(s) for s in inferred_states]
    state_names = [str(s) for s in state_names]

    len_real = len(real_states)
    len_inf = len(inferred_states)
    note = ""
    if len_real != len_inf:
        note = f" (longitudes difieren: real={len_real}, inferida={len_inf})"
        L = min(len_real, len_inf)
        if L == 0:
            ax.set_title("Secuencia real o inferida vacía; no hay comparación posible")
            return
        real_states = real_states[:L]
        inferred_states = inferred_states[:L]

    mapping = {s: i for i, s in enumerate(state_names)}

    def safe_map(s):
        return mapping.get(s, -1)

    real_idx = [safe_map(s) for s in real_states]
    inf_idx = [safe_map(s) for s in inferred_states]

    if all(i == -1 for i in real_idx) and all(i == -1 for i in inf_idx):
        ax.set_title("Estados no reconocidos; revisa nombres en el modelo")
        return

    x = range(len(real_idx))
    real_plot = [np.nan if i == -1 else i for i in real_idx]
    inf_plot = [np.nan if i == -1 else i for i in inf_idx]

    ax.plot(x, real_plot, label="Real", color="#2b8cbe", marker="o", linewidth=1.5)
    ax.plot(x, inf_plot, label="Inferida", color="#ff7f0e", marker="s", linewidth=1.2)

    mismatches_x = [i for i, (r, inf) in enumerate(zip(real_idx, inf_idx)) if r != -1 and inf != -1 and r != inf]
    mismatches_y = [inf_idx[i] for i in mismatches_x]
    if mismatches_x:
        ax.scatter(mismatches_x, mismatches_y, color="red", s=60, label="Mismatch", zorder=5)

    yticks = list(range(len(state_names)))
    ax.set_yticks(yticks)
    ax.set_yticklabels(state_names)
    ax.set_xlabel("Período (t)")
    ax.set_title("Comparación: secuencia real vs inferida (Viterbi)" + note)
    ax.legend(fontsize=8)
    ax.grid(True)


def plot_viterbi_lines(ax, real_states, inferred_states, state_names):
    """
    Dibuja solo las líneas de evolución (real vs inferida) sin marcar mismatches.
    Útil para mostrar la evolución pura sin puntos rojos.
    """
    ax.clear()
    if not real_states and not inferred_states:
        ax.set_title("No hay datos para Viterbi")
        return

    real_states = [str(s) for s in real_states]
    inferred_states = [str(s) for s in inferred_states]
    state_names = [str(s) for s in state_names]

    L = min(len(real_states), len(inferred_states))
    if L == 0:
        ax.set_title("Secuencia real o inferida vacía")
        return

    real_states = real_states[:L]
    inferred_states = inferred_states[:L]

    mapping = {s: i for i, s in enumerate(state_names)}
    real_idx = [mapping.get(s, -1) for s in real_states]
    inf_idx = [mapping.get(s, -1) for s in inferred_states]

    x = range(L)
    real_plot = [np.nan if i == -1 else i for i in real_idx]
    inf_plot = [np.nan if i == -1 else i for i in inf_idx]

    ax.plot(x, real_plot, label="Real", color="#2b8cbe", marker='o', linewidth=1.5)
    ax.plot(x, inf_plot, label="Inferida", color="#ff7f0e", marker='s', linewidth=1.2)

    yticks = list(range(len(state_names)))
    ax.set_yticks(yticks)
    ax.set_yticklabels(state_names)
    ax.set_xlabel("Período (t)")
    ax.set_title("Evolución: real vs inferida")
    ax.legend(fontsize=8)
    ax.grid(True)


def plot_convergence_curve(ax, history, tol):
    """
    Dibuja la curva de convergencia (history: lista de max_diff por run).
    """
    ax.clear()
    if not history:
        ax.set_title("No hay datos de convergencia")
        return

    x = np.arange(1, len(history) + 1)
    ax.plot(x, history, marker='o', color="#2b8cbe", linewidth=1.5, label="max_diff (empírica vs teórica)")
    ax.axhline(tol, color='red', linestyle='--', label=f"tol = {tol:.4f}")

    crossing = None
    for i, val in enumerate(history):
        if val <= tol:
            crossing = i + 1
            break
    if crossing:
        ax.axvline(crossing, color='green', linestyle=':', label=f"convergencia en run {crossing}")
        ax.annotate(f"run {crossing}", xy=(crossing, history[crossing - 1]), xytext=(crossing, max(history)),
                    arrowprops=dict(arrowstyle="->", color="green"))

    ax.set_xlabel("Número de simulaciones (runs)")
    ax.set_ylabel("max_diff (empírica vs teórica)")
    ax.set_title("Convergencia de frecuencias empíricas")
    ax.legend()
    ax.grid(True)


def plot_state_evolution_comparison(ax, model_A, state_sequence, state_names):
    """
    Compara la proporción acumulada simulada con la probabilidad estacionaria teórica.
    - model_A: matriz de transición teórica (para calcular vector estacionario)
    - state_sequence: secuencia simulada (lista)
    """
    ax.clear()
    if not state_sequence:
        ax.set_title("Evolución de estados (no hay datos)")
        return

    A = np.array(model_A)
    vals, vecs = np.linalg.eig(A.T)
    idx = np.argmin(np.abs(vals - 1.0))
    stat = np.real(vecs[:, idx])
    stat = stat / stat.sum()

    seq = pd.Series(state_sequence)
    cum_props = []
    for t in range(1, len(seq) + 1):
        counts = seq.iloc[:t].value_counts().reindex(state_names).fillna(0)
        cum_props.append((counts / counts.sum()).values)
    cum_props = np.array(cum_props)
    x = np.arange(1, len(seq) + 1)
    colors = ["#2b8cbe", "#6c757d", "#de2d26"]
    for i, s in enumerate(state_names):
        ax.plot(x, cum_props[:, i], label=f"Simulado: {s}", color=colors[i], linewidth=1.2)
        ax.hlines(stat[i], xmin=1, xmax=len(seq), colors=colors[i], linestyles='--', alpha=0.6, label=f"Teórico: {s}")
    ax.set_xlabel("Período (t)")
    ax.set_ylabel("Proporción acumulada")
    ax.set_title("Evolución: proporción simulada vs prob. estacionaria teórica")
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True)


def plot_state_heatmap(ax, state_sequence, state_names):
    """
    Dibuja un heatmap binario de presencia de estados por período.
    """
    ax.clear()
    if not state_sequence:
        ax.set_title("Heatmap de estados (no hay datos)")
        return
    seq = pd.Series(state_sequence)
    T = len(seq)
    n = len(state_names)
    mat = np.zeros((n, T))
    mapping = {s: i for i, s in enumerate(state_names)}
    for t, s in enumerate(seq):
        mat[mapping[s], t] = 1

    sns.heatmap(mat, cmap="viridis", cbar=True, ax=ax, xticklabels=max(1, T // 10), yticklabels=state_names)
    ax.set_xlabel("Período (t)")
    ax.set_ylabel("Estado")
    ax.set_title("Heatmap: presencia de estados por período")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
