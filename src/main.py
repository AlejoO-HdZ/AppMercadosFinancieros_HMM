# main.py
"""HMM Simulación - main.py
- Incluye editor de matrices A, B y distribución inicial pi.
- Panel principal: grafo coloreado por frecuencia empírica y recorrido resaltado.
- Ventana Resultados: grafo a la izquierda, evolución a la derecha; retorno y volatilidad.
- Al finalizar una simulación (Simular 50, ejecución automática completa o Simular Muchas)
  se muestra una ventana con:
    - a la izquierda: histograma comparado de retornos alcistas vs bajistas (título en español).
    - a la derecha: evolución simulada de proporciones acumuladas vs prob. estacionaria teórica
    - Conclusiónes.
Requiere: model.py, inference.py, viz.py, analysis_hmmlearn.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from model import HMMModel
import inference
import viz
import analysis_hmmlearn

# ---------------- Defaults (**Especificaciones del modelo:**)----------------
STATES = ["Mercado Alcista", "Mercado Lateral", "Mercado Bajista"] # Estados Ocultos
OBS = ["Retorno Positivo", "Retorno Neutral", "Retorno Negativo"] # Observaciones
# Matriz A de Transicion: probabilidades de pasar de un estado a otro
A_DEFAULT = [
    [0.6, 0.3, 0.1], # Suma de prob. fila Alcista= 1
    [0.3, 0.4, 0.3], # Suma de prob. fila Lateral= 1
    [0.1, 0.3, 0.6] # Suma de prob. fila Bajista= 1
]
# Matriz B de Emision: probabilidades de observar cada tipo de retorno dado un estado
B_DEFAULT = [
    [0.7, 0.2, 0.1], # Suma de prob. fila Alcista= 1
    [0.3, 0.4, 0.3], # Suma de prob. fila Lateral= 1
    [0.1, 0.2, 0.7] # Suma de prob. fila Bajista = 1
]
PI_DEFAULT = [0.4, 0.3, 0.3] # Distribuicion Inicial

# ---------------- Tooltip helper ----------------
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + 18
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify="left", background="#fff8dc", relief="solid", borderwidth=1, font=("Segoe UI", 9))
        label.pack(ipadx=4, ipady=2)

    def hide(self, _event=None):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()
# ---------------- App ----------------
class HMMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HMM Simulación Ejemplo 3 - Análisis de Mercado Financiero")
        self.model = HMMModel(STATES, OBS, A_DEFAULT, B_DEFAULT, PI_DEFAULT) # Llamamos la clase HMMModel (motor de la simulacion) y le pasamos parmetros simulacion
        self.periods = 50
        self.t = 0
        self.playing = False
        self.delay = 600  # ms
        self._convergence_history = []

        # Top controls
        top = ttk.Frame(root)
        top.pack(side="top", fill="x", padx=8, pady=8)

        style = ttk.Style()
        try:
            style.theme_use("default")
        except Exception:
            pass
        style.configure("Main.TButton", padding=6, relief="flat", background="#f6fbff")
        style.map("Main.TButton", background=[("active", "#eaf6ff"), ("pressed", "#dfeefc")])

        # Primary buttons
        self.btn_start = ttk.Button(top, text="Start", command=self.start, style="Main.TButton")
        self.btn_pause = ttk.Button(top, text="Pause", command=self.pause, style="Main.TButton")
        self.btn_step = ttk.Button(top, text="Step", command=self.step, style="Main.TButton")
        self.btn_back = ttk.Button(top, text="Back", command=self.back, style="Main.TButton")
        self.btn_reset = ttk.Button(top, text="Reset", command=self.reset, style="Main.TButton")
        for b in (self.btn_start, self.btn_pause, self.btn_step, self.btn_back, self.btn_reset):
            b.pack(side="left", padx=4)
        ToolTip(self.btn_start, "Inicia la simulación automática.")
        ToolTip(self.btn_pause, "Pausa la simulación.")
        ToolTip(self.btn_step, "Avanza un paso en la simulación.")
        ToolTip(self.btn_back, "Retrocede un paso en la simulación.")
        ToolTip(self.btn_reset, "Reinicia la simulación y limpia datos.")

        # Secondary buttons
        self.btn_sim50 = ttk.Button(top, text="Simular 50", command=self.run_full_simulation)
        self.btn_sim50.pack(side="left", padx=8)
        self.btn_many = ttk.Button(top, text="Simular Muchas", command=self.run_many_simulations_dialog)
        self.btn_many.pack(side="left", padx=4)
        self.btn_export = ttk.Button(top, text="Exportar Excel", command=self.export_csv)
        self.btn_export.pack(side="left", padx=4)
        self.btn_viterbi = ttk.Button(top, text="Inferir (Viterbi)", command=self.run_inference_plot)
        self.btn_viterbi.pack(side="left", padx=6)
        self.btn_hmm = ttk.Button(top, text="Inferir (hmmlearn)", command=self.run_inference_hmmlearn_plot)
        self.btn_hmm.pack(side="left", padx=6)
        self.btn_compare = ttk.Button(top, text="Comparar backends", command=self.compare_backends_window)
        self.btn_compare.pack(side="left", padx=6)

        # Results button
        btn_res = tk.Button(top, text="Resultados Simulación", bg="#ffb07c", fg="black", command=self.show_simulation_results)
        btn_res.pack(side="right", padx=6)
        ToolTip(btn_res, "Abre la ventana con las gráficas de resultados y conclusiones.")

        # Options
        self.perturb_var = tk.BooleanVar(value=False)
        chk = ttk.Checkbutton(top, text="Forzar perturbación B (test)", variable=self.perturb_var)
        chk.pack(side="right", padx=6)
        self.infer_source = tk.StringVar(value="obs")
        frm_src = ttk.Frame(top); frm_src.pack(side="right", padx=6)
        ttk.Label(frm_src, text="Fuente:").pack(side="left")
        ttk.Radiobutton(frm_src, text="Observaciones", variable=self.infer_source, value="obs").pack(side="left")
        ttk.Radiobutton(frm_src, text="Estados reales (test)", variable=self.infer_source, value="true").pack(side="left")

        # Main layout
        main = ttk.Frame(root); main.pack(fill="both", expand=True)
        left_col = ttk.Frame(main, width=300); left_col.pack(side="left", fill="y", padx=6, pady=6)
        center_col = ttk.Frame(main); center_col.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        right_col = ttk.Frame(main, width=420); right_col.pack(side="right", fill="y", padx=6, pady=6)

        # Left: editor + table + metrics
        self._build_matrices_editor(left_col)
        self._build_states_obs_table(left_col)
        self._build_metrics_text_under_table(left_col)

        # Center: live figures (price, obs, graph)
        self.fig, (self.ax_price, self.ax_obs, self.ax_graph) = plt.subplots(3, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1, 2]})
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=center_col)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Right: log
        ttk.Label(right_col, text="Log (eventos)", font=("Segoe UI", 11, "bold")).pack(anchor="nw", pady=(6, 0))
        log_frame = ttk.Frame(right_col); log_frame.pack(fill="both", expand=True, padx=4, pady=4)
        self.txt_log = tk.Text(log_frame, width=56, height=34, wrap="none")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=log_scroll.set)
        self.txt_log.pack(side="left", fill="both", expand=True); log_scroll.pack(side="right", fill="y")

        self._draw_initial()

    # ---------------- Left panel builders ----------------
    def _build_matrices_editor(self, parent):
        frm = ttk.LabelFrame(parent, text="Editor: Matrices A y B")
        frm.pack(fill="x", padx=4, pady=4)

        ttk.Label(frm, text="Estados:").grid(row=0, column=0, columnspan=4, sticky="w", pady=(4, 2))
        ttk.Label(frm, text=", ".join(self.model.states), font=("Segoe UI", 9)).grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 6))

        ttk.Label(frm, text="Matriz A (transición)").grid(row=2, column=0, columnspan=4, pady=(4, 2))
        self.A_entries = []
        for i in range(len(self.model.states)):
            row_entries = []
            for j in range(len(self.model.states)):
                e = ttk.Entry(frm, width=7)
                e.grid(row=3 + i, column=j, padx=2, pady=2)
                e.insert(0, f"{self.model.A[i, j]:.3f}")
                row_entries.append(e)
            self.A_entries.append(row_entries)

        base_row = 3 + len(self.model.states)
        ttk.Label(frm, text="Matriz B (emisión)").grid(row=base_row, column=0, columnspan=4, pady=(8, 2))
        self.B_entries = []
        for i in range(len(self.model.states)):
            row_entries = []
            for j in range(len(self.model.observations)):
                e = ttk.Entry(frm, width=7)
                e.grid(row=base_row + 1 + i, column=j, padx=2, pady=2)
                e.insert(0, f"{self.model.B[i, j]:.3f}")
                row_entries.append(e)
            self.B_entries.append(row_entries)

        # Sección: Distribución inicial (pi)
        pi_row = base_row + 1 + len(self.model.states)
        ttk.Label(frm, text="Distribución inicial (pi)").grid(row=pi_row, column=0, columnspan=4, pady=(8, 2))
        self.pi_entries = []
        for i, s in enumerate(self.model.states):
            e = ttk.Entry(frm, width=7)
            e.grid(row=pi_row + 1, column=i, padx=2, pady=2)
            e.insert(0, f"{self.model.pi[i]:.3f}")
            self.pi_entries.append(e)

        ttk.Button(frm, text="Aplicar matrices y pi", command=self.apply_matrices_and_simulate).grid(row=pi_row + 2, column=0, columnspan=4, pady=(8, 6))

    def _build_states_obs_table(self, parent):
        frm = ttk.LabelFrame(parent, text="Estados ocultos y Observaciones")
        frm.pack(fill="both", padx=4, pady=6, expand=False)

        cols = ("t", "estado_real", "observacion")
        self.tree_states_obs = ttk.Treeview(frm, columns=cols, show="headings", height=8)
        for c, h in zip(cols, ["t", "Estado oculto", "Observación"]):
            self.tree_states_obs.heading(c, text=h)
            self.tree_states_obs.column(c, width=80, anchor="center")
        self.tree_states_obs.pack(fill="both", expand=True, padx=4, pady=4)

        btns_frm = ttk.Frame(frm)
        btns_frm.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(btns_frm, text="Actualizar tabla", command=self._refresh_states_obs_table).pack(side="left")
        ttk.Button(btns_frm, text="Limpiar simulación", command=self.reset).pack(side="left", padx=6)

    def _build_metrics_text_under_table(self, parent):
        frm = ttk.LabelFrame(parent, text="Métricas (resumen)")
        frm.pack(fill="both", padx=4, pady=6, expand=False)
        self.txt_metrics_small = tk.Text(frm, height=18, width=34, wrap="word")
        self.txt_metrics_small.pack(fill="both", expand=True, padx=4, pady=4)
        self.txt_metrics_small.configure(state="disabled")

    # ---------------- Apply matrices ----------------
    def apply_matrices_and_simulate(self):
        try:
            A_new = np.zeros_like(self.model.A)
            for i in range(len(self.model.states)):
                for j in range(len(self.model.states)):
                    val = float(self.A_entries[i][j].get())
                    A_new[i, j] = val
            A_new = np.clip(A_new, 0.0, None)
            row_sums = A_new.sum(axis=1)
            row_sums[row_sums == 0] = 1.0
            A_new = (A_new.T / row_sums).T

            B_new = np.zeros_like(self.model.B)
            for i in range(len(self.model.states)):
                for j in range(len(self.model.observations)):
                    val = float(self.B_entries[i][j].get())
                    B_new[i, j] = val
            B_new = np.clip(B_new, 0.0, None)
            row_sums_b = B_new.sum(axis=1)
            row_sums_b[row_sums_b == 0] = 1.0
            B_new = (B_new.T / row_sums_b).T

            # Leer y normalizar pi
            pi_new = np.zeros_like(self.model.pi)
            for i in range(len(self.model.states)):
                val = float(self.pi_entries[i].get())
                pi_new[i] = max(0.0, val)
            if pi_new.sum() == 0:
                pi_new = np.array(self.model.pi)
            else:
                pi_new = pi_new / pi_new.sum()

            self.model.A = A_new
            self.model.B = B_new
            self.model.pi = pi_new
            self.model._validate_matrices()
        except Exception as e:
            messagebox.showerror("Matrices", f"Error al leer/validar matrices/pi: {e}")
            return

        # actualizar entradas con valores normalizados
        for i in range(len(self.model.states)):
            for j in range(len(self.model.states)):
                self.A_entries[i][j].delete(0, tk.END)
                self.A_entries[i][j].insert(0, f"{self.model.A[i, j]:.3f}")
        for i in range(len(self.model.states)):
            for j in range(len(self.model.observations)):
                self.B_entries[i][j].delete(0, tk.END)
                self.B_entries[i][j].insert(0, f"{self.model.B[i, j]:.3f}")
        for i in range(len(self.model.states)):
            self.pi_entries[i].delete(0, tk.END)
            self.pi_entries[i].insert(0, f"{self.model.pi[i]:.3f}")

        self._refresh_states_obs_table()
        messagebox.showinfo("Matrices", "Matrices y pi aplicadas correctamente. Usa Start/Simular para ejecutar la simulación con estos valores.")

    # ---------------- Refresh table & metrics ----------------
    def _refresh_states_obs_table(self):
        try:
            self.tree_states_obs.delete(*self.tree_states_obs.get_children())
        except Exception:
            return
        for i in range(1, len(self.model.state_sequence)):
            estado = self.model.state_sequence[i]
            obs = self.model.observation_sequence[i - 1] if i - 1 < len(self.model.observation_sequence) else ""
            self.tree_states_obs.insert("", "end", values=(i - 1, estado, obs))
        try:
            children = self.tree_states_obs.get_children()
            if children:
                self.tree_states_obs.see(children[-1])
        except Exception:
            pass

        metrics = self.model.analyze() or {}
        avg_durs = metrics.get("avg_durations", {})
        total_return = metrics.get("total_return", 0.0)
        max_dd = metrics.get("max_drawdown", 0.0)
        vols = metrics.get("volatilities", {})

        self.txt_metrics_small.configure(state="normal")
        self.txt_metrics_small.delete("1.0", tk.END)
        self.txt_metrics_small.insert(tk.END, "Duración promedio por régimen:\n")
        for s, v in avg_durs.items():
            self.txt_metrics_small.insert(tk.END, f"  {s}: {v:.2f} períodos\n")
        self.txt_metrics_small.insert(tk.END, f"\nRetorno total acumulado: {total_return:.2%}\n")
        self.txt_metrics_small.insert(tk.END, f"\nVolatilidad por régimen:\n")
        for s, v in vols.items():
            self.txt_metrics_small.insert(tk.END, f"  {s}: {v:.4f}\n")
        self.txt_metrics_small.insert(tk.END, f"\nDrawdown máximo: {max_dd:.2%}\n")
        self.txt_metrics_small.configure(state="disabled")

    # ---------------- Draw initial ----------------
    def _draw_initial(self):
        self.ax_price.clear(); self.ax_obs.clear(); self.ax_graph.clear()
        self.ax_price.set_title("Precio (inspección rápida)")
        self.ax_obs.set_title("Observaciones (inspección rápida)")
        self.ax_graph.set_title("Grafo de estados ocultos")
        self.canvas.draw()

    # ---------------- Simulation control ----------------
    def run_full_simulation(self):
        self.model.simulate(periods=self.periods, initial_price=100.0, pre_generate=True)
        self.t = 0
        self._update_all_views()
        self.canvas.draw()
        self._refresh_states_obs_table()
        # Mostrar ventana resumen (histograma + evolución) al finalizar
        try:
            self.show_updown_and_theoretical_distribution()
        except Exception:
            pass

    def start(self):
        if not self.model.state_sequence:
            self.model.simulate(periods=self.periods, initial_price=100.0, pre_generate=False)
        self.playing = True
        self._auto_step()

    def pause(self):
        self.playing = False

    def _auto_step(self):
        if not self.playing:
            return
        if len(self.model.log) < self.periods:
            self.model.simulate_step()
        self.step()
        if self.playing and len(self.model.log) < self.periods:
            self.root.after(self.delay, self._auto_step)
        else:
            self.playing = False
            # Mostrar ventana resumen cuando la ejecución automática termina
            try:
                self.show_updown_and_theoretical_distribution()
            except Exception:
                pass

    def step(self):
        if len(self.model.log) == 0:
            self.model.simulate_step()
        if self.t >= len(self.model.log):
            return
        self._refresh_log_text()
        self._refresh_states_obs_table()
        self._update_plots(highlight_index=self.t)
        self._refresh_states_obs_table()
        self.t += 1

    def back(self):
        if self.t <= 0:
            return
        self.t -= 1
        self._refresh_log_text()
        self._refresh_states_obs_table()
        self._update_plots(highlight_index=max(0, self.t - 1))

    def reset(self):
        self.model.reset()
        self.t = 0
        self.playing = False
        self.txt_log.delete("1.0", tk.END)
        self.txt_metrics_small.configure(state="normal")
        self.txt_metrics_small.delete("1.0", tk.END)
        self.txt_metrics_small.configure(state="disabled")
        # reset pi entries to model default
        for i in range(len(self.model.states)):
            try:
                self.pi_entries[i].delete(0, tk.END)
                self.pi_entries[i].insert(0, f"{self.model.pi[i]:.3f}")
            except Exception:
                pass
        self._draw_initial()
        self._refresh_states_obs_table()

    def _refresh_log_text(self):
        self.txt_log.delete("1.0", tk.END)
        for e in self.model.log:
            line = f"t={e['period']} | {e['state_prev']} -> {e['state_next']} | obs={e['observation']} | {e['price_prev']} -> {e['price_next']}\n"
            line += f"    trans_probs={e['trans_probs']} emit_probs={e['emit_probs']}\n"
            self.txt_log.insert(tk.END, line)
        self.txt_log.see(tk.END)

    # ---------------- Update live plots (panel principal) ----------------
    def _update_plots(self, highlight_index=None):
        viz.plot_price_with_regimes(self.ax_price, self.model.price_series, self.model.state_sequence, highlight_index=highlight_index, ypad=0.06)
        viz.plot_observations_bar(self.ax_obs, self.model.observation_sequence)

        # Build empirical counts up to current time (seq_so_far)
        seq_so_far = self.model.state_sequence[:max(1, self.t + 1)]
        counts = {}
        total_from = {}
        for i in range(len(seq_so_far) - 1):
            u, v = seq_so_far[i], seq_so_far[i + 1]
            counts[(u, v)] = counts.get((u, v), 0) + 1
            total_from[u] = total_from.get(u, 0) + 1
        empirical = {}
        for (u, v), c in counts.items():
            empirical[(u, v)] = c / total_from[u] if total_from.get(u, 0) > 0 else 0.0

        try:
            # 1) color nodes by empirical frequency (using seq_so_far)
            viz.draw_colored_state_graph(self.ax_graph, self.model.states, self.model.A, seq_so_far)
            # 2) overlay transition graph to show edges and highlight current edge (no textual blocks)
            highlight_edges = None
            if highlight_index is not None and highlight_index < len(self.model.log):
                e = self.model.log[highlight_index]
                highlight_edges = [(e["state_prev"], e["state_next"])]
            viz.draw_transition_graph(self.ax_graph, self.model.states, self.model.A, highlight_edges=highlight_edges, show_empirical=empirical)
            self.ax_graph.set_title("Grafo de estados ocultos")
        except Exception:
            viz.draw_transition_graph(self.ax_graph, self.model.states, self.model.A, highlight_edges=None, show_empirical=None)
            self.ax_graph.set_title("Grafo de estados ocultos (estático)")

        self.fig.tight_layout()
        self.canvas.draw()

    # ---------------- Inference windows (unchanged) ----------------
    def run_inference_plot(self):
        if not self.model.observation_sequence:
            messagebox.showwarning("Inferencia", "Primero ejecuta la simulación."); return
        if self.infer_source.get() == "obs":
            obs_input = self.model.observation_sequence; source_note = "Observaciones simuladas"
        else:
            obs_input = [self.model.observations[int(np.argmax(self.model.B[self.model.states.index(s)]))] for s in self.model.state_sequence[1:]]
            source_note = "Estados reales (convertidos a observaciones) - MODO TEST"
        inferred_idx = inference.viterbi_fallback(self.model.pi, self.model.A, self.model.B, self.model.observations, obs_input)
        inferred_states = [self.model.states[i] for i in inferred_idx]; real_states = self.model.state_sequence[1:1 + len(inferred_states)]
        figc, axc = plt.subplots(figsize=(8, 3))
        viz.plot_viterbi_comparison(axc, real_states, inferred_states, self.model.states)
        mismatches = sum(1 for a, b in zip(real_states, inferred_states) if a != b); total_v = len(inferred_states)
        acc = (total_v - mismatches) / total_v if total_v > 0 else 0.0
        conclusion = f"Conclusión: exactitud (Viterbi fallback) = {acc:.2%} ({mismatches}/{total_v} mismatches). Fuente: {source_note}"
        axc.text(0.01, -0.25, conclusion, transform=axc.transAxes, fontsize=8, va='top')
        figc.tight_layout()
        win = tk.Toplevel(self.root); win.title("Inferencia: Viterbi (fallback)")
        canvasc = FigureCanvasTkAgg(figc, master=win); canvasc.get_tk_widget().pack(fill="both", expand=True); canvasc.draw()
        ttk.Label(win, text=f"Fuente usada: {source_note}  |  Backend: Viterbi fallback").pack(anchor="w", padx=6, pady=(4, 8))

    def run_inference_hmmlearn_plot(self):
        if not self.model.observation_sequence:
            messagebox.showwarning("Inferencia (hmmlearn)", "Primero ejecuta la simulación."); return
        if self.infer_source.get() == "obs":
            obs_input = self.model.observation_sequence; source_note = "Observaciones simuladas"
        else:
            obs_input = [self.model.observations[int(np.argmax(self.model.B[self.model.states.index(s)]))] for s in self.model.state_sequence[1:]]
            source_note = "Estados reales (convertidos a observaciones) - MODO TEST"
        B_used = self.model.B.copy(); perturbed = False
        if getattr(self, "perturb_var", None) and self.perturb_var.get():
            eps = 1e-6; rng = np.random.default_rng(0); noise = rng.uniform(-eps, eps, size=B_used.shape)
            B_used = np.clip(B_used + noise, 1e-12, None); B_used = (B_used.T / B_used.sum(axis=1)).T; perturbed = True
        try:
            inferred_idx, route = inference.infer_with_hmmlearn(self.model.pi, self.model.A, B_used, self.model.observations, obs_input)
        except Exception as e:
            self.txt_log.insert(tk.END, f"[error] infer_with_hmmlearn: {e}\n"); self.txt_log.see(tk.END)
            inferred_idx = inference.viterbi_fallback(self.model.pi, self.model.A, self.model.B, self.model.observations, obs_input); route = "fallback-used"
        inferred_states = [self.model.states[i] for i in inferred_idx]; real_states = self.model.state_sequence[1:1 + len(inferred_states)]
        figc, axc = plt.subplots(figsize=(8, 3))
        viz.plot_viterbi_comparison(axc, real_states, inferred_states, self.model.states)
        mismatches = sum(1 for a, b in zip(real_states, inferred_states) if a != b); total_v = len(inferred_states)
        acc = (total_v - mismatches) / total_v if total_v > 0 else 0.0
        diag = f"Conclusión: exactitud (hmmlearn) = {acc:.2%} ({mismatches}/{total_v} mismatches). Perturbación: {'Sí' if perturbed else 'No'}. Fuente: {source_note}"
        axc.text(0.01, -0.25, diag, transform=axc.transAxes, fontsize=8, va='top')
        figc.tight_layout()
        win = tk.Toplevel(self.root); win.title("Inferencia: Viterbi (hmmlearn)")
        canvasc = FigureCanvasTkAgg(figc, master=win); canvasc.get_tk_widget().pack(fill="both", expand=True); canvasc.draw()
        ttk.Label(win, text=f"Ruta usada por hmmlearn: {route}  |  Fuente: {source_note}  |  Backend: hmmlearn").pack(anchor="w", padx=6, pady=(4, 8))

    # ---------------- Compare backends (unchanged) ----------------
    def compare_backends_window(self):
        if not self.model.observation_sequence:
            messagebox.showwarning("Comparar backends", "Primero ejecuta la simulación."); return
        res = analysis_hmmlearn.compare_and_summary(self.model)
        win = tk.Toplevel(self.root); win.title("Comparativa backends: fallback vs hmmlearn"); win.geometry("1100x800")
        top_frame = ttk.Frame(win); top_frame.pack(side="top", fill="both", expand=True)
        bottom_frame = ttk.Frame(win, height=220); bottom_frame.pack(side="bottom", fill="x")
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        ax00 = axes[0, 0]; ax01 = axes[0, 1]; ax10 = axes[1, 0]; ax11 = axes[1, 1]
        fb_states = res['fallback']['states']; real_fb = self.model.state_sequence[1:1 + len(fb_states)]
        if fb_states and real_fb:
            viz.plot_viterbi_comparison(ax00, real_fb, fb_states, self.model.states)
            ax00.set_title(f"Fallback Viterbi (acc {res['fallback']['acc']:.2%})")
            ax00.text(0.01, -0.18, "Comparativa con mismatches marcados en rojo.", transform=ax00.transAxes, fontsize=8, va='top')
        else:
            ax00.text(0.5, 0.5, "No hay datos para fallback", ha='center', va='center'); ax00.set_axis_off()
        if res.get('hmmlearn'):
            hl_states = res['hmmlearn']['states']; real_hl = self.model.state_sequence[1:1 + len(hl_states)]
            if hl_states and real_hl:
                viz.plot_viterbi_comparison(ax01, real_hl, hl_states, self.model.states)
                route = res['hmmlearn'].get('route', 'unknown'); ax01.set_title(f"hmmlearn (acc {res['hmmlearn']['acc']:.2%}) route={route}")
                ax01.text(0.01, -0.18, "Comparativa con mismatches marcados en rojo.", transform=ax01.transAxes, fontsize=8, va='top')
            else:
                ax01.text(0.5, 0.5, "No hay datos para hmmlearn", ha='center', va='center'); ax01.set_axis_off()
        else:
            ax01.text(0.5, 0.5, res['note'], ha='center', va='center'); ax01.set_axis_off()
        if fb_states and real_fb:
            viz.plot_viterbi_lines(ax10, real_fb, fb_states, self.model.states); ax10.set_title("Evolución: real vs inferida (fallback) - sin mismatches")
            ax10.text(0.01, -0.18, "Evolución pura: trayectoria de estados real y la inferida.", transform=ax10.transAxes, fontsize=8, va='top')
        else:
            ax10.text(0.5, 0.5, "No hay datos para fallback (evolución)", ha='center', va='center'); ax10.set_axis_off()
        if res.get('hmmlearn') and res['hmmlearn']['states']:
            hl_states = res['hmmlearn']['states']; real_hl = self.model.state_sequence[1:1 + len(hl_states)]
            viz.plot_viterbi_lines(ax11, real_hl, hl_states, self.model.states); ax11.set_title("Evolución: real vs inferida (hmmlearn) - sin mismatches")
            ax11.text(0.01, -0.18, "Evolución pura: trayectoria de estados real y la inferida.", transform=ax11.transAxes, fontsize=8, va='top')
        else:
            ax11.text(0.5, 0.5, "No hay datos para hmmlearn (evolución)", ha='center', va='center'); ax11.set_axis_off()
        fig.tight_layout(); canvas = FigureCanvasTkAgg(fig, master=top_frame); canvas.get_tk_widget().pack(fill="both", expand=True); canvas.draw()
        txt = tk.Text(bottom_frame, height=8, wrap="word", font=("Segoe UI", 9)); txt.pack(fill="both", expand=True, padx=6, pady=6)
        txt.insert("1.0", "Resumen comparativa\n\n"); txt.insert("end", f"Fallback: acc={res['fallback']['acc']:.2%}, mismatches={res['fallback']['mismatches']}\n")
        if res['hmmlearn']:
            txt.insert("end", f"hmmlearn: acc={res['hmmlearn']['acc']:.2%}, mismatches={res['hmmlearn']['mismatches']}, route={res['hmmlearn'].get('route','?')}\n")
        else:
            txt.insert("end", f"hmmlearn: no disponible o falló. Nota: {res['note']}\n")
        txt.insert("end", f"\nPosiciones donde fallback != hmmlearn: {res['diff_positions']}\n\n")
        txt.insert("end", "Interpretación:\n- Las gráficas superiores muestran mismatches (puntos rojos) donde las secuencias difieren.\n- Las gráficas inferiores muestran solo la evolución de estados (sin marcar mismatches) para comparar tendencias.\n")
        txt.configure(state="disabled")

    # ---------------- Results window (grafo a la izquierda, evolución a la derecha; retorno y volatilidad intercambiados) ----------------
    def show_simulation_results(self):
        if not self.model.state_sequence:
            messagebox.showwarning("Resultados", "Primero ejecuta la simulación (Simular 50 o Start)."); return

        win = tk.Toplevel(self.root); win.title("Resultados de Simulación (final)"); win.geometry("1250x980")

        # Top: recommendations label (visible immediately)
        metrics = self.model.analyze()
        last_state = self.dom_state
        vol_by_state = metrics.get("volatilities", {}); current_vol = vol_by_state.get(last_state, None) if last_state else None
        total_return = metrics.get("total_return", 0.0); max_dd = metrics.get("max_drawdown", 0.0)
        if last_state == "Mercado Alcista":
            rec_text = "Recomendación: Mercado alcista — aumentar exposición moderadamente; favorecer posiciones largas con trailing stop."
        elif last_state == "Mercado Bajista":
            rec_text = "Recomendación: Mercado bajista — reducir exposición; priorizar coberturas y órdenes limitadas."
        elif last_state == "Mercado Lateral":
            rec_text = "Recomendación: Mercado lateral — estrategias de rango; evitar posiciones direccionales grandes."
        else:
            rec_text = "Recomendación: Estado no determinado — mantener tamaño de posición conservador hasta confirmar régimen."
        risk_line1 = f"Análisis de riesgo: Retorno acumulado {total_return:.2%}, Drawdown máximo {max_dd:.2%}."
        if current_vol is not None:
            risk_line2 = f"Volatilidad actual (régimen {last_state}): {current_vol:.4f}. Ajustar límites de posición según volatilidad."
        else:
            risk_line2 = "Volatilidad por régimen no disponible; usar límites de posición conservadores."
        final_text = f"{rec_text}  {risk_line1} {risk_line2}"
        lbl_top = tk.Label(win, text=final_text, justify="left", anchor="w", font=("Segoe UI", 10, "bold"), background="#f3f9f1", relief="ridge", bd=1, padx=10, pady=8)
        lbl_top.pack(fill="x", padx=12, pady=(10, 6))

        # Grid layout: left column = grafo (mediano), right column = evolución (grande)
        fig = plt.figure(figsize=(13, 9))
        gs = fig.add_gridspec(3, 3, width_ratios=[1.6, 2.4, 1.0], height_ratios=[0.9, 1.0, 2.0], hspace=0.45, wspace=0.35)

        ax_dist = fig.add_subplot(gs[0, 0])
        ax_vol = fig.add_subplot(gs[0, 1])    # <-- swapped: volatility now in center top
        ax_return = fig.add_subplot(gs[0, 2]) # <-- swapped: return now in right top

        ax_dd = fig.add_subplot(gs[1, 0])
        ax_heat = fig.add_subplot(gs[1, 1])
        ax_timestamps = fig.add_subplot(gs[1, 2])

        ax_colored = fig.add_subplot(gs[2, 0])       # grafo a la izquierda (vertical)
        ax_evol = fig.add_subplot(gs[2, 1:3])        # evolución a la derecha (más ancha)

        # 1) Duration / distribution
        viz.plot_state_distribution(ax_dist, self.model.state_sequence, self.model.states)
        avg_durs = metrics.get("avg_durations", {})
        ax_dist.text(0.01, -0.18, "Duración promedio: " + ", ".join([f"{s}: {avg_durs.get(s,0):.1f} períodos" for s in self.model.states]), transform=ax_dist.transAxes, fontsize=8, va='top')

        # 2) Volatilidad por régimen (ahora en la posición central superior)
        ax_vol.clear()
        vols = metrics.get("volatilities", {})
        vals = [vols.get(s, 0.0) for s in self.model.states]
        ax_vol.bar(self.model.states, vals, color=["#b7e4c7", "#fff3b0", "#ffb4a2"])
        ax_vol.set_title("Volatilidad por régimen")
        ax_vol.text(0.01, -0.18, "Conclusión: volatilidades por régimen.", transform=ax_vol.transAxes, fontsize=8, va='top')

        # 3) Retorno total acumulado (ahora en la esquina superior derecha, más pequeña)
        ax_return.clear()
        ax_return.bar([0], [total_return], color="#2b8cbe", width=0.35)
        ax_return.set_xticks([0]); ax_return.set_xticklabels(["Retorno total"])
        ax_return.set_title("Retorno total acumulado")
        ax_return.text(0.01, -0.18, f"Retorno acumulado = {total_return:.2%}.", transform=ax_return.transAxes, fontsize=8, va='top')

        # 4) Drawdown (small)
        ax_dd.clear(); ax_dd.bar([0], [max_dd], color="#de2d26", width=0.35); ax_dd.set_xticks([0]); ax_dd.set_xticklabels(["Max Drawdown"])
        ax_dd.set_title("Drawdown máximo"); ax_dd.text(0.01, -0.18, f"Drawdown máximo = {max_dd:.2%}.", transform=ax_dd.transAxes, fontsize=8, va='top')

        # 5) Heatmap
        emp_res = self.model.validate_empirical_transitions(); emp = emp_res["empirical"]
        df_emp = pd.DataFrame(emp, index=self.model.states, columns=self.model.states)
        try:
            import seaborn as sns_local
            sns_local.heatmap(df_emp, cmap="viridis", annot=True, fmt=".2f", ax=ax_heat)
        except Exception:
            ax_heat.imshow(emp, cmap="viridis", aspect="auto")
            ax_heat.set_xticks(range(len(self.model.states))); ax_heat.set_xticklabels(self.model.states, rotation=45)
            ax_heat.set_yticks(range(len(self.model.states))); ax_heat.set_yticklabels(self.model.states)
        ax_heat.set_title("Heatmap: transiciones empíricas")
        ax_heat.text(0.01, -0.18, f"Max diff vs teórico = {emp_res['max_diff']:.3f}", transform=ax_heat.transAxes, fontsize=8, va='top')

        # 6) Timestamps (clear markers)
        ax_timestamps.clear()
        seq = self.model.state_sequence[1:]
        change_idxs = []; change_states = []
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                change_idxs.append(i); change_states.append(seq[i])
        if change_idxs:
            ax_timestamps.vlines(change_idxs, ymin=0, ymax=1, color="#6a51a3", linewidth=2, alpha=0.95)
            ax_timestamps.scatter(change_idxs, [0.6] * len(change_idxs), color="#6a51a3", s=60)
            for xi, st in zip(change_idxs, change_states):
                ax_timestamps.text(xi, 0.78, st, rotation=45, ha='center', va='bottom', fontsize=8, color="#222222")
            ax_timestamps.set_xlim(0, max(1, len(seq))); ax_timestamps.set_ylim(0, 1); ax_timestamps.set_yticks([])
            ax_timestamps.set_xlabel("Período (t)"); ax_timestamps.set_title("Timestamps: cambios de régimen")
            ax_timestamps.text(0.01, -0.18, f"Total cambios: {len(change_idxs)}", transform=ax_timestamps.transAxes, fontsize=8, va='top')
        else:
            ax_timestamps.text(0.5, 0.5, "No se detectaron cambios de régimen", ha='center', va='center'); ax_timestamps.set_axis_off()

        # 7) Evolution (big, right) - usar la nueva función mejorada
        ax_e = ax_evol; ax_e.clear()
        seq_real = self.model.state_sequence[1:]
        try:
            self.plot_state_evolution_comparison(ax_e, self.model.A, seq_real, self.model.states)
        except Exception:
            mapping = {s: i for i, s in enumerate(self.model.states)}
            if seq_real:
                real_idx = [mapping.get(s, -1) for s in seq_real]; ax_e.plot(range(len(real_idx)), real_idx, drawstyle='steps-post', marker='o', color="#2b8cbe", label='Real')
            ax_e.set_yticks(range(len(self.model.states))); ax_e.set_yticklabels(self.model.states)
            ax_e.set_xlabel("Período (t)"); ax_e.set_title("Evolución: estados (real) vs inferida (fallback)")

        # 8) Colored graph (left) - colored by frequency and overlay edges (no textual blocks)
        ax_colored.clear()
        viz.draw_colored_state_graph(ax_colored, self.model.states, self.model.A, self.model.state_sequence)
        counts = {}; total_from = {}
        seq_full = self.model.state_sequence
        for i in range(len(seq_full) - 1):
            u, v = seq_full[i], seq_full[i + 1]; counts[(u, v)] = counts.get((u, v), 0) + 1; total_from[u] = total_from.get(u, 0) + 1
        empirical_edges = {}
        for (u, v), c in counts.items():
            empirical_edges[(u, v)] = c / total_from[u] if total_from.get(u, 0) > 0 else 0.0
        viz.draw_transition_graph(ax_colored, self.model.states, self.model.A, highlight_edges=None, show_empirical=empirical_edges)
        ax_colored.set_title("Grafo coloreado por frecuencia empírica")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win); canvas.get_tk_widget().pack(fill="both", expand=True); canvas.draw()

    # ---------------- Improved evolution plot helper ----------------
    def plot_state_evolution_comparison(self, ax, model_A, state_sequence, state_names):
        """
        Dibuja la evolución de la proporción acumulada de cada estado (simulado)
        y la probabilidad estacionaria teórica (línea discontinua).
        Leyenda colocada debajo de la gráfica para evitar solapamiento con el título.
        """
        ax.clear()
        if not state_sequence:
            ax.set_title("Evolución de estados (no hay datos)")
            return

        A = np.array(model_A, dtype=float)
        try:
            vals, vecs = np.linalg.eig(A.T)
            idx = np.argmin(np.abs(vals - 1.0))
            stat = np.real(vecs[:, idx])
            stat = stat / stat.sum()
        except Exception:
            stat = np.ones(len(state_names)) / len(state_names)

        seq = pd.Series(state_sequence)
        cum_props = []
        for t in range(1, len(seq) + 1):
            counts = seq.iloc[:t].value_counts().reindex(state_names).fillna(0)
            total = counts.sum()
            cum_props.append((counts / total).values if total > 0 else np.zeros(len(state_names)))
        cum_props = np.array(cum_props)
        x = np.arange(1, len(seq) + 1)

        colors = ["#2ca02c", "#ffbf00", "#de2d26"]
        if len(state_names) > len(colors):
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i) for i in range(len(state_names))]

        # Área apilada ligera
        try:
            ax.stackplot(x, cum_props.T, colors=colors, alpha=0.18)
        except Exception:
            bottom = np.zeros_like(x, dtype=float)
            for i in range(len(state_names)):
                ax.fill_between(x, bottom, bottom + cum_props[:, i], color=colors[i], alpha=0.12)
                bottom += cum_props[:, i]

        # Líneas simuladas
        for i, s in enumerate(state_names):
            ax.plot(x, cum_props[:, i], label=f"Simulado: {s}", color=colors[i], linewidth=1.6, marker='o', markersize=4, alpha=0.95)

        # Líneas estacionarias teóricas
        for i, s in enumerate(state_names):
            ax.hlines(stat[i], xmin=1, xmax=len(seq), colors=colors[i], linestyles='--', alpha=0.9, label=f"Teórico: {s}")

        # Ajustes estéticos y leyenda **debajo** de la gráfica
        ax.set_xlabel("Período (t)")
        ax.set_ylabel("Prop. acumulada")
        ax.set_title("Evolución: Propor. simulada vs prob. estacionaria teórica")
        # Colocar la leyenda fuera y debajo de la gráfica
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.22)
        ax.set_xlim(1, max(1, len(seq)))
        if len(x) > 40:
            ax.xaxis.set_major_locator(plt.MaxNLocator(8))
        # Ajuste de layout para que la leyenda no tape nada
        try:
            ax.figure.tight_layout(rect=[0, 0.06, 1, 0.95])
        except Exception:
            pass

    # ---------------- Histogram + Theoretical distribution (replaced by improved evolution) ----------------
    def show_updown_and_theoretical_distribution(self):
        """
        Ventana resumen: histograma (izq) + evolución mejorada (der).
        La conclusión se genera a partir de las métricas actuales de la simulación (self.model.analyze()).
        """
        if not getattr(self.model, "price_series", None) or len(self.model.price_series) < 2:
            return

        prices = np.asarray(self.model.price_series)
        returns = np.diff(prices) / prices[:-1]
        up_returns = returns[returns > 0]
        down_returns = returns[returns < 0]

        win = tk.Toplevel(self.root)
        win.title("Histograma y Evolución de Estados (simulado vs teórico)")
        win.geometry("1000x520")

        fig, (ax_hist, ax_evol) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [1, 1]})

        # Histograma
        bins = 20
        if up_returns.size > 0:
            ax_hist.hist(up_returns, bins=bins, color="#2ca02c", alpha=0.7, label="Alcista (retornos > 0)")
        if down_returns.size > 0:
            ax_hist.hist(down_returns, bins=bins, color="#d62728", alpha=0.7, label="Bajista (retornos < 0)")
        ax_hist.set_title("Histograma: Tendencias alcistas y bajistas")
        ax_hist.set_xlabel("Retorno por período")
        ax_hist.set_ylabel("Frecuencia")
        ax_hist.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False)
        ax_hist.grid(alpha=0.25)

        # Evolución mejorada (usa la función que ya coloca la leyenda debajo)
        seq_real = self.model.state_sequence[1:]
        try:
            self.plot_state_evolution_comparison(ax_evol, self.model.A, seq_real, self.model.states)
        except Exception:
            ax_evol.text(0.5, 0.5, "No se pudo generar la evolución", ha='center', va='center')
            ax_evol.set_axis_off()

        fig.subplots_adjust(hspace=0.25, wspace=0.35)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

        # --- Generar conclusión DINÁMICA basada en métricas reales de la simulación ---
        metrics = self.model.analyze() or {}
        total_return = metrics.get("total_return", 0.0)
        max_dd = metrics.get("max_drawdown", 0.0)
        vols = metrics.get("volatilities", {})
        avg_durs = metrics.get("avg_durations", {})

        n_up = int((returns > 0).sum())
        n_down = int((returns < 0).sum())
        avg_up = float(up_returns.mean()) if up_returns.size > 0 else 0.0
        avg_down = float(down_returns.mean()) if down_returns.size > 0 else 0.0
        pct_up = 100.0 * n_up / (n_up + n_down) if (n_up + n_down) > 0 else 0.0

        # Prob. estacionaria teórica (vector estacionario)
        # --- Prob. estacionaria teórica (vector estacionario) ---
        try:
            A = np.asarray(self.model.A, dtype=float)

            # Normalizar filas (cada fila debe sumar 1)
            row_sums = A.sum(axis=1)
            for i in range(len(row_sums)):
                if row_sums[i] > 0:
                    A[i, :] = A[i, :] / row_sums[i]
                else:
                    # Si la fila está en ceros (ej. Lateral sin transiciones),
                    # repartir uniformemente entre todos los estados
                    A[i, :] = np.ones(A.shape[1]) / A.shape[1]

            # Calcular distribución estacionaria
            vals, vecs = np.linalg.eig(A.T)
            idx = np.argmin(np.abs(vals - 1.0))
            stat = np.real(vecs[:, idx])
            stat = stat / stat.sum()

        except Exception:
            # Si falla, usar distribución uniforme
            stat = np.ones(len(self.model.states)) / len(self.model.states)

        # Selección del estado dominante
        idx_dom = int(np.argmax(stat))
        dom_state = self.model.states[idx_dom]
        dom_prob = stat[idx_dom]
        self.dom_state=dom_state

        # Ajustar con métricas empíricas
        if total_return < 0 and dom_state == "Mercado Alcista":
            dom_state = "Mercado Bajista"
        elif pct_up < 40 and dom_state == "Mercado Alcista":
            dom_state = "Mercado Bajista"



        # Construir conclusión basada en métricas
        lines = [
            f"Períodos con retorno positivo: {n_up}",
            f"Períodos con retorno negativo: {n_down}",
            f"% períodos alcistas: {pct_up:.1f}%",
            f"Retorno medio alcista: {avg_up:.4f}",
            f"Retorno medio bajista: {avg_down:.4f}",
            f"Retorno total acumulado (simulación): {total_return:.2%}",
            f"Drawdown máximo (simulación): {max_dd:.2%}",
            f"Prob. estacionaria teórica dominante: {dom_state} ({dom_prob:.2%})"
        ]

        # Interpretación automática (reglas simples basadas en métricas)
        if pct_up > 60 and avg_up > abs(avg_down) and dom_state == "Mercado Alcista":
            interp = "Interpretación: predominio alcista observado; la prob. estacionaria también sugiere persistencia alcista."
        elif pct_up < 40 and abs(avg_down) > avg_up and dom_state == "Mercado Bajista":
            interp = "Interpretación: predominio bajista observado; la prob. estacionaria confirma riesgo de continuidad bajista."
        else:
            # usar volatilidad y drawdown para matizar
            high_vol = any(v > 0.03 for v in vols.values())  # umbral ejemplo
            if high_vol or max_dd > 0.15:
                interp = "Interpretación: comportamiento mixto con riesgo; aplicar gestión activa y límites de pérdida."
            else:
                interp = "Interpretación: comportamiento mixto; condiciones relativamente estables, ajustar exposición según objetivos."

        final_text = "\n".join(lines + ["", interp])

        # Label sencillo debajo de la figura (dinámico)
        lbl = tk.Label(win, text=final_text, justify="left", anchor="w",
                       font=("Segoe UI", 10), background="#f7f7f7", relief="flat", padx=8, pady=6)
        lbl.pack(fill="x", padx=12, pady=(6, 12))

    # ---------------- Export, convergence (unchanged) ----------------
    def export_csv(self):

        messagebox.showinfo("Exportar", "Funcionalidad de exportación deshabilitada en esta versión, pendiente por mejorar construccion.")
        return


    def run_many_simulations_dialog(self):
        dlg = tk.Toplevel(self.root); dlg.title("Simular muchas - parámetros")
        ttk.Label(dlg, text="Tolerancia (max_diff) ej: 0.02").grid(row=0, column=0, padx=6, pady=6)
        tol_var = tk.DoubleVar(value=0.02); ttk.Entry(dlg, textvariable=tol_var).grid(row=0, column=1, padx=6, pady=6)
        ttk.Label(dlg, text="Max runs (ej:200)").grid(row=1, column=0, padx=6, pady=6); runs_var = tk.IntVar(value=200); ttk.Entry(dlg, textvariable=runs_var).grid(row=1, column=1, padx=6, pady=6)
        ttk.Label(dlg, text="Periods por run (ej:200)").grid(row=2, column=0, padx=6, pady=6); periods_var = tk.IntVar(value=200); ttk.Entry(dlg, textvariable=periods_var).grid(row=2, column=1, padx=6, pady=6)
        def start_runs(): dlg.destroy(); self.run_many_simulations(tol=tol_var.get(), max_runs=runs_var.get(), periods=periods_var.get())
        ttk.Button(dlg, text="Iniciar", command=start_runs).grid(row=3, column=0, columnspan=2, pady=8)

    def run_many_simulations(self, tol=0.02, max_runs=200, periods=200):
        history = []
        for i in range(1, max_runs + 1):
            self.model.simulate(periods=periods, initial_price=100.0, pre_generate=True)
            res = self.model.validate_empirical_transitions(); max_diff = res["max_diff"]; history.append(max_diff)
            self.txt_log.insert(tk.END, f"Run {i}: max_diff={max_diff:.6f}\n"); self.txt_log.see(tk.END); self.root.update()
            if max_diff <= tol: break
        self._convergence_history = history; self._update_all_views()
        # Mostrar ventana resumen al terminar las múltiples simulaciones
        try:
            self.show_updown_and_theoretical_distribution()
        except Exception:
            pass
        try:
            win = tk.Toplevel(self.root); win.title("Convergencia - max_diff por run"); fig, ax = plt.subplots(figsize=(8, 4))
            viz.plot_convergence_curve(ax, history, tol); canvas = FigureCanvasTkAgg(fig, master=win); canvas.get_tk_widget().pack(fill="both", expand=True); canvas.draw()
        except Exception as e:
            self.txt_log.insert(tk.END, f"[error] al mostrar convergencia: {e}\n"); self.txt_log.see(tk.END)

    def _update_all_views(self):
        self._refresh_log_text(); self._refresh_states_obs_table(); self._update_plots(highlight_index=None)

# ---------------- Main ----------------
def main():
    root = tk.Tk(); app = HMMApp(root); root.mainloop()

if __name__ == "__main__":
    main()
