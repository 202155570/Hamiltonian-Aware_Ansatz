import time
import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit_algorithms.optimizers import SPSA, ADAM

from load_mj import load_interaction_energy
from hamiltonian_Aware_VQE import (
    folding_hamiltonian_vectorized,
    parse_hamiltonian_robust,
    get_strong_interaction_pairs,
    build_interaction_aware_ansatz,
)

# ==========================================
# 1. 실험 설정
# ==========================================
TARGET_PROTEIN = "KLVFFA"
LAYER_RANGE = range(1, 6)
SHOTS = 4096
MAX_ITER = 1000
NUM_RESTARTS = 10
BACKEND = AerSimulator()

OPTIMIZERS_TO_TEST = {
    "COBYLA": {"color": "blue", "tol": 0.1},
    "Nelder-Mead": {"color": "green", "tol": 0.1},
    "SLSQP": {"color": "red", "tol": 0.1},
    "SPSA": {"color": "orange", "tol": 0.1},
    "Adam": {"color": "brown", "tol": 0.1},
}


def aggregate_histories(histories):
    max_len = max(len(history) for history in histories)
    history_array = np.full((len(histories), max_len), np.nan)

    for idx, history in enumerate(histories):
        history_array[idx, :len(history)] = history

    return np.nanmean(history_array, axis=0), np.nanstd(history_array, axis=0)


def safe_nanmean(values):
    values = np.asarray(values, dtype=float)
    if np.isnan(values).all():
        return np.nan
    return float(np.nanmean(values))


def safe_nanstd(values):
    values = np.asarray(values, dtype=float)
    if np.isnan(values).all():
        return np.nan
    return float(np.nanstd(values))


def run_layer_sweep_benchmark():
    print("🚀 Starting Layer Sweep Benchmark by Optimizer")
    print(f"   Target: {TARGET_PROTEIN} | Layers: {list(LAYER_RANGE)}")
    print(f"   Random Starts per Setting: {NUM_RESTARTS}\n")

    interaction_energy = load_interaction_energy("mj_potential.csv", TARGET_PROTEIN)
    model, _ = folding_hamiltonian_vectorized(TARGET_PROTEIN, interaction_energy)
    hamiltonian, real_qubits = parse_hamiltonian_robust(str(model.obj.expr))

    top_k = min(15, real_qubits * 2)
    strong_pairs = get_strong_interaction_pairs(hamiltonian, real_qubits, top_k=top_k)

    results = []
    history_columns = {}

    estimator = EstimatorV2()
    estimator.options.default_shots = SHOTS

    def run_strategy(optimizer_name, options, layer, ansatz_transpiled):
        print(
            f"👉 Testing [{optimizer_name}] at Layer={layer} "
            f"({NUM_RESTARTS} random starts, averaging results)..."
        )
        start_time = time.time()

        num_params = ansatz_transpiled.num_parameters
        histories = []
        final_energies = []
        evaluation_counts = []
        iteration_counts = []
        run_times = []

        for restart_idx in range(NUM_RESTARTS):
            current_history = []
            run_start_time = time.time()

            def cost_func(params):
                pub = (ansatz_transpiled, [hamiltonian], [params])
                job = estimator.run([pub])
                energy = float(job.result()[0].data.evs[0])
                current_history.append(energy)
                return energy

            initial_params = np.random.uniform(0, 2 * np.pi, num_params)

            if optimizer_name == "SPSA":
                opt = SPSA(maxiter=MAX_ITER)
                res = opt.minimize(cost_func, initial_params)
            elif optimizer_name == "Adam":
                opt = ADAM(maxiter=MAX_ITER, lr=0.1, tol=options["tol"])
                res = opt.minimize(cost_func, initial_params)
            elif optimizer_name == "COBYLA":
                res = minimize(
                    cost_func,
                    initial_params,
                    method="COBYLA",
                    tol=options["tol"],
                    options={"maxiter": MAX_ITER},
                )
            elif optimizer_name == "Nelder-Mead":
                res = minimize(
                    cost_func,
                    initial_params,
                    method="Nelder-Mead",
                    options={
                        "maxiter": MAX_ITER,
                        "xatol": options["tol"],
                        "fatol": options["tol"],
                    },
                )
            elif optimizer_name == "SLSQP":
                res = minimize(
                    cost_func,
                    initial_params,
                    method="SLSQP",
                    options={"maxiter": MAX_ITER, "ftol": options["tol"]},
                )
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")

            res_fun = float(res.fun if hasattr(res, "fun") else np.min(current_history))
            # res_nfev = int(res.nfev) if hasattr(res, "nfev") else len(current_history)
            res_nfev = len(current_history)
            res_nit = getattr(res, "nit", np.nan)
            run_elapsed = time.time() - run_start_time

            histories.append(current_history)
            final_energies.append(res_fun)
            evaluation_counts.append(res_nfev)
            iteration_counts.append(np.nan if isinstance(res_nit, str) else res_nit)
            run_times.append(run_elapsed)

            print(
                f"     [Run {restart_idx + 1}] Energy: {res_fun:.4f} | "
                f"Eval(nfev): {res_nfev} | Time: {run_elapsed:.2f}s"
            )

        elapsed_time = time.time() - start_time
        mean_history, std_history = aggregate_histories(histories)

        return {
            "Optimizer": optimizer_name,
            "Layer": layer,
            "Num Parameters": num_params,
            "Mean Final Energy": float(np.mean(final_energies)),
            "Std Final Energy": float(np.std(final_energies)),
            "Mean Steps (nit)": safe_nanmean(iteration_counts),
            "Std Steps (nit)": safe_nanstd(iteration_counts),
            "Mean Evaluations (nfev)": float(np.mean(evaluation_counts)),
            "Std Evaluations (nfev)": float(np.std(evaluation_counts)),
            "Mean Time (s)": round(float(np.mean(run_times)), 2),
            "Std Time (s)": round(float(np.std(run_times)), 2),
            "Total Time (s)": round(elapsed_time, 2),
            "Mean History": mean_history,
            "Std History": std_history,
            "All Histories": histories,
            "Color": options["color"],
        }

    for layer in LAYER_RANGE:
        print(f"\n{'=' * 80}")
        print(f"🔬 Layer Sweep: reps={layer}")
        print(f"{'=' * 80}")

        ansatz = build_interaction_aware_ansatz(real_qubits, strong_pairs, reps=layer)
        ansatz_transpiled = transpile(ansatz, BACKEND)

        for optimizer_name, options in OPTIMIZERS_TO_TEST.items():
            res = run_strategy(optimizer_name, options, layer, ansatz_transpiled)
            results.append(res)

            base_name = f"{optimizer_name} Layer {layer}"
            history_columns[f"{base_name} Mean"] = pd.Series(res["Mean History"])
            history_columns[f"{base_name} Std"] = pd.Series(res["Std History"])
            for run_idx, history in enumerate(res["All Histories"], start=1):
                history_columns[f"{base_name} Run {run_idx}"] = pd.Series(history)

    save_dir = "optimizer_layer_result"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = os.path.join(save_dir, f"optimizer_layer_summary_{TARGET_PROTEIN}_{timestamp}.txt")
    csv_filename = os.path.join(save_dir, f"optimizer_layer_history_{TARGET_PROTEIN}_{timestamp}.csv")
    energy_plot_filename = os.path.join(
        save_dir, f"optimizer_layer_energy_{TARGET_PROTEIN}_{timestamp}.png"
    )
    evaluation_plot_filename = os.path.join(
        save_dir, f"optimizer_layer_evaluations_{TARGET_PROTEIN}_{timestamp}.png"
    )
    time_plot_filename = os.path.join(
        save_dir, f"optimizer_layer_runtime_{TARGET_PROTEIN}_{timestamp}.png"
    )

    df_summary = pd.DataFrame(
        [
            {k: v for k, v in result.items() if k not in {"Mean History", "Std History", "All Histories", "Color"}}
            for result in results
        ]
    ).sort_values(["Optimizer", "Layer"]).reset_index(drop=True)

    best_idx = df_summary.groupby("Optimizer")["Mean Final Energy"].idxmin()
    df_best = df_summary.loc[best_idx].sort_values("Optimizer").reset_index(drop=True)
    overall_best = df_summary.loc[df_summary["Mean Final Energy"].idxmin()]

    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("=== VQE Optimizer Layer Sweep Summary ===\n")
        f.write(f"Target: {TARGET_PROTEIN} | Date: {timestamp}\n")
        f.write(f"Layers Tested: {list(LAYER_RANGE)}\n")
        f.write(f"Random Starts per Setting: {NUM_RESTARTS}\n\n")
        f.write("[Full Summary]\n")
        f.write(df_summary.to_string(index=False))
        f.write("\n\n[Best Layer per Optimizer]\n")
        f.write(df_best.to_string(index=False))
        f.write("\n\n[Overall Best Setting]\n")
        f.write(overall_best.to_string())

    pd.DataFrame(history_columns).to_csv(csv_filename, index_label="Evaluation")

    print("\n" + "=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90)
    print("\n[Best Layer per Optimizer]")
    print(df_best.to_string(index=False))
    print("\n[Overall Best Setting]")
    print(overall_best.to_string())

    fig1, ax1 = plt.subplots(figsize=(12, 7))
    for optimizer_name, options in OPTIMIZERS_TO_TEST.items():
        subset = df_summary[df_summary["Optimizer"] == optimizer_name].sort_values("Layer")
        ax1.errorbar(
            subset["Layer"],
            subset["Mean Final Energy"],
            yerr=subset["Std Final Energy"],
            marker="o",
            linewidth=2,
            capsize=4,
            color=options["color"],
            label=optimizer_name,
        )
    ax1.set_title("Mean Final Energy vs Layer", fontsize=16, fontweight="bold")
    ax1.set_xlabel("Layer (reps)")
    ax1.set_ylabel("Mean Final Energy")
    ax1.set_xticks(list(LAYER_RANGE))
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(energy_plot_filename, dpi=300, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    for optimizer_name, options in OPTIMIZERS_TO_TEST.items():
        subset = df_summary[df_summary["Optimizer"] == optimizer_name].sort_values("Layer")
        ax2.errorbar(
            subset["Layer"],
            subset["Mean Evaluations (nfev)"],
            yerr=subset["Std Evaluations (nfev)"],
            marker="o",
            linewidth=2,
            capsize=4,
            color=options["color"],
            label=optimizer_name,
        )
    ax2.set_title("Function Evaluations vs Layer", fontsize=16, fontweight="bold")
    ax2.set_xlabel("Layer (reps)")
    ax2.set_ylabel("Mean Evaluations (nfev)")
    ax2.set_xticks(list(LAYER_RANGE))
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(evaluation_plot_filename, dpi=300, bbox_inches="tight")

    fig3, ax3 = plt.subplots(figsize=(12, 7))
    for optimizer_name, options in OPTIMIZERS_TO_TEST.items():
        subset = df_summary[df_summary["Optimizer"] == optimizer_name].sort_values("Layer")
        ax3.errorbar(
            subset["Layer"],
            subset["Mean Time (s)"],
            yerr=subset["Std Time (s)"],
            marker="o",
            linewidth=2,
            capsize=4,
            color=options["color"],
            label=optimizer_name,
        )
    ax3.set_title("Runtime vs Layer", fontsize=16, fontweight="bold")
    ax3.set_xlabel("Layer (reps)")
    ax3.set_ylabel("Mean Time (seconds)")
    ax3.set_xticks(list(LAYER_RANGE))
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(time_plot_filename, dpi=300, bbox_inches="tight")

    print(f"\n📄 Detailed report saved to: {txt_filename}")
    print(f"💾 Layer sweep history saved to: {csv_filename}")
    print(f"🖼️ Energy-vs-layer plot saved to: {energy_plot_filename}")
    print(f"🖼️ Evaluation-vs-layer plot saved to: {evaluation_plot_filename}")
    print(f"🖼️ Runtime-vs-layer plot saved to: {time_plot_filename}")

    plt.show()


if __name__ == "__main__":
    run_layer_sweep_benchmark()
