import datetime
import os
import secrets
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/codex-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_protein")
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from qiskit import transpile
from qiskit.circuit.library import TwoLocal
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2

from CDQA_simul import build_cd_only_ansatz
from hamiltonian_Aware_VQE import (
    build_interaction_aware_ansatz,
    folding_hamiltonian_vectorized,
    get_strong_interaction_pairs,
    parse_hamiltonian_robust,
)
from load_mj import load_interaction_energy

# "YGGFM", "KLVFFA", "APRLPFY", "AVDINNNA", 
TARGET_PROTEINS = ["YGGFM", "KLVFFA", "APRLPFY", "AVDINNNA", "CYIQNCPLG", "GYDPETGTWG", "RPKPQQFFGLM"]
OPTIMAL_REPS = 2
OPTIMAL_SHOTS = 2048
MAX_ITER = 3000
NUM_RESTARTS = 10
RANDOM_SEED = secrets.randbits(32)
BACKEND = AerSimulator(max_parallel_threads=1, max_parallel_experiments=1)

ANSATZ_CONFIGS = {
    "Hamiltonian-Aware": {
        "color": "#1f77b4",
        "builder": lambda num_qubits, hamiltonian: build_interaction_aware_ansatz(
            num_qubits,
            get_strong_interaction_pairs(
                hamiltonian, num_qubits, top_k=min(15, num_qubits * 2)
            ),
            reps=OPTIMAL_REPS,
        ),
    },
    "TwoLocal": {
        "color": "#2ca02c",
        "builder": lambda num_qubits, _hamiltonian: TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks="ry",
            entanglement_blocks="cz",
            entanglement="linear",
            reps=OPTIMAL_REPS,
            insert_barriers=True,
        ).decompose(),
    },
    "DC-QAOA": {
        "color": "#d62728",
        "builder": lambda num_qubits, _hamiltonian: build_cd_only_ansatz(
            num_qubits, reps=OPTIMAL_REPS
        ),
    },
}


def aggregate_histories(histories):
    max_len = max(len(history) for history in histories)
    history_array = np.full((len(histories), max_len), np.nan)

    for idx, history in enumerate(histories):
        if not history:
            continue
        history_array[idx, : len(history)] = history
        history_array[idx, len(history) :] = history[-1]

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


def build_problem(target_protein):
    interaction_energy = load_interaction_energy("mj_potential.csv", target_protein)
    model, _ = folding_hamiltonian_vectorized(target_protein, interaction_energy)
    hamiltonian, real_qubits = parse_hamiltonian_robust(str(model.obj.expr))
    return hamiltonian, real_qubits


def run_single_ansatz(target_protein, hamiltonian, real_qubits, ansatz_name, config, protein_seed):
    ansatz = config["builder"](real_qubits, hamiltonian)
    ansatz_transpiled = transpile(ansatz, BACKEND, optimization_level=1)
    estimator = EstimatorV2()
    estimator.options.default_shots = OPTIMAL_SHOTS

    histories = []
    final_energies = []
    evaluation_counts = []
    iteration_counts = []
    run_times = []

    print(
        f"👉 [{target_protein}] {ansatz_name} | reps={OPTIMAL_REPS}, "
        f"qubits={real_qubits}, params={ansatz_transpiled.num_parameters}, depth={ansatz_transpiled.depth()}"
    )

    for restart_idx in range(NUM_RESTARTS):
        rng = np.random.default_rng(protein_seed + restart_idx)
        initial_params = rng.uniform(0.0, 2.0 * np.pi, ansatz_transpiled.num_parameters)
        current_history = []
        run_start = time.time()

        def cost_func(params):
            pub = (ansatz_transpiled, [hamiltonian], [params])
            job = estimator.run([pub])
            energy = float(job.result()[0].data.evs[0])
            current_history.append(energy)
            return energy

        res = minimize(
            cost_func,
            initial_params,
            method="COBYLA",
            options={"maxiter": MAX_ITER, "tol": 0.01},
        )

        final_energy = float(res.fun if hasattr(res, "fun") else np.min(current_history))
        nfev = int(getattr(res, "nfev", len(current_history)))
        nit = getattr(res, "nit", np.nan)
        elapsed = time.time() - run_start

        histories.append(current_history)
        final_energies.append(final_energy)
        evaluation_counts.append(nfev)
        iteration_counts.append(np.nan if isinstance(nit, str) else nit)
        run_times.append(elapsed)

        print(
            f"     [Run {restart_idx + 1}] Energy: {final_energy:.4f} | "
            f"Eval: {nfev} | Time: {elapsed:.2f}s"
        )

    mean_history, std_history = aggregate_histories(histories)

    return {
        "Target Protein": target_protein,
        "Protein Length": len(target_protein),
        "Qubits": real_qubits,
        "Hamiltonian Terms": len(hamiltonian),
        "Ansatz": ansatz_name,
        "Reps": OPTIMAL_REPS,
        "Parameters": ansatz_transpiled.num_parameters,
        "Depth": ansatz_transpiled.depth(),
        "Shots": OPTIMAL_SHOTS,
        "Max Iter": MAX_ITER,
        "Restarts": NUM_RESTARTS,
        "Mean Final Energy": float(np.mean(final_energies)),
        "Std Final Energy": float(np.std(final_energies)),
        "Best Final Energy": float(np.min(final_energies)),
        "Mean Evaluations (nfev)": float(np.mean(evaluation_counts)),
        "Std Evaluations (nfev)": float(np.std(evaluation_counts)),
        "Mean Steps (nit)": safe_nanmean(iteration_counts),
        "Std Steps (nit)": safe_nanstd(iteration_counts),
        "Mean Time (s)": float(np.mean(run_times)),
        "Std Time (s)": float(np.std(run_times)),
        "Mean History": mean_history,
        "Std History": std_history,
        "All Histories": histories,
        "Color": config["color"],
    }


def plot_convergence_by_protein(results, save_dir, timestamp):
    fig, axes = plt.subplots(1, len(TARGET_PROTEINS), figsize=(5 * len(TARGET_PROTEINS), 5), sharey=True)

    if len(TARGET_PROTEINS) == 1:
        axes = [axes]

    for ax, target_protein in zip(axes, TARGET_PROTEINS):
        protein_results = [
            result for result in results if result["Target Protein"] == target_protein
        ]
        protein_results.sort(key=lambda item: item["Ansatz"])

        for result in protein_results:
            x = np.arange(len(result["Mean History"]))
            lower = result["Mean History"] - result["Std History"]
            upper = result["Mean History"] + result["Std History"]

            ax.plot(
                x,
                result["Mean History"],
                label=result["Ansatz"],
                color=result["Color"],
                linewidth=2.0,
            )
            ax.fill_between(x, lower, upper, color=result["Color"], alpha=0.15)

        qubits = protein_results[0]["Qubits"] if protein_results else "N/A"
        ax.set_title(f"{target_protein} ({qubits} qubits)")
        ax.set_xlabel("Evaluation")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Energy")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"Ansatz Convergence by Problem Size (reps={OPTIMAL_REPS}, shots={OPTIMAL_SHOTS})",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = os.path.join(save_dir, f"ansatz_problem_size_optimal_convergence_{timestamp}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_scaling_summary(df_summary, save_dir, timestamp):
    metrics = [
        ("Mean Final Energy", "Mean Final Energy"),
        ("Parameters", "Number of Parameters"),
        ("Depth", "Transpiled Circuit Depth"),
        ("Mean Time (s)", "Mean Runtime (s)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (column, ylabel) in zip(axes, metrics):
        for ansatz_name, config in ANSATZ_CONFIGS.items():
            subset = df_summary[df_summary["Ansatz"] == ansatz_name].sort_values("Protein Length")
            ax.plot(
                subset["Protein Length"],
                subset[column],
                marker="o",
                linewidth=2.0,
                color=config["color"],
                label=ansatz_name,
            )
        ax.set_xlabel("Protein Length")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(
        f"Optimal-Setting Ansatz Scaling (reps={OPTIMAL_REPS}, shots={OPTIMAL_SHOTS})",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = os.path.join(save_dir, f"ansatz_problem_size_optimal_scaling_{timestamp}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_outputs(results):
    save_dir = "ansatz_problem_size_optimal_result"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = os.path.join(save_dir, f"ansatz_problem_size_optimal_summary_{timestamp}.txt")
    csv_filename = os.path.join(save_dir, f"ansatz_problem_size_optimal_summary_{timestamp}.csv")
    history_csv_filename = os.path.join(
        save_dir, f"ansatz_problem_size_optimal_histories_{timestamp}.csv"
    )

    df_summary = pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key not in {"Mean History", "Std History", "All Histories", "Color"}
            }
            for result in results
        ]
    ).sort_values(["Protein Length", "Target Protein", "Ansatz"]).reset_index(drop=True)

    best_by_protein = (
        df_summary.loc[df_summary.groupby("Target Protein")["Mean Final Energy"].idxmin()]
        .sort_values(["Protein Length", "Target Protein"])
        .reset_index(drop=True)
    )

    history_columns = {}
    for result in results:
        base = f"{result['Target Protein']} | {result['Ansatz']}"
        history_columns[f"{base} Mean"] = pd.Series(result["Mean History"])
        history_columns[f"{base} Std"] = pd.Series(result["Std History"])
        for run_idx, history in enumerate(result["All Histories"], start=1):
            padded_history = history + [history[-1]] * (len(result["Mean History"]) - len(history))
            history_columns[f"{base} Run {run_idx}"] = pd.Series(padded_history)

    df_summary.to_csv(csv_filename, index=False)
    pd.DataFrame(history_columns).to_csv(history_csv_filename, index_label="Evaluation")

    with open(txt_filename, "w", encoding="utf-8") as file:
        file.write("=== Optimal-Setting Ansatz Comparison by Problem Size ===\n")
        file.write(f"Date: {timestamp}\n")
        file.write(f"Target proteins: {', '.join(TARGET_PROTEINS)}\n")
        file.write(
            f"Reps: {OPTIMAL_REPS} | Shots: {OPTIMAL_SHOTS} | Max Iter: {MAX_ITER} | "
            f"Restarts: {NUM_RESTARTS} | Seed: {RANDOM_SEED}\n\n"
        )
        file.write("[Full Summary]\n")
        file.write(df_summary.to_string(index=False))
        file.write("\n\n[Best Ansatz per Protein]\n")
        file.write(best_by_protein.to_string(index=False))

    convergence_plot = plot_convergence_by_protein(results, save_dir, timestamp)
    scaling_plot = plot_scaling_summary(df_summary, save_dir, timestamp)

    print("\n" + "=" * 100)
    print(df_summary.to_string(index=False))
    print("=" * 100)
    print("\n[Best Ansatz per Protein]")
    print(best_by_protein.to_string(index=False))
    print(f"\n📄 Summary saved to: {txt_filename}")
    print(f"💾 Summary CSV saved to: {csv_filename}")
    print(f"💾 Histories CSV saved to: {history_csv_filename}")
    print(f"🖼️ Convergence plot saved to: {convergence_plot}")
    print(f"🖼️ Scaling plot saved to: {scaling_plot}")


def run_problem_size_benchmark():
    print("🚀 Starting Optimal-Setting Ansatz Comparison Across Problem Sizes")
    print(f"   Target proteins: {TARGET_PROTEINS}")
    print(
        f"   Reps={OPTIMAL_REPS}, Shots={OPTIMAL_SHOTS}, "
        f"MaxIter={MAX_ITER}, Restarts={NUM_RESTARTS}\n"
    )

    results = []

    for protein_idx, target_protein in enumerate(TARGET_PROTEINS):
        print(f"\n{'=' * 100}")
        print(f"🧬 Target Protein: {target_protein}")
        print(f"{'=' * 100}")

        hamiltonian, real_qubits = build_problem(target_protein)
        print(f"   -> Qubits: {real_qubits}, Hamiltonian Terms: {len(hamiltonian)}")

        for ansatz_idx, (ansatz_name, config) in enumerate(ANSATZ_CONFIGS.items()):
            protein_seed = RANDOM_SEED + protein_idx * 1000 + ansatz_idx * 100
            result = run_single_ansatz(
                target_protein,
                hamiltonian,
                real_qubits,
                ansatz_name,
                config,
                protein_seed,
            )
            results.append(result)

    save_outputs(results)


if __name__ == "__main__":
    run_problem_size_benchmark()
