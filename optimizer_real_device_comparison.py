import datetime
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_protein")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from qiskit import transpile
from qiskit_algorithms.optimizers import ADAM, SPSA
from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService, Session

from load_mj import load_interaction_energy
from hamiltonian_Aware_VQE import (
    build_interaction_aware_ansatz,
    folding_hamiltonian_vectorized,
    get_strong_interaction_pairs,
    parse_hamiltonian_robust,
)


TARGET_PROTEIN = "KLVFFA"
REPS = 4
SHOTS = 2048
MAX_ITER = 25
NUM_RESTARTS = 1
RANDOM_SEED = 1234
OPTIMIZATION_LEVEL = 1
RESILIENCE_LEVEL = 1

IBM_CHANNEL = os.getenv("IBM_RUNTIME_CHANNEL", "ibm_quantum_platform")
IBM_INSTANCE = os.getenv("IBM_RUNTIME_INSTANCE")
IBM_BACKEND_NAME = os.getenv("IBM_BACKEND_NAME")

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
        history_array[idx, : len(history)] = history

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


def initialize_service():
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if token:
        kwargs = {"channel": IBM_CHANNEL, "token": token}
    else:
        kwargs = {"channel": IBM_CHANNEL}

    if IBM_INSTANCE:
        kwargs["instance"] = IBM_INSTANCE

    try:
        return QiskitRuntimeService(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            "IBM Runtime service initialization failed. "
            "Set a saved IBM Quantum account or IBM_QUANTUM_TOKEN."
        ) from exc


def build_problem():
    interaction_energy = load_interaction_energy("mj_potential.csv", TARGET_PROTEIN)
    model, _ = folding_hamiltonian_vectorized(TARGET_PROTEIN, interaction_energy)
    hamiltonian, real_qubits = parse_hamiltonian_robust(str(model.obj.expr))

    top_k = min(15, real_qubits * 2)
    strong_pairs = get_strong_interaction_pairs(hamiltonian, real_qubits, top_k=top_k)
    ansatz = build_interaction_aware_ansatz(real_qubits, strong_pairs, reps=REPS)
    return hamiltonian, ansatz, real_qubits


def select_backend(service, min_num_qubits):
    if IBM_BACKEND_NAME:
        backend = service.backend(IBM_BACKEND_NAME)
    else:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=min_num_qubits,
        )

    status = backend.status()
    print(
        f"🔌 Selected Backend: {backend.name} "
        f"(status={status.status_msg}, queue={status.pending_jobs})"
    )
    return backend


def prepare_isa_problem(ansatz, hamiltonian, backend):
    ansatz_isa = transpile(ansatz, backend=backend, optimization_level=OPTIMIZATION_LEVEL)
    hamiltonian_isa = hamiltonian.apply_layout(ansatz_isa.layout)
    return ansatz_isa, hamiltonian_isa


def run_single_optimizer(
    optimizer_name,
    optimizer_options,
    backend,
    ansatz_isa,
    hamiltonian_isa,
    initial_points,
):
    print(f"👉 Testing [{optimizer_name}] on real device ({NUM_RESTARTS} run)")

    num_params = ansatz_isa.num_parameters
    histories = []
    final_energies = []
    evaluation_counts = []
    iteration_counts = []
    run_times = []
    all_job_ids = []

    for restart_idx, initial_params in enumerate(initial_points, start=1):
        current_history = []
        current_job_ids = []
        run_start = time.time()

        with Session(backend=backend) as session:
            estimator = EstimatorV2(mode=session)
            estimator.options.default_shots = SHOTS
            estimator.options.resilience_level = RESILIENCE_LEVEL

            def cost_func(params):
                pub = (ansatz_isa, [hamiltonian_isa], [params])
                job = estimator.run([pub])
                current_job_ids.append(job.job_id())
                result = job.result()
                energy = float(result[0].data.evs[0])
                current_history.append(energy)
                print(
                    f"     Eval {len(current_history):>3}: "
                    f"Energy = {energy:.4f} | Job ID = {job.job_id()}"
                )
                return energy

            if optimizer_name == "SPSA":
                opt = SPSA(maxiter=max(1, int(MAX_ITER / 2)))
                res = opt.minimize(cost_func, initial_params)
            elif optimizer_name == "Adam":
                opt = ADAM(maxiter=MAX_ITER, lr=0.1, tol=optimizer_options["tol"])
                res = opt.minimize(cost_func, initial_params)
            elif optimizer_name == "COBYLA":
                res = minimize(
                    cost_func,
                    initial_params,
                    method="COBYLA",
                    tol=optimizer_options["tol"],
                    options={"maxiter": MAX_ITER},
                )
            elif optimizer_name == "Nelder-Mead":
                res = minimize(
                    cost_func,
                    initial_params,
                    method="Nelder-Mead",
                    options={
                        "maxiter": MAX_ITER,
                        "xatol": optimizer_options["tol"],
                        "fatol": optimizer_options["tol"],
                    },
                )
            elif optimizer_name == "SLSQP":
                res = minimize(
                    cost_func,
                    initial_params,
                    method="SLSQP",
                    options={"maxiter": MAX_ITER, "ftol": optimizer_options["tol"]},
                )
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        final_energy = float(res.fun if hasattr(res, "fun") else np.min(current_history))
        nfev = int(getattr(res, "nfev", len(current_history)))
        nit = getattr(res, "nit", np.nan)
        elapsed = time.time() - run_start

        histories.append(current_history)
        final_energies.append(final_energy)
        evaluation_counts.append(nfev)
        iteration_counts.append(np.nan if isinstance(nit, str) else nit)
        run_times.append(elapsed)
        all_job_ids.append(",".join(current_job_ids))

        print(
            f"     [Run {restart_idx}] Final Energy: {final_energy:.4f} | "
            f"Eval: {nfev} | Time: {elapsed / 60:.2f} min"
        )

    mean_history, std_history = aggregate_histories(histories)

    return {
        "Optimizer": optimizer_name,
        "Backend": backend.name,
        "Num Parameters": num_params,
        "Mean Final Energy": float(np.mean(final_energies)),
        "Std Final Energy": float(np.std(final_energies)),
        "Mean Steps (nit)": safe_nanmean(iteration_counts),
        "Std Steps (nit)": safe_nanstd(iteration_counts),
        "Mean Evaluations (nfev)": float(np.mean(evaluation_counts)),
        "Std Evaluations (nfev)": float(np.std(evaluation_counts)),
        "Mean Time (s)": float(np.mean(run_times)),
        "Std Time (s)": float(np.std(run_times)),
        "Mean History": mean_history,
        "Std History": std_history,
        "All Histories": histories,
        "Job IDs": all_job_ids,
        "Color": optimizer_options["color"],
    }


def save_outputs(results, backend_name):
    save_dir = "optimizer_real_device_result"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = os.path.join(
        save_dir,
        f"optimizer_real_device_summary_{TARGET_PROTEIN}_{backend_name}_{timestamp}.txt",
    )
    csv_filename = os.path.join(
        save_dir,
        f"optimizer_real_device_history_{TARGET_PROTEIN}_{backend_name}_{timestamp}.csv",
    )
    energy_plot_filename = os.path.join(
        save_dir,
        f"optimizer_real_device_energy_{TARGET_PROTEIN}_{backend_name}_{timestamp}.png",
    )
    evaluation_plot_filename = os.path.join(
        save_dir,
        f"optimizer_real_device_evaluations_{TARGET_PROTEIN}_{backend_name}_{timestamp}.png",
    )
    time_plot_filename = os.path.join(
        save_dir,
        f"optimizer_real_device_runtime_{TARGET_PROTEIN}_{backend_name}_{timestamp}.png",
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
    ).sort_values("Mean Final Energy").reset_index(drop=True)

    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("=== Optimizer Benchmark on IBM Real Device ===\n")
        f.write(f"Target: {TARGET_PROTEIN} | Date: {timestamp}\n")
        f.write(f"Backend: {backend_name}\n")
        f.write(
            f"Settings: reps={REPS}, shots={SHOTS}, max_iter={MAX_ITER}, "
            f"restarts={NUM_RESTARTS}, resilience_level={RESILIENCE_LEVEL}\n\n"
        )
        f.write(df_summary.to_string(index=False))

    history_columns = {}
    for result in results:
        optimizer_name = result["Optimizer"]
        history_columns[f"{optimizer_name} Mean"] = pd.Series(result["Mean History"])
        history_columns[f"{optimizer_name} Std"] = pd.Series(result["Std History"])
        for run_idx, history in enumerate(result["All Histories"], start=1):
            history_columns[f"{optimizer_name} Run {run_idx}"] = pd.Series(history)

    pd.DataFrame(history_columns).to_csv(csv_filename, index_label="Evaluation")

    fig1, ax1 = plt.subplots(figsize=(12, 7))
    for result in results:
        x = np.arange(len(result["Mean History"]))
        lower = result["Mean History"] - result["Std History"]
        upper = result["Mean History"] + result["Std History"]
        ax1.plot(
            x,
            result["Mean History"],
            label=f"{result['Optimizer']} ({result['Mean Final Energy']:.2f})",
            color=result["Color"],
            linewidth=2,
        )
        ax1.fill_between(x, lower, upper, color=result["Color"], alpha=0.18)
    ax1.set_title(f"Optimizer Convergence on {backend_name}", fontsize=16, fontweight="bold")
    ax1.set_xlabel("Function Evaluations")
    ax1.set_ylabel("Energy")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(energy_plot_filename, dpi=300, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(
        [result["Optimizer"] for result in results],
        [result["Mean Evaluations (nfev)"] for result in results],
        yerr=[result["Std Evaluations (nfev)"] for result in results],
        color=[result["Color"] for result in results],
        alpha=0.75,
        capsize=5,
    )
    ax2.set_title("Function Evaluations on Real Device")
    ax2.set_ylabel("Evaluations (nfev)")
    fig2.tight_layout()
    fig2.savefig(evaluation_plot_filename, dpi=300, bbox_inches="tight")

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.bar(
        [result["Optimizer"] for result in results],
        [result["Mean Time (s)"] / 60.0 for result in results],
        yerr=[result["Std Time (s)"] / 60.0 for result in results],
        color=[result["Color"] for result in results],
        alpha=0.75,
        capsize=5,
    )
    ax3.set_title("Runtime on Real Device")
    ax3.set_ylabel("Time (minutes)")
    fig3.tight_layout()
    fig3.savefig(time_plot_filename, dpi=300, bbox_inches="tight")

    print("\n" + "=" * 100)
    print(df_summary.to_string(index=False))
    print("=" * 100)
    print(f"\n📄 Summary saved to: {txt_filename}")
    print(f"💾 History saved to: {csv_filename}")
    print(f"🖼️ Energy plot saved to: {energy_plot_filename}")
    print(f"🖼️ Evaluation plot saved to: {evaluation_plot_filename}")
    print(f"🖼️ Runtime plot saved to: {time_plot_filename}")


def run_real_device_optimizer_benchmark():
    print("🚀 Starting optimizer benchmark on IBM real device")
    print(
        f"   Target: {TARGET_PROTEIN} | reps={REPS} | shots={SHOTS} | "
        f"max_iter={MAX_ITER} | restarts={NUM_RESTARTS}"
    )

    hamiltonian, ansatz, real_qubits = build_problem()
    service = initialize_service()
    backend = select_backend(service, min_num_qubits=real_qubits)
    ansatz_isa, hamiltonian_isa = prepare_isa_problem(ansatz, hamiltonian, backend)

    rng = np.random.default_rng(RANDOM_SEED)
    initial_points = [
        rng.uniform(0, 2 * np.pi, ansatz_isa.num_parameters)
        for _ in range(NUM_RESTARTS)
    ]

    print(
        f"   Logical qubits: {real_qubits} | Physical qubits after layout: {ansatz_isa.num_qubits} | "
        f"Parameters: {ansatz_isa.num_parameters}"
    )
    print("⚠️ Each function evaluation submits a real IBM Runtime job. Expect queue time and usage cost.\n")

    results = []
    for optimizer_name, optimizer_options in OPTIMIZERS_TO_TEST.items():
        result = run_single_optimizer(
            optimizer_name,
            optimizer_options,
            backend,
            ansatz_isa,
            hamiltonian_isa,
            initial_points,
        )
        results.append(result)

    save_outputs(results, backend.name)


if __name__ == "__main__":
    run_real_device_optimizer_benchmark()
