import datetime
import os
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

from qiskit import ClassicalRegister, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import ADAM, SPSA
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

from load_mj import load_interaction_energy
from hamiltonian_Aware_VQE import (
    build_interaction_aware_ansatz,
    folding_hamiltonian_vectorized,
    get_strong_interaction_pairs,
    parse_hamiltonian_robust,
)


TARGET_PROTEIN = "KLVFFA"
REPS = 1
MAX_ITER = 1000
NUM_RESTARTS = 10
SHOTS = 4096
RANDOM_SEED = 1234

OPTIMIZERS_TO_TEST = {
    "COBYLA": {"color": "blue", "tol": 0.1},
    "Nelder-Mead": {"color": "green", "tol": 0.1},
    "SLSQP": {"color": "red", "tol": 0.1},
    "SPSA": {"color": "orange", "tol": 0.1},
    "Adam": {"color": "brown", "tol": 0.1},
}

SIMULATOR_CONFIGS = {
    "statevector": {
        "label": "Statevector",
        "shots": None,
        "color": "#1f77b4",
        "tol_scale": 1.0,
    },
    "shot-noise": {
        "label": f"Shot Noise ({SHOTS} shots)",
        "shots": SHOTS,
        "color": "#ff7f0e",
        "tol_scale": 1.0,
    },
    "noise-model": {
        "label": f"Noise Model ({SHOTS} shots, FakeBrisbane profile)",
        "shots": SHOTS,
        "color": "#2ca02c",
        "tol_scale": 0.5,
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


def build_problem():
    interaction_energy = load_interaction_energy("mj_potential.csv", TARGET_PROTEIN)
    model, _ = folding_hamiltonian_vectorized(TARGET_PROTEIN, interaction_energy)
    hamiltonian, real_qubits = parse_hamiltonian_robust(str(model.obj.expr))

    top_k = min(15, real_qubits * 2)
    strong_pairs = get_strong_interaction_pairs(hamiltonian, real_qubits, top_k=top_k)
    ansatz = build_interaction_aware_ansatz(real_qubits, strong_pairs, reps=REPS)
    return hamiltonian, ansatz


def build_term_cache(hamiltonian):
    terms = []
    for label, coeff in hamiltonian.to_list():
        support = [idx for idx, pauli in enumerate(label) if pauli != "I"]
        term_op = SparsePauliOp.from_list([(label, 1.0)])
        pauli = term_op.paulis[0]
        support = [idx for idx in range(pauli.num_qubits) if pauli.x[idx] or pauli.z[idx]]
        terms.append(
            {
                "label": label,
                "coeff": complex(coeff),
                "support": support,
                "operator": term_op,
                "pauli": pauli,
            }
        )
    return terms


def extract_noise_profile(fake_backend, reference_circuit, num_logical_qubits):
    properties = fake_backend.properties()
    logical_to_physical = list(reference_circuit.layout.final_index_layout())[:num_logical_qubits]
    physical_to_logical = {physical: logical for logical, physical in enumerate(logical_to_physical)}

    readout_errors = {}
    for logical, physical in enumerate(logical_to_physical):
        qubit_params = {item.name: item.value for item in properties.qubits[physical]}
        readout_errors[logical] = float(qubit_params.get("readout_error", 0.0))

    gate_error_lookup = {}
    avg_gate_errors = {}
    for gate in properties.gates:
        param_dict = {param.name: param.value for param in gate.parameters}
        gate_error = float(param_dict.get("gate_error", 0.0))
        gate_error_lookup[(gate.gate, tuple(gate.qubits))] = gate_error
        avg_gate_errors.setdefault(gate.gate, []).append(gate_error)

    avg_gate_errors = {
        gate_name: float(np.mean(values)) for gate_name, values in avg_gate_errors.items()
    }

    qubit_gate_factors = {logical: 1.0 for logical in range(num_logical_qubits)}
    for instruction in reference_circuit.data:
        operation = instruction.operation
        if operation.name in {"barrier", "measure"}:
            continue

        physical_qubits = tuple(reference_circuit.find_bit(qubit).index for qubit in instruction.qubits)
        gate_error = gate_error_lookup.get(
            (operation.name, physical_qubits),
            avg_gate_errors.get(operation.name, 0.0),
        )
        fidelity = max(0.0, 1.0 - gate_error)

        for physical in physical_qubits:
            logical = physical_to_logical.get(physical)
            if logical is not None:
                qubit_gate_factors[logical] *= fidelity

    return {
        "backend_name": fake_backend.name,
        "logical_to_physical": logical_to_physical,
        "readout_errors": readout_errors,
        "qubit_gate_factors": qubit_gate_factors,
    }


def prepare_simulation(mode, ansatz, term_cache):
    if mode == "statevector":
        return {
            "mode": mode,
            "label": SIMULATOR_CONFIGS[mode]["label"],
            "backend_name": "Statevector",
            "circuit": ansatz,
            "shots": None,
            "term_cache": term_cache,
            "noise_profile": None,
            "num_params": ansatz.num_parameters,
        }

    if mode == "shot-noise":
        aer_backend = AerSimulator(
            max_parallel_threads=1,
            max_parallel_experiments=1,
        )
        measurement_circuits = [
            transpile(
                build_pauli_measurement_circuit(ansatz, term["pauli"], term["support"]),
                aer_backend,
                optimization_level=1,
            )
            for term in term_cache
        ]
        return {
            "mode": mode,
            "label": SIMULATOR_CONFIGS[mode]["label"],
            "backend_name": "AerSimulator",
            "circuit": ansatz,
            "shots": SIMULATOR_CONFIGS[mode]["shots"],
            "term_cache": term_cache,
            "noise_profile": None,
            "backend": aer_backend,
            "measurement_circuits": measurement_circuits,
            "num_params": ansatz.num_parameters,
        }

    if mode == "noise-model":
        fake_backend = FakeBrisbane()
        reference_circuit = transpile(ansatz, fake_backend, optimization_level=1)
        noise_profile = extract_noise_profile(fake_backend, reference_circuit, ansatz.num_qubits)

        return {
            "mode": mode,
            "label": SIMULATOR_CONFIGS[mode]["label"],
            "backend_name": noise_profile["backend_name"],
            "circuit": ansatz,
            "shots": SIMULATOR_CONFIGS[mode]["shots"],
            "term_cache": term_cache,
            "noise_profile": noise_profile,
            "backend": None,
            "measurement_circuits": None,
            "num_params": ansatz.num_parameters,
        }

    raise ValueError(f"Unsupported simulation mode: {mode}")


def sample_pauli_mean(expectation_value, shots, rng):
    clipped = float(np.clip(expectation_value, -1.0, 1.0))
    probability_plus = 0.5 * (1.0 + clipped)
    samples = rng.choice([1.0, -1.0], size=shots, p=[probability_plus, 1.0 - probability_plus])
    return float(np.mean(samples))


def build_pauli_measurement_circuit(ansatz, pauli, support):
    measurement_circuit = ansatz.copy()
    if not support:
        return measurement_circuit

    classical_register = ClassicalRegister(len(support), name="meas")
    measurement_circuit.add_register(classical_register)

    for classical_idx, qubit_idx in enumerate(support):
        has_x = bool(pauli.x[qubit_idx])
        has_z = bool(pauli.z[qubit_idx])

        if has_x and has_z:
            measurement_circuit.sdg(qubit_idx)
            measurement_circuit.h(qubit_idx)
        elif has_x:
            measurement_circuit.h(qubit_idx)

        measurement_circuit.measure(qubit_idx, classical_register[classical_idx])

    return measurement_circuit


def counts_to_pauli_mean(counts, shots):
    if shots <= 0:
        raise ValueError("shots must be positive for sampling-based estimation.")

    parity_sum = 0.0
    for bitstring, count in counts.items():
        eigenvalue = 1.0 if bitstring.count("1") % 2 == 0 else -1.0
        parity_sum += eigenvalue * count

    return float(parity_sum / shots)


def apply_noise_profile(expectation_value, support, noise_profile):
    if not support:
        return float(expectation_value)

    gate_factor = 1.0
    readout_factor = 1.0
    for logical_qubit in support:
        gate_factor *= noise_profile["qubit_gate_factors"].get(logical_qubit, 1.0)
        readout_error = noise_profile["readout_errors"].get(logical_qubit, 0.0)
        readout_factor *= max(-1.0, 1.0 - 2.0 * readout_error)

    noisy_expectation = float(expectation_value) * gate_factor * readout_factor
    return float(np.clip(noisy_expectation, -1.0, 1.0))


def evaluate_energy(
    circuit,
    term_cache,
    params,
    mode,
    shots,
    rng,
    noise_profile=None,
    backend=None,
    measurement_circuits=None,
):
    bound_circuit = circuit.assign_parameters(params, inplace=False)

    if mode == "shot-noise":
        bound_measurement_circuits = []
        sampled_term_indices = []
        for term_idx, (term, measurement_circuit) in enumerate(zip(term_cache, measurement_circuits)):
            if not term["support"]:
                continue
            bound_measurement_circuits.append(
                measurement_circuit.assign_parameters(params, inplace=False)
            )
            sampled_term_indices.append(term_idx)

        sampled_expectations = {}
        if bound_measurement_circuits:
            seed = int(rng.integers(0, 2**31 - 1))
            result = backend.run(
                bound_measurement_circuits,
                shots=shots,
                seed_simulator=seed,
            ).result()
            for result_idx, term_idx in enumerate(sampled_term_indices):
                counts = result.get_counts(result_idx)
                sampled_expectations[term_idx] = counts_to_pauli_mean(counts, shots)

        energy = 0.0
        for term_idx, term in enumerate(term_cache):
            estimated_expectation = 1.0 if not term["support"] else sampled_expectations[term_idx]
            energy += float(np.real(term["coeff"] * estimated_expectation))

        return float(energy)

    state = Statevector.from_instruction(bound_circuit)

    energy = 0.0
    for term in term_cache:
        ideal_expectation = float(np.real(state.expectation_value(term["operator"])))

        if mode == "statevector":
            estimated_expectation = ideal_expectation
        elif mode == "shot-noise":
            estimated_expectation = sample_pauli_mean(ideal_expectation, shots, rng)
        elif mode == "noise-model":
            noisy_expectation = apply_noise_profile(
                ideal_expectation,
                term["support"],
                noise_profile,
            )
            estimated_expectation = sample_pauli_mean(noisy_expectation, shots, rng)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        energy += float(np.real(term["coeff"] * estimated_expectation))

    return float(energy)


def run_single_optimizer(optimizer_name, optimizer_options, sim_context, initial_points, run_seed_offset):
    mode = sim_context["mode"]
    print(f"👉 [{sim_context['label']}] {optimizer_name} ({NUM_RESTARTS} random starts)")

    histories = []
    final_energies = []
    evaluation_counts = []
    iteration_counts = []
    run_times = []

    tol = optimizer_options["tol"] * SIMULATOR_CONFIGS[mode]["tol_scale"]
    tol = max(tol, 1e-4)

    for restart_idx, initial_params in enumerate(initial_points, start=1):
        current_history = []
        run_start = time.time()
        rng = np.random.default_rng(run_seed_offset + restart_idx)

        def cost_func(params):
            energy = evaluate_energy(
                sim_context["circuit"],
                sim_context["term_cache"],
                params,
                mode,
                sim_context["shots"],
                rng,
                noise_profile=sim_context["noise_profile"],
                backend=sim_context.get("backend"),
                measurement_circuits=sim_context.get("measurement_circuits"),
            )
            current_history.append(energy)
            return energy

        if optimizer_name == "SPSA":
            maxiter = MAX_ITER if mode != "noise-model" else int(MAX_ITER / 2)
            opt = SPSA(maxiter=maxiter)
            res = opt.minimize(cost_func, initial_params)
        elif optimizer_name == "Adam":
            opt = ADAM(maxiter=MAX_ITER, lr=0.1, tol=tol)
            res = opt.minimize(cost_func, initial_params)
        elif optimizer_name == "COBYLA":
            res = minimize(
                cost_func,
                initial_params,
                method="COBYLA",
                tol=tol,
                options={"maxiter": MAX_ITER},
            )
        elif optimizer_name == "Nelder-Mead":
            res = minimize(
                cost_func,
                initial_params,
                method="Nelder-Mead",
                options={"maxiter": MAX_ITER, "xatol": tol, "fatol": tol},
            )
        elif optimizer_name == "SLSQP":
            res = minimize(
                cost_func,
                initial_params,
                method="SLSQP",
                options={"maxiter": MAX_ITER, "ftol": tol},
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

        print(
            f"     [Run {restart_idx}] Energy: {final_energy:.4f} | "
            f"Eval: {nfev} | Time: {elapsed:.2f}s"
        )

    mean_history, std_history = aggregate_histories(histories)

    return {
        "Simulation Mode": mode,
        "Simulation Label": sim_context["label"],
        "Backend": sim_context["backend_name"],
        "Optimizer": optimizer_name,
        "Num Parameters": sim_context["num_params"],
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
        "Optimizer Color": optimizer_options["color"],
    }


def save_outputs(results):
    save_dir = "optimizer_simulator_result"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = os.path.join(
        save_dir, f"optimizer_simulator_summary_{TARGET_PROTEIN}_{timestamp}.txt"
    )
    csv_filename = os.path.join(
        save_dir, f"optimizer_simulator_history_{TARGET_PROTEIN}_{timestamp}.csv"
    )
    energy_plot_filename = os.path.join(
        save_dir, f"optimizer_simulator_energy_{TARGET_PROTEIN}_{timestamp}.png"
    )
    evaluation_plot_filename = os.path.join(
        save_dir, f"optimizer_simulator_evaluations_{TARGET_PROTEIN}_{timestamp}.png"
    )
    time_plot_filename = os.path.join(
        save_dir, f"optimizer_simulator_runtime_{TARGET_PROTEIN}_{timestamp}.png"
    )

    df_summary = pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key not in {"Mean History", "Std History", "All Histories", "Optimizer Color"}
            }
            for result in results
        ]
    ).sort_values(["Simulation Mode", "Optimizer"]).reset_index(drop=True)

    best_idx = df_summary.groupby("Simulation Mode")["Mean Final Energy"].idxmin()
    df_best = df_summary.loc[best_idx].sort_values("Simulation Mode").reset_index(drop=True)

    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("=== Optimizer Comparison Across Simulation Modes ===\n")
        f.write(f"Target: {TARGET_PROTEIN} | Reps: {REPS} | Date: {timestamp}\n")
        f.write(f"Restarts: {NUM_RESTARTS} | Max Iter: {MAX_ITER} | Shots: {SHOTS}\n")
        f.write("Noise-model mode uses FakeBrisbane readout/gate error profile.\n\n")
        f.write("[Full Summary]\n")
        f.write(df_summary.to_string(index=False))
        f.write("\n\n[Best Optimizer per Simulation Mode]\n")
        f.write(df_best.to_string(index=False))

    history_columns = {}
    for result in results:
        base_name = f"{result['Simulation Mode']} | {result['Optimizer']}"
        history_columns[f"{base_name} Mean"] = pd.Series(result["Mean History"])
        history_columns[f"{base_name} Std"] = pd.Series(result["Std History"])
        for run_idx, history in enumerate(result["All Histories"], start=1):
            history_columns[f"{base_name} Run {run_idx}"] = pd.Series(history)

    pd.DataFrame(history_columns).to_csv(csv_filename, index_label="Evaluation")

    fig1, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    for ax, mode in zip(axes, SIMULATOR_CONFIGS.keys()):
        mode_results = [result for result in results if result["Simulation Mode"] == mode]
        for result in mode_results:
            x = np.arange(len(result["Mean History"]))
            lower = result["Mean History"] - result["Std History"]
            upper = result["Mean History"] + result["Std History"]
            ax.plot(
                x,
                result["Mean History"],
                label=result["Optimizer"],
                color=result["Optimizer Color"],
                linewidth=2,
            )
            ax.fill_between(x, lower, upper, color=result["Optimizer Color"], alpha=0.18)

        ax.set_title(SIMULATOR_CONFIGS[mode]["label"])
        ax.set_xlabel("Function Evaluations")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Energy")
    axes[-1].legend(loc="best")
    fig1.suptitle("Optimizer Energy Convergence by Simulation Mode", fontsize=16, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig(energy_plot_filename, dpi=300, bbox_inches="tight")

    optimizer_names = list(OPTIMIZERS_TO_TEST.keys())
    mode_names = list(SIMULATOR_CONFIGS.keys())
    x = np.arange(len(optimizer_names))
    width = 0.24

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    for idx, mode in enumerate(mode_names):
        subset = df_summary[df_summary["Simulation Mode"] == mode].set_index("Optimizer").loc[optimizer_names]
        ax2.bar(
            x + (idx - 1) * width,
            subset["Mean Evaluations (nfev)"],
            width=width,
            yerr=subset["Std Evaluations (nfev)"],
            color=SIMULATOR_CONFIGS[mode]["color"],
            alpha=0.8,
            capsize=4,
            label=SIMULATOR_CONFIGS[mode]["label"],
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(optimizer_names, rotation=20)
    ax2.set_ylabel("Mean Evaluations (nfev)")
    ax2.set_title("Function Evaluations by Optimizer and Simulation Mode")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(evaluation_plot_filename, dpi=300, bbox_inches="tight")

    fig3, ax3 = plt.subplots(figsize=(12, 7))
    for idx, mode in enumerate(mode_names):
        subset = df_summary[df_summary["Simulation Mode"] == mode].set_index("Optimizer").loc[optimizer_names]
        ax3.bar(
            x + (idx - 1) * width,
            subset["Mean Time (s)"],
            width=width,
            yerr=subset["Std Time (s)"],
            color=SIMULATOR_CONFIGS[mode]["color"],
            alpha=0.8,
            capsize=4,
            label=SIMULATOR_CONFIGS[mode]["label"],
        )
    ax3.set_xticks(x)
    ax3.set_xticklabels(optimizer_names, rotation=20)
    ax3.set_ylabel("Mean Time (s)")
    ax3.set_title("Runtime by Optimizer and Simulation Mode")
    ax3.legend()
    ax3.grid(True, axis="y", alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(time_plot_filename, dpi=300, bbox_inches="tight")

    print("\n" + "=" * 100)
    print(df_summary.to_string(index=False))
    print("=" * 100)
    print("\n[Best Optimizer per Simulation Mode]")
    print(df_best.to_string(index=False))
    print(f"\n📄 Summary saved to: {txt_filename}")
    print(f"💾 History saved to: {csv_filename}")
    print(f"🖼️ Energy plot saved to: {energy_plot_filename}")
    print(f"🖼️ Evaluation plot saved to: {evaluation_plot_filename}")
    print(f"🖼️ Runtime plot saved to: {time_plot_filename}")


def run_optimizer_simulator_benchmark():
    print("🚀 Starting optimizer comparison across simulation modes")
    print(f"   Target: {TARGET_PROTEIN} | Reps: {REPS}")
    print(f"   Modes: {list(SIMULATOR_CONFIGS.keys())}")
    print(f"   Restarts per optimizer: {NUM_RESTARTS} | Max iter: {MAX_ITER} | Shots: {SHOTS}\n")

    hamiltonian, ansatz = build_problem()
    term_cache = build_term_cache(hamiltonian)
    results = []

    sim_contexts = {
        mode: prepare_simulation(mode, ansatz, term_cache)
        for mode in SIMULATOR_CONFIGS
    }

    for mode, sim_context in sim_contexts.items():
        print(f"\n{'=' * 100}")
        print(f"🔬 Simulation Mode: {sim_context['label']} | Backend: {sim_context['backend_name']}")
        print(f"{'=' * 100}")

        rng = np.random.default_rng(RANDOM_SEED)
        initial_points = [
            rng.uniform(0, 2 * np.pi, sim_context["num_params"])
            for _ in range(NUM_RESTARTS)
        ]

        for optimizer_idx, (optimizer_name, optimizer_options) in enumerate(OPTIMIZERS_TO_TEST.items()):
            result = run_single_optimizer(
                optimizer_name,
                optimizer_options,
                sim_context,
                initial_points,
                run_seed_offset=RANDOM_SEED + 1000 * list(SIMULATOR_CONFIGS.keys()).index(mode) + 100 * optimizer_idx,
            )
            results.append(result)

    save_outputs(results)


if __name__ == "__main__":
    run_optimizer_simulator_benchmark()
