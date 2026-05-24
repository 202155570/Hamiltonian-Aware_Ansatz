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

# 노이즈 모델 및 가짜 백엔드 관련 임포트
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

# 기존 파일들에서 필요한 함수 임포트
from load_mj import load_interaction_energy
from hamiltonian_Aware_VQE import (
    folding_hamiltonian_vectorized,
    parse_hamiltonian_robust,
    get_strong_interaction_pairs,
    build_interaction_aware_ansatz
)

# ==========================================
# 1. 노이즈 및 벤치마크 설정
# ==========================================
TARGET_PROTEIN = "KLVFFA"  
REPS = 1  
SHOTS = 4096
MAX_ITER = 1000  # 노이즈 환경이므로 함수 호출 횟수를 적절히 제한
NUM_RESTARTS = 10

# 노이즈 모델 설정 (FakeBrisbane)
fake_backend = FakeBrisbane()
BACKEND = AerSimulator.from_backend(fake_backend)

# 테스트할 옵티마이저 (노이즈 환경을 고려하여 허용 오차 tol 완화)
OPTIMIZERS_TO_TEST = {
    "COBYLA": {"color": "blue", "tol": 0.05},
    "Nelder-Mead": {"color": "green", "tol": 0.05},
    "SLSQP": {"color": "red", "tol": 0.05},
    "SPSA": {"color": "orange", "tol": 0.05},   # 노이즈에 가장 강건함
    "Adam": {"color": "brown", "tol": 0.05},
}


def aggregate_histories(histories):
    max_len = max(len(history) for history in histories)
    history_array = np.full((len(histories), max_len), np.nan)

    for idx, history in enumerate(histories):
        history_array[idx, :len(history)] = history

    return (
        np.nanmean(history_array, axis=0),
        np.nanstd(history_array, axis=0),
    )


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

def run_noisy_optimizer_benchmark():
    print(f"🚀 Starting Noisy Optimizer Benchmark for Target: {TARGET_PROTEIN}")
    print(f"   Backend: {fake_backend.name} (Simulated Noise)")
    print(f"   Ansatz: Hamiltonian-Aware (Reps={REPS})\n")

    # 1. Hamiltonian 및 Ansatz 생성
    print("🔨 Building and Mapping Hamiltonian...")
    interaction_energy = load_interaction_energy('mj_potential.csv', TARGET_PROTEIN)
    model, num_qubits = folding_hamiltonian_vectorized(TARGET_PROTEIN, interaction_energy)
    hamiltonian, real_qubits = parse_hamiltonian_robust(str(model.obj.expr))
    
    top_k = min(15, real_qubits * 2)
    strong_pairs = get_strong_interaction_pairs(hamiltonian, real_qubits, top_k=top_k)
    ansatz = build_interaction_aware_ansatz(real_qubits, strong_pairs, reps=REPS)

    results = []

    # 2. 노이즈 환경에 맞춘 회로 트랜스파일 및 해밀토니안 매핑
    # 노이즈 시뮬레이션을 위해 실제 장치의 토폴로지(Layout)에 맞춰 변환해야 합니다.
    ansatz_transpiled = transpile(ansatz, BACKEND, optimization_level=3)
    layout_info = ansatz_transpiled.layout.final_index_layout()

    if isinstance(layout_info, dict):
        physical_qubits = list(layout_info.values())
    else:
        physical_qubits = layout_info
        
    # 14큐비트 해밀토니안을 127큐비트(Brisbane) 실제 위치로 매핑
    hamiltonian_mapped = hamiltonian.apply_layout(
        layout=physical_qubits, 
        num_qubits=BACKEND.num_qubits
    )
    print(f"   -> Successfully mapped to {len(physical_qubits)} physical qubits.")

    num_params = ansatz_transpiled.num_parameters
    estimator = EstimatorV2()
    estimator.options.default_shots = SHOTS

    # ==========================================
    # 3. 옵티마이저별 실행 함수
    # ==========================================
    def run_strategy(optimizer_name, options):
        print(f"\n👉 Testing [{optimizer_name}] in Noisy Environment ({NUM_RESTARTS} random starts, averaging results)...")
        start_time = time.time()
        
        histories = []
        final_energies = []
        evaluation_counts = []
        iteration_counts = []
        run_times = []

        for i in range(NUM_RESTARTS):
            current_history = []
            run_start_time = time.time()
            
            def cost_func(params):
                pub = (ansatz_transpiled, [hamiltonian_mapped], [params])
                job = estimator.run([pub])
                energy = float(job.result()[0].data.evs[0])
                current_history.append(energy)
                return energy

            initial_params = np.random.uniform(0, 2*np.pi, num_params)
            
            # --- 최적화 실행 ---
            if optimizer_name == 'SPSA':
                # SPSA는 한 단계당 2번 측정하므로 MAX_ITER/2로 설정하여 공평하게 비교
                opt = SPSA(maxiter=int(MAX_ITER/2))
                res = opt.minimize(cost_func, initial_params)
            elif optimizer_name == 'Adam':
                opt = ADAM(maxiter=MAX_ITER, lr=0.1, tol=options["tol"])
                res = opt.minimize(cost_func, initial_params)
            elif optimizer_name == 'COBYLA':
                res = minimize(cost_func, initial_params, method='COBYLA', 
                               tol=options["tol"], options={'maxiter': MAX_ITER})
            elif optimizer_name == 'Nelder-Mead':
                res = minimize(cost_func, initial_params, method='Nelder-Mead', 
                               options={'maxiter': MAX_ITER, 'xatol': options["tol"], 'fatol': options["tol"]})
            elif optimizer_name == 'SLSQP':
                res = minimize(cost_func, initial_params, method='SLSQP', 
                               options={'maxiter': MAX_ITER, 'ftol': options["tol"]})
            
            # 결과 추출
            res_fun = res.fun if hasattr(res, 'fun') else min(current_history)
            res_nfev = res.nfev if hasattr(res, 'nfev') else len(current_history)
            res_nit = getattr(res, 'nit', np.nan)
            run_elapsed = time.time() - run_start_time

            histories.append(current_history)
            final_energies.append(res_fun)
            evaluation_counts.append(res_nfev)
            iteration_counts.append(np.nan if isinstance(res_nit, str) else res_nit)
            run_times.append(run_elapsed)
            
            print(f"     [Run {i+1}] Final Energy: {res_fun:.4f} | Eval(nfev): {res_nfev} | Time: {run_elapsed:.2f}s")
        
        elapsed_time = time.time() - start_time
        mean_history, std_history = aggregate_histories(histories)
        
        return {
            "Optimizer": optimizer_name,
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

    # 4. 각 옵티마이저 실행
    for name, opts in OPTIMIZERS_TO_TEST.items():
        results.append(run_strategy(name, opts))

    # ==========================================
    # 5. 결과 요약 및 파일 저장
    # ==========================================
    save_dir = "optimizer_noise_result"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = os.path.join(save_dir, f"noisy_optimizer_summary_{TARGET_PROTEIN}_{timestamp}.txt")
    csv_filename = os.path.join(save_dir, f"noisy_optimizer_history_{TARGET_PROTEIN}_{timestamp}.csv")
    energy_plot_filename = os.path.join(save_dir, f"noisy_optimizer_energy_convergence_{TARGET_PROTEIN}_{timestamp}.png")
    evaluation_plot_filename = os.path.join(save_dir, f"noisy_optimizer_evaluations_{TARGET_PROTEIN}_{timestamp}.png")
    time_plot_filename = os.path.join(save_dir, f"noisy_optimizer_runtime_{TARGET_PROTEIN}_{timestamp}.png")

    df_summary = pd.DataFrame([
        {
            k: v for k, v in r.items()
            if k not in {"Mean History", "Std History", "All Histories", "Color"}
        }
        for r in results
    ])

    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(f"=== Noisy VQE Optimizer Benchmark Summary ===\n")
        f.write(f"Target: {TARGET_PROTEIN} | Date: {timestamp}\n")
        f.write(f"Backend: {fake_backend.name} (Simulated Noise)\n")
        f.write(f"Settings: Shots={SHOTS}, MaxIter={MAX_ITER}, Restarts={NUM_RESTARTS}\n")
        f.write("="*80 + "\n\n")
        f.write(df_summary.to_string(index=False))

    history_columns = {}
    for res in results:
        optimizer_name = res["Optimizer"]
        history_columns[f"{optimizer_name} Mean"] = pd.Series(res["Mean History"])
        history_columns[f"{optimizer_name} Std"] = pd.Series(res["Std History"])
        for run_idx, history in enumerate(res["All Histories"], start=1):
            history_columns[f"{optimizer_name} Run {run_idx}"] = pd.Series(history)

    pd.DataFrame(history_columns).to_csv(csv_filename, index_label="Function Evaluation")

    print("\n" + "="*75)
    print(df_summary.to_string(index=False))
    print("="*75)
    print(f"📄 Detailed report saved to: {txt_filename}")
    print(f"💾 Energy history data saved to: {csv_filename}")

    # ==========================================
    # 6. 시각화 및 개별 그래프 저장
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    for res in results:
        x = np.arange(len(res["Mean History"]))
        lower = res["Mean History"] - res["Std History"]
        upper = res["Mean History"] + res["Std History"]
        ax1.plot(
            x,
            res["Mean History"],
            label=f"{res['Optimizer']} ({res['Mean Final Energy']:.2f} ± {res['Std Final Energy']:.2f})",
            color=res["Color"],
            linewidth=2.5,
            alpha=0.9,
        )
        ax1.fill_between(x, lower, upper, color=res["Color"], alpha=0.18)
    ax1.set_title(f"Noisy Energy Convergence Mean ± Std (Target: {TARGET_PROTEIN} on {fake_backend.name})", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Total Function Evaluations (n_fev)")
    ax1.set_ylabel("Energy")
    ax1.legend()
    ax1.grid(True, alpha=0.4, linestyle='--')
    fig1.tight_layout()
    fig1.savefig(energy_plot_filename, dpi=300, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(
        [r['Optimizer'] for r in results],
        [r['Mean Evaluations (nfev)'] for r in results],
        yerr=[r['Std Evaluations (nfev)'] for r in results],
        color=[r['Color'] for r in results],
        alpha=0.7,
        edgecolor='black',
        capsize=5,
    )
    ax2.set_title("Total Resource Consumption (Function Evaluations)")
    ax2.set_ylabel("Evaluations (nfev)")
    ax2.tick_params(axis='x', rotation=45)
    fig2.tight_layout()
    fig2.savefig(evaluation_plot_filename, dpi=300, bbox_inches="tight")

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.bar(
        [r['Optimizer'] for r in results],
        [r['Mean Time (s)'] for r in results],
        yerr=[r['Std Time (s)'] for r in results],
        color=[r['Color'] for r in results],
        alpha=0.7,
        edgecolor='black',
        capsize=5,
    )
    ax3.set_title("Total Computation Time")
    ax3.set_ylabel("Time (seconds)")
    ax3.tick_params(axis='x', rotation=45)
    fig3.tight_layout()
    fig3.savefig(time_plot_filename, dpi=300, bbox_inches="tight")

    print(f"🖼️ Energy convergence plot saved to: {energy_plot_filename}")
    print(f"🖼️ Evaluation comparison plot saved to: {evaluation_plot_filename}")
    print(f"🖼️ Runtime comparison plot saved to: {time_plot_filename}")

    plt.show()

if __name__ == "__main__":
    run_noisy_optimizer_benchmark()
