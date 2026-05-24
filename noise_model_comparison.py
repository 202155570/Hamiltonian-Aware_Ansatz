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
from qiskit.circuit.library import TwoLocal
# [추가] 노이즈 모델 및 가짜 백엔드 관련 임포트
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeBrisbane # 최신 127큐비트급 장치 노이즈 모델
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.quantum_info import SparsePauliOp

# 기존 파일들에서 필요한 함수 임포트
from load_mj import load_interaction_energy
from hamiltonian_Aware_VQE import (
    folding_hamiltonian_vectorized,
    parse_hamiltonian_robust,
    get_strong_interaction_pairs,
    build_interaction_aware_ansatz
)
from CDQA_simul import build_cd_only_ansatz

# ==========================================
# 1. 노이즈 설정 (Noise Configuration)
# ==========================================
TARGET_PROTEIN = "CYIQNCPLG"
REPS = 2
SHOTS = 4096
MAX_ITER = 1000
NUM_RESTARTS = 10

# [수정] 실제 장치(FakeBrisbane)의 노이즈 모델을 가져와 시뮬레이터 생성
fake_backend = FakeBrisbane()
noise_model = NoiseModel.from_backend(fake_backend)
# 노이즈 모델이 적용된 시뮬레이터 (리소스 소모 0)
BACKEND = AerSimulator.from_backend(fake_backend)

# service = QiskitRuntimeService()
# real_backend = service.backend("ibm_yonsei") # 혹은 사용 가능한 다른 장치

# # 2. 지금 이 순간의 실제 노이즈 프로파일 다운로드
# noise_model = NoiseModel.from_backend(real_backend)

# # 3. 실시간 노이즈 모델을 탑재한 시뮬레이터 생성
# # 이 시뮬레이터는 실제 장치의 현재 에러율을 그대로 흉내냅니다.
# BACKEND = AerSimulator(noise_model=noise_model)

print(f"✅ Noise Model Loaded from: {fake_backend.name}")


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


def run_benchmark():
    print(f"🚀 Starting Noisy Simulation Benchmark for Target: {TARGET_PROTEIN}")
    
    # 1. Hamiltonian 생성 (공통)
    interaction_energy = load_interaction_energy('mj_potential.csv', TARGET_PROTEIN)
    model, num_qubits = folding_hamiltonian_vectorized(TARGET_PROTEIN, interaction_energy)
    hamiltonian, real_qubits = parse_hamiltonian_robust(str(model.obj.expr))
    print(f"   -> Qubits: {real_qubits}, Hamiltonian Terms: {len(hamiltonian)}\n")

    results = []

    def run_strategy(name, ansatz, color):
        print(f"👉 Running [{name}] in Noisy Environment ({NUM_RESTARTS} random starts, averaging results)...")
        start_time = time.time()
        
        # [중요] 노이즈 환경에서는 장치의 레이아웃에 맞게 트랜스파일이 필수입니다.
        ansatz_transpiled = transpile(ansatz, BACKEND, optimization_level=3)
        layout_info = ansatz_transpiled.layout.final_index_layout()
    
        # 만약 layout_info가 딕셔너리면 values()를 쓰고, 리스트면 그대로 사용
        if isinstance(layout_info, dict):
            physical_qubits = list(layout_info.values())
        else:
            physical_qubits = layout_info  # 이미 리스트인 경우
            
        # 14큐비트 해밀토니안을 127큐비트 시스템의 실제 위치로 매핑
        hamiltonian_mapped = hamiltonian.apply_layout(
            layout=physical_qubits, 
            num_qubits=BACKEND.num_qubits
        )

        print(f"   -> Hamiltonian successfully mapped to {len(physical_qubits)} physical qubits on a {BACKEND.num_qubits}-qubit device.")

        num_params = ansatz_transpiled.num_parameters
        
        # Estimator 설정 (노이즈 환경에서는 V2 사용 권장)
        estimator = EstimatorV2()
        estimator.options.default_shots = SHOTS
        
        histories = []
        final_energies = []
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

            # 최적화 (COBYLA)
            # 최적화 (SLSQP)
            initial_params = np.random.uniform(0, 2*np.pi, num_params)
            # SLSQP: Gradient 기반으로 시뮬레이션에서 더 정밀한 수렴 가능
            # res = minimize(cost_func, initial_params, method='SLSQP', options={'maxiter': MAX_ITER, 'ftol': 0.02}) 
            # COBYLA
            # [수정] 노이즈 환경(FakeBackend)에서는 1e-6 수준의 정밀도로 수렴하기 어렵습니다.
            # 샷 노이즈(약 1/sqrt(4096) ≈ 0.015)를 고려하여 허용 오차(tol)를 완화해야 조기 종료됩니다.
            res = minimize(cost_func, initial_params, method='COBYLA', options={'maxiter': MAX_ITER, 'tol': 0.02})
            
            # [수정] COBYLA는 'nit' 속성이 없으므로 안전하게 가져오기 (없으면 nfev 사용)
            current_nit = getattr(res, 'nit', res.nfev)
            run_elapsed = time.time() - run_start_time

            histories.append(current_history)
            final_energies.append(res.fun)
            iteration_counts.append(np.nan if isinstance(current_nit, str) else current_nit)
            run_times.append(run_elapsed)
            
            print(f"     [Run {i+1}] Energy: {res.fun:.4f} (Iter: {current_nit}) | Time: {run_elapsed:.2f}s")
        
        elapsed_time = time.time() - start_time
        mean_history, std_history = aggregate_histories(histories)
        
        print(f"   -> Mean Final Energy: {np.mean(final_energies):.4f} ± {np.std(final_energies):.4f}, Total Time: {elapsed_time:.2f}s")
        
        return {
            "Algorithm": name,
            "Mean Final Energy": float(np.mean(final_energies)),
            "Std Final Energy": float(np.std(final_energies)),
            "Parameters": num_params,
            "Mean Iterations": safe_nanmean(iteration_counts),
            "Std Iterations": safe_nanstd(iteration_counts),
            "Mean Time (s)": round(float(np.mean(run_times)), 2),
            "Std Time (s)": round(float(np.std(run_times)), 2),
            "Total Time (s)": round(elapsed_time, 2),
            "Mean History": mean_history,
            "Std History": std_history,
            "All Histories": histories,
            "Color": color,
        }

    # --- 전략 실행 (리소스 절약을 위해 대표 항목만 선별) ---
    
    # 1. Proposed Method (Reps=1)
    top_k = min(15, real_qubits * 2)
    strong_pairs = get_strong_interaction_pairs(hamiltonian, real_qubits, top_k=top_k)
    ansatz_aware = build_interaction_aware_ansatz(real_qubits, strong_pairs, reps=1)
    results.append(run_strategy("Proposed (Noisy)", ansatz_aware, "blue"))

    # 2. Standard VQE (Reps=1)
    ansatz_std_1 = TwoLocal(real_qubits, 'ry', 'cz', 'linear', reps=1).decompose()
    results.append(run_strategy("Standard VQE (Noisy)", ansatz_std_1, "lightgreen"))

    # --- Strategy B2: Standard VQE (Reps=2) ---
    ansatz_std_2 = TwoLocal(
        num_qubits=real_qubits,
        rotation_blocks='ry',
        entanglement_blocks='cz',
        entanglement='linear',
        reps=REPS,
        insert_barriers=True
    ).decompose()
    results.append(run_strategy(f"Standard VQE (Depth={REPS})", ansatz_std_2, "green"))

    # --- Strategy C1: CD-QAOA (Reps=1) ---
    ansatz_cd_1 = build_cd_only_ansatz(real_qubits, reps=1)
    results.append(run_strategy("CD-QAOA (Depth=1)", ansatz_cd_1, "lightcoral"))

    # --- Strategy C2: CD-QAOA (Reps=2) ---
    ansatz_cd_2 = build_cd_only_ansatz(real_qubits, reps=REPS)
    results.append(run_strategy(f"CD-QAOA (Depth={REPS})", ansatz_cd_2, "red"))

    # 결과 시각화 부분은 기존 코드와 동일 (results 리스트 사용)


    print("\n✅ Noisy Simulation Complete. Check the plots for hardware-realistic results.")

    df = pd.DataFrame([
        {
            k: v for k, v in r.items()
            if k not in {"Mean History", "Std History", "All Histories", "Color"}
        }
        for r in results
    ])
    print(df.to_string(index=False))
    print("="*60)

    # --- [추가] 결과 파일 저장 로직 ---
    save_dir = "noise_simulation_result"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = os.path.join(save_dir, f"benchmark_summary_{TARGET_PROTEIN}_{timestamp}.txt")
    csv_filename = os.path.join(save_dir, f"benchmark_history_{TARGET_PROTEIN}_{timestamp}.csv")
    energy_plot_filename = os.path.join(save_dir, f"benchmark_energy_convergence_{TARGET_PROTEIN}_{timestamp}.png")
    parameter_plot_filename = os.path.join(save_dir, f"benchmark_parameter_vs_energy_{TARGET_PROTEIN}_{timestamp}.png")
    iteration_plot_filename = os.path.join(save_dir, f"benchmark_iterations_{TARGET_PROTEIN}_{timestamp}.png")

    # 1. TXT 파일 저장 (요약 및 상세 설정)
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(f"=== Quantum Protein Folding Benchmark Results ===\n")
        f.write(f"Target: {TARGET_PROTEIN}\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Backend: {fake_backend.name} (Simulated Noise)\n")
        f.write(f"Settings: Reps={REPS}, Shots={SHOTS}, MaxIter={MAX_ITER}, Restarts={NUM_RESTARTS}\n")
        f.write("="*80 + "\n\n")
        f.write("[Summary Table]\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "="*80 + "\n")
        for res in results:
            f.write(f"\n>> Algorithm: {res['Algorithm']}\n")
            f.write(f"   Mean Final Energy: {res['Mean Final Energy']:.6f}\n")
            f.write(f"   Std Final Energy: {res['Std Final Energy']:.6f}\n")
            f.write(f"   Mean Iterations: {res['Mean Iterations']:.2f}\n")
            f.write(f"   Std Iterations: {res['Std Iterations']:.2f}\n")
            f.write(f"   Mean Time: {res['Mean Time (s)']}s\n")
            f.write(f"   Std Time: {res['Std Time (s)']}s\n")
    print(f"📄 Detailed report saved to: {txt_filename}")

    # 2. CSV 파일 저장 (에너지 수렴 기록)
    history_columns = {}
    for res in results:
        algorithm_name = res["Algorithm"]
        history_columns[f"{algorithm_name} Mean"] = pd.Series(res["Mean History"])
        history_columns[f"{algorithm_name} Std"] = pd.Series(res["Std History"])
        for run_idx, history in enumerate(res["All Histories"], start=1):
            history_columns[f"{algorithm_name} Run {run_idx}"] = pd.Series(history)

    pd.DataFrame(history_columns).to_csv(csv_filename, index_label="Iteration")
    print(f"💾 Energy history data saved to: {csv_filename}")
    print("="*60)

    # 그래프 그리기 및 개별 저장
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    for res in results:
        x = np.arange(len(res["Mean History"]))
        lower = res["Mean History"] - res["Std History"]
        upper = res["Mean History"] + res["Std History"]

        ax1.plot(
            x,
            res["Mean History"],
            label=f"{res['Algorithm']} ({res['Mean Final Energy']:.2f} ± {res['Std Final Energy']:.2f})",
            color=res["Color"],
            linewidth=2.5,
            alpha=0.9,
        )
        ax1.fill_between(x, lower, upper, color=res["Color"], alpha=0.18)
    
    ax1.set_title(f"Energy Convergence Mean ± Std (Target: {TARGET_PROTEIN})", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Energy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(energy_plot_filename, dpi=300, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for res in results:
        ax2.errorbar(
            res["Parameters"],
            res["Mean Final Energy"],
            yerr=res["Std Final Energy"],
            fmt='o',
            markersize=12,
            color=res["Color"],
            ecolor=res["Color"],
            elinewidth=2,
            capsize=5,
            label=res["Algorithm"],
            markeredgecolor='black',
        )

    # ax2.set_title("Efficiency Comparison (Parameters vs Energy)")
    ax2.set_xlabel("Number of Parameters", fontsize = 20)
    ax2.set_ylabel("Minimum Energy", fontsize = 20)

    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(parameter_plot_filename, dpi=300, bbox_inches="tight")

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    names = [res['Algorithm'] for res in results]
    iters = [res['Mean Iterations'] for res in results]
    iter_stds = [res['Std Iterations'] for res in results]
    colors = [res['Color'] for res in results]
    
    ax3.bar(names, iters, yerr=iter_stds, color=colors, alpha=0.7, edgecolor='black', capsize=5)
    ax3.set_title("Iterations to Convergence")
    ax3.set_ylabel("Iterations (nit)")
    ax3.tick_params(axis='x', rotation=45)
    fig3.tight_layout()
    fig3.savefig(iteration_plot_filename, dpi=300, bbox_inches="tight")

    print(f"🖼️ Energy convergence plot saved to: {energy_plot_filename}")
    print(f"🖼️ Parameter vs energy plot saved to: {parameter_plot_filename}")
    print(f"🖼️ Iteration comparison plot saved to: {iteration_plot_filename}")

    plt.show()

if __name__ == "__main__":
    run_benchmark()
