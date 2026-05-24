import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize
import re
import sympy as sp
import pyomo.environ as pyo

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

# Aer 관련 모듈
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2

# 사용자 환경의 load_mj가 있다면 사용, 없으면 더미 함수 사용
from load_mj import load_interaction_energy


# ==========================================
# 1. Hamiltonian & Physics Logic
# ==========================================

def get_coords_from_bitstring_3d(bitstring):
    bits = [int(b) for b in bitstring]
    f_vals = bits[::-1] 
    
    coords = [(0, 0, 0), (1, 1, 1)]; cx, cy, cz = 1, 1, 1
    vec_map = {(0,0): (1,1,1), (0,1): (1,-1,-1), (1,0): (-1,1,-1), (1,1): (-1,-1,1)}
    num_turns = len(f_vals) // 2
    for k in range(num_turns):
        if 2*k+1 >= len(f_vals): break
        b0, b1 = f_vals[2*k], f_vals[2*k+1]
        parity = (-1)**(k) 
        vx, vy, vz = vec_map[(b0, b1)]
        cx += vx*parity; cy += vy*parity; cz += vz*parity
        coords.append((cx, cy, cz))
    return coords

def folding_hamiltonian_vectorized(main_chain: str, interaction_energy: dict):
    model = pyo.ConcreteModel()
    N = len(main_chain); num_turns = N - 2; num_qubits = num_turns * 2   
    cutoff_dist_sq = num_turns * 4.0; base_penalty = 50.0               
    model.f = pyo.Var(range(num_qubits), domain=pyo.Binary)
    
    def ind(b0, b1, type):
        if type==0: return (1-b0)*(1-b1)
        if type==1: return (1-b0)*b1
        if type==2: return b0*(1-b1)
        if type==3: return b0*b1
    
    vecs = []
    for k in range(num_turns):
        b0, b1 = model.f[2*k], model.f[2*k+1]
        vx = ind(b0,b1,0)+ind(b0,b1,1)-ind(b0,b1,2)-ind(b0,b1,3)
        vy = ind(b0,b1,0)-ind(b0,b1,1)+ind(b0,b1,2)-ind(b0,b1,3)
        vz = ind(b0,b1,0)-ind(b0,b1,1)-ind(b0,b1,2)+ind(b0,b1,3)
        vecs.append((vx,vy,vz))

    def get_dist_sq(i, j):
        sx, sy, sz = 0,0,0
        for k in range(i, j):
            if k < 0: vx,vy,vz = 1,1,1
            else: vx,vy,vz = vecs[k]
            p = (-1)**k
            sx+=vx*p; sy+=vy*p; sz+=vz*p
        return sx**2 + sy**2 + sz**2

    E = 0
    for i in range(num_turns+1):
        for j in range(i+4, num_turns+2):
            if j-1 > i-1: E += -2.0 * get_dist_sq(i-1, j-1)
    for k in range(num_turns-1): E += base_penalty * (8.0 - get_dist_sq(k, k+2))
    for i in range(N):
        for j in range(i+2, N):
            if j==i+2: continue
            dist = get_dist_sq(i-1, j-1)
            try: s = interaction_energy[main_chain[i]][main_chain[j]]
            except: s = -1.0
            if s < 0: E += s * (cutoff_dist_sq - dist) * 0.5
            else: E += -1.0 * s * dist * 0.1
    model.obj = pyo.Objective(expr=E, sense=pyo.minimize)
    return model, num_qubits

def parse_hamiltonian_robust(obj_str):
    indices = [int(x) for x in re.findall(r'f\[(\d+)\]', obj_str)]
    if not indices: return SparsePauliOp(["I"], [0]), 0
    real_num = max(indices) + 1
    z = [sp.Symbol(f'Z{i}') for i in range(real_num)]
    f = [(1-zi)/2 for zi in z]
    expr = eval(obj_str, {"__builtins__": None}, {'f': f})
    if not isinstance(expr, sp.Add): expr = sp.Add(expr)
    pl, cs = [], []
    for term in sp.expand(expr).args:
        c, op = term.as_coeff_Mul()
        if abs(c) < 1e-5: continue
        cs.append(complex(c))
        s = ["I"]*real_num
        if op != 1:
            for fac in (op.args if op.func == sp.Mul else [op]):
                base_str = str(fac.base if fac.is_Pow else fac)
                if base_str.startswith('Z'):
                    idx = int(base_str[1:])
                    s[real_num-1-idx] = "Z"
        pl.append("".join(s))
    return SparsePauliOp(pl, cs), real_num

def analyze_hamiltonian(hamiltonian, num_qubits=None, max_display=20):
    print(f"\n📊 Hamiltonian Analysis Report")
    print(f"   - Total Terms: {len(hamiltonian)}")
    if num_qubits:
        print(f"   - Qubits: {num_qubits}")
    print("=" * 80)
    print(f"{'No.':<5} | {'Pauli String':<20} | {'Coeff':<10} | {'Type (Guess)'}")
    print("-" * 80)

    sorted_indices = np.argsort(np.abs(hamiltonian.coeffs))[::-1]

    for i, idx in enumerate(sorted_indices):
        if i >= max_display:
            print(f"... (and {len(hamiltonian) - max_display} more terms) ...")
            break

        p = hamiltonian.paulis[idx]
        c = hamiltonian.coeffs[idx]
        p_str = str(p)
        val = np.real(c) 
        
        z_count = p_str.count('Z')
        meaning = ""

        if z_count == 0:
            meaning = "🟢 Global Offset"
        elif z_count == 1:
            meaning = "🔹 Linear Bias"
        elif z_count == 2:
            if val > 10.0:
                meaning = "AAA Penalty (제약)"
            elif val > 0:
                meaning = "🔸 Repulsion (척력)"
            else:
                meaning = "❤️ Attraction (인력)"
        else:
             meaning = "🌀 Multi-body"

        print(f"{i+1:<5} | {p_str:<20} | {val:<10.4f} | {meaning}")
    print("=" * 80)

# ==========================================
# 2. Strategy: Interaction Screening
# ==========================================

def get_strong_interaction_pairs(hamiltonian, num_qubits, top_k=5):
    interactions = []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        pauli_str = pauli.to_label()
        reversed_str = pauli_str[::-1] 
        active_indices = [i for i, char in enumerate(reversed_str) if char != 'I']
        if len(active_indices) == 2:
            interactions.append((tuple(sorted(active_indices)), abs(np.real(coeff))))
    
    interactions.sort(key=lambda x: x[1], reverse=True)
    unique_pairs = []
    seen = set()
    for pair, weight in interactions:
        if pair not in seen:
            unique_pairs.append(pair)
            seen.add(pair)
    return unique_pairs[:top_k]

DEFAULT_ENTANGLER = "cz"


def build_interaction_aware_ansatz(num_qubits, interaction_pairs, reps=1, entangler=DEFAULT_ENTANGLER):
    entangler = entangler.lower()
    if entangler not in {"cz", "yz"}:
        raise ValueError(f"Unsupported entangler: {entangler}")

    qc = QuantumCircuit(num_qubits)
    params_per_layer = num_qubits
    if entangler == "yz":
        params_per_layer += len(interaction_pairs)

    total_params = params_per_layer * reps
    theta = ParameterVector('θ', total_params)
    param_idx = 0
    
    for layer in range(reps):
        for i in range(num_qubits):
            qc.ry(theta[param_idx], i)
            param_idx += 1
        qc.barrier()

        for (q_i, q_j) in interaction_pairs:
            if entangler == "cz":
                qc.cz(q_i, q_j)
            else:
                # Original interaction-aware YZ evolution path is kept here
                # so we can switch back easily for ablation studies.
                pauli_list = ["I"] * num_qubits
                pauli_list[num_qubits - 1 - q_i] = "Y"
                pauli_list[num_qubits - 1 - q_j] = "Z"
                op = SparsePauliOp("".join(pauli_list))
                qc.append(PauliEvolutionGate(op, time=theta[param_idx]), qc.qubits)
                param_idx += 1

        qc.barrier()

    return qc

# ==========================================
# 3. Main Simulation & Visualization
# ==========================================

def run_simulation_and_visualize():
    # --- A. 문제 정의 ---
    print("🚀 [Step 1] Problem Setup...")
    # 예시 단백질: CYIQNCPLG (Cys-Tyr-Ile-Gln-Asn-Cys-Pro-Leu-Gly)
    target_protein = "KLVFFA" 
    
    interaction_energy = load_interaction_energy('mj_potential.csv', target_protein)
    
    model, num_qubits = folding_hamiltonian_vectorized(target_protein, interaction_energy)
    hamiltonian, real_qubits = parse_hamiltonian_robust(str(model.obj.expr))
    print(f"   -> Chain: {target_protein}, Qubits: {real_qubits}")

    analyze_hamiltonian(hamiltonian, num_qubits=real_qubits)

    # --- B. Ansatz 구성 ---
    print("\n🔍 [Step 2] Building Ansatz...")
    # top_k = min(15, real_qubits * 2)
    top_k = real_qubits * 2
    strong_pairs = get_strong_interaction_pairs(hamiltonian, real_qubits, top_k=top_k)
    print(f"   -> Top-{top_k} Interactions: {strong_pairs}")
    
    ansatz_raw = build_interaction_aware_ansatz(real_qubits, strong_pairs, reps=2)
    
    print("🔄 [Step 2.5] Transpiling Circuit for Aer Simulator...")
    backend = AerSimulator()
    ansatz = transpile(ansatz_raw, backend=backend)
    
    print(f"   -> Transpiled Ansatz Parameters: {ansatz.num_parameters}")

    # --- C. VQE Optimization Loop ---
    print("⚙️ [Step 3] Running VQE Optimization...")
    estimator = EstimatorV2()
    estimator.options.default_shots = 4096
    history = []
    
    best_energy = float('inf')
    
    def cost_func(params):
        nonlocal best_energy
        pub = (ansatz, [hamiltonian], [params])
        job = estimator.run([pub])
        result = job.result()[0]
        energy = float(result.data.evs[0]) 
        
        if energy < best_energy:
            best_energy = energy
            
        history.append(energy)
        if len(history) % 50 == 0:
            print(f"   Iter {len(history)}: Energy = {energy:.4f}")
        return energy

    initial_theta = np.random.random(ansatz.num_parameters) * 2 * np.pi
    
    # COBYLA 최적화
    res = scipy.optimize.minimize(cost_func, initial_theta, method='COBYLA', options={'maxiter': 200})
    print(f"✅ Optimization Done! Min Energy: {res.fun:.4f}")

    # --- D. Final Sampling ---
    print("📊 [Step 4] Collecting Results...")
    sampler = SamplerV2()
    
    qc_optimal = ansatz.assign_parameters(res.x)
    qc_optimal.measure_all()
    qc_optimal = transpile(qc_optimal, backend=backend)

    pub = (qc_optimal,) 
    job = sampler.run([pub], shots=4096) # 샷 수 증가
    
    result = job.result()[0]
    counts = result.data.meas.get_counts()
    
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    best_bitstring = sorted_counts[0][0]

    # ==========================================
    # E. Visualization (User Requested Style)
    # ==========================================
    print("\n🎨 Generating Plots...")
    
    # 2x2 레이아웃의 기본 크기 유지 (하지만 4번째 그래프는 뺌)
    fig = plt.figure(figsize=(16, 12)) 

    # [Plot 1] Convergence (Energy)
    ax1 = fig.add_subplot(221)
    ax1.plot(history, color='blue', linewidth=2)
    ax1.set_title("Energy Convergence", fontsize=14, fontweight='bold')
    ax1.axhline(best_energy, color='red', linestyle='--', label=f'Min: {best_energy:.2f}')
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Energy")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # [Plot 2] Result Histogram
    ax2 = fig.add_subplot(222)
    top_counts = dict(sorted_counts) # 이미 정렬된 데이터 사용
    ax2.bar(top_counts.keys(), top_counts.values(), color='mediumpurple', edgecolor='black')
    ax2.set_title('Top 10 Measurement Outcomes', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Bitstring')
    ax2.set_ylabel('Counts')
    ax2.tick_params(axis='x', rotation=45)

    # [Plot 3] 3D Structure (Hydrophobicity Coloring)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d') 
    
    coords = get_coords_from_bitstring_3d(best_bitstring)
    xs = [p[0] for p in coords]; ys = [p[1] for p in coords]; zs = [p[2] for p in coords]
    aa_seq = list(target_protein)
    
    # Backbone
    ax3.plot(xs, ys, zs, color='gray', linestyle='--', linewidth=1, label='Backbone')

    # 친수성/소수성 색상 매핑
    hydrophobic_residues = {'G', 'A', 'V', 'C', 'P', 'L', 'I', 'M', 'W', 'F'}
    
    # 범례용 가짜 플롯
    ax3.scatter([], [], c='orange', s=100, label='Hydrophobic (소수성)')
    ax3.scatter([], [], c='dodgerblue', s=100, label='Hydrophilic (친수성)')

    for idx, (x, y, z) in enumerate(coords):
        if idx < len(aa_seq):
            aa = aa_seq[idx]
            label = f"{idx+1}:{aa}"
            
            # 색상 결정
            if aa in hydrophobic_residues:
                color = 'orange' 
            else:
                color = 'dodgerblue' 
            
            ax3.scatter(x, y, z, s=300, color=color, edgecolor='k', alpha=0.9)
            ax3.text(x, y, z+0.2, label, fontsize=10, ha='center', fontweight='bold')
            
    ax3.set_title(f"Predicted 3D Structure ({target_protein})", fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right')

    plt.tight_layout()
    plt.show()



def solve_exactly_brute_force(hamiltonian, num_qubits):
    """
    모든 비트스트링(2^n)을 대입하여 가장 낮은 에너지를 찾습니다.
    """
    best_energy = float('inf')
    best_bitstring = None
    
    # 2^n개의 모든 경우의 수 생성
    for i in range(2**num_qubits):
        # 숫자를 비트스트링으로 변환 (예: 5 -> '00000101')
        bitstring = bin(i)[2:].zfill(num_qubits)
        
        # SparsePauliOp의 에너지를 계산하는 간단한 로직
        # Z는 비트가 0이면 +1, 1이면 -1을 반환함
        current_energy = 0
        for pauli_str, coeff in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs):
            val = 1.0
            for idx, char in enumerate(reversed(pauli_str)): # Qiskit 순서 (오른쪽이 0번)
                if char == 'Z':
                    if bitstring[idx] == '1':
                        val *= -1.0
            current_energy += val * coeff
        
        energy_real = np.real(current_energy)
        if energy_real < best_energy:
            best_energy = energy_real
            best_bitstring = bitstring
            
    return best_energy, best_bitstring

# 사용 예시:
# min_e, best_bits = solve_exactly_brute_force(hamiltonian, real_qubits)
# print(f"최적 에너지(정답): {min_e:.4f}")
# print(f"최적 구조 비트스트링: {best_bits}")

if __name__ == "__main__":

    print("🚀 [Step 1] Problem Setup...")
    # 예시 단백질: CYIQNCPLG (Cys-Tyr-Ile-Gln-Asn-Cys-Pro-Leu-Gly)
    target_protein = "CYIQNCPLG" 
    
    interaction_energy = load_interaction_energy('mj_potential.csv', target_protein)
    
    model, num_qubits = folding_hamiltonian_vectorized(target_protein, interaction_energy)
    hamiltonian, real_qubits = parse_hamiltonian_robust(str(model.obj.expr))
    print(f"   -> Chain: {target_protein}, Qubits: {real_qubits}")

    analyze_hamiltonian(hamiltonian, num_qubits=real_qubits)
    min_e, best_bits = solve_exactly_brute_force(hamiltonian, real_qubits)
    print(f"최적 에너지(정답): {min_e:.4f}")
    print(f"최적 구조 비트스트링: {best_bits}")
    
    
    # run_simulation_and_visualize()
