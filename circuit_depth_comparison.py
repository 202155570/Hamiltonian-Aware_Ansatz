"""
Compare raw circuit.depth() across peptides for each ansatz at reps=2.

Unlike table3_resource_comparison.py (which transpiles to a fixed basis
before measuring depth), this script reports the depth of the ansatz
circuit as-built, i.e. the value returned by `QuantumCircuit.depth()`.
"""

import pandas as pd

from qiskit.circuit.library import TwoLocal

from hamiltonian_Aware_VQE import (
    build_interaction_aware_ansatz,
    folding_hamiltonian_vectorized,
    get_strong_interaction_pairs,
    parse_hamiltonian_robust,
)
from CDQA_simul import build_cd_only_ansatz
from load_mj import load_interaction_energy


REPS = 2
HA_TOPK_FACTOR = 2
HA_ENTANGLER = "cz"
TWOLOCAL_ROT = "ry"
TWOLOCAL_ENT = "cz"
TWOLOCAL_PATTERN = "linear"

PEPTIDES = [
    "YGGFM",
    "KLVFFA",
    "APRLPFY",
    "AVDINNNA",
    "CYIQNCPLG",
    "GYDPETGTWG",
    "RPKPQQFFGLM",
]
ANSATZ_NAMES = ["HA", "TwoLocal", "DC-QAOA"]


def build_hamiltonian(peptide):
    interaction_energy = load_interaction_energy("mj_potential.csv", peptide)
    model, _ = folding_hamiltonian_vectorized(peptide, interaction_energy)
    hamiltonian, num_qubits = parse_hamiltonian_robust(str(model.obj.expr))
    return hamiltonian, num_qubits


def make_ha_ansatz(hamiltonian, num_qubits):
    top_k = HA_TOPK_FACTOR * num_qubits
    pairs = get_strong_interaction_pairs(hamiltonian, num_qubits, top_k=top_k)
    return build_interaction_aware_ansatz(
        num_qubits, pairs, reps=REPS, entangler=HA_ENTANGLER
    )


def make_twolocal_ansatz(num_qubits):
    return TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks=TWOLOCAL_ROT,
        entanglement_blocks=TWOLOCAL_ENT,
        entanglement=TWOLOCAL_PATTERN,
        reps=REPS,
        insert_barriers=False,
    ).decompose()


def make_dcqaoa_ansatz(num_qubits):
    return build_cd_only_ansatz(num_qubits, reps=REPS)


def build_ansatz(name, hamiltonian, num_qubits):
    if name == "HA":
        return make_ha_ansatz(hamiltonian, num_qubits)
    if name == "TwoLocal":
        return make_twolocal_ansatz(num_qubits)
    if name == "DC-QAOA":
        return make_dcqaoa_ansatz(num_qubits)
    raise ValueError(f"Unknown ansatz: {name}")


def collect_depths():
    rows = []
    for peptide in PEPTIDES:
        print(f"\n=== Peptide: {peptide} ===")
        hamiltonian, num_qubits = build_hamiltonian(peptide)
        print(f"   qubits = {num_qubits}")

        row = {"peptide": peptide, "qubits": num_qubits, "reps": REPS}
        for name in ANSATZ_NAMES:
            ansatz = build_ansatz(name, hamiltonian, num_qubits)
            depth = ansatz.depth()
            print(f"   [{name:<8}] circuit.depth() = {depth}")
            row[name] = depth

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    df = collect_depths()

    out_path = "circuit_depth_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    print("\n=== Circuit depth (reps=2, no transpile) ===")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    main()
