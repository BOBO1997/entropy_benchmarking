import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import pickle
from datetime import datetime
from zoneinfo import ZoneInfo

# -----------------------------
# Pauli matrices (sparse)
# -----------------------------
I2 = sp.csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.complex128))
X2 = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128))
Y2 = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
Z2 = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128))

PAULI = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}


def kron_all(ops):
    """Kronecker product of a list of sparse matrices."""
    out = ops[0]
    for op in ops[1:]:
        out = sp.kron(out, op, format="csr")
    return out


def two_site_term(N, i, j, P, Q):
    """Return operator P_i Q_j on N qubits (0-indexed) as sparse matrix."""
    ops = []
    for k in range(N):
        if k == i:
            ops.append(PAULI[P])
        elif k == j:
            ops.append(PAULI[Q])
        else:
            ops.append(I2)
    return kron_all(ops)


def heisenberg_ring_pauli(N, J=1.0):
    """
    H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}) with periodic boundary.
    """
    dim = 2**N
    H = sp.csr_matrix((dim, dim), dtype=np.complex128)
    for i in range(N):
        j = (i + 1) % N
        H += J * (two_site_term(N, i, j, "X", "X") +
                  two_site_term(N, i, j, "Y", "Y") +
                  two_site_term(N, i, j, "Z", "Z"))
    return H


def heisenberg_open_chain_pauli(L, J=1.0):
    """
    Open chain block Hamiltonian on L spins:
    H_block = J * sum_{i=0}^{L-2} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    """
    dim = 2**L
    H = sp.csr_matrix((dim, dim), dtype=np.complex128)
    for i in range(L - 1):
        j = i + 1
        H += J * (two_site_term(L, i, j, "X", "X") +
                  two_site_term(L, i, j, "Y", "Y") +
                  two_site_term(L, i, j, "Z", "Z"))
    return H


# -----------------------------
# Upper bound via variational principle
# -----------------------------
def variational_upper_bound_eigsh(H, ncv=50, maxiter=2000, tol=1e-12, seed=0):
    """
    Compute an upper bound on the ground energy by running eigsh (Lanczos).
    The returned Rayleigh quotient of the approximate eigenvector is >= true ground energy.
    """
    rng = np.random.default_rng(seed)
    dim = H.shape[0]
    v0 = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v0 = v0 / np.linalg.norm(v0)

    # eigsh returns smallest algebraic eigenvalue (which is what we want)
    evals, evecs = spla.eigsh(H, k=1, which="SA", v0=v0, ncv=min(ncv, dim-1),
                             maxiter=maxiter, tol=tol)
    psi = evecs[:, 0]
    # Rayleigh quotient (robust)
    num = np.vdot(psi, H @ psi)
    den = np.vdot(psi, psi)
    E_var = np.real(num / den)
    return E_var


# -----------------------------
# Lower bound via sliding-block Anderson bound
# -----------------------------
def sliding_block_lower_bound(N, L, J=1.0):
    """
    Lower bound using length-L open-chain blocks.

    Define blocks B_i = {i, i+1, ..., i+L-1} (mod N).
    Each nearest-neighbor bond appears in exactly (L-1) blocks.
    Hence:
        H_ring = (1/(L-1)) * sum_i H_block(B_i)
    Therefore:
        E0(H_ring) >= (1/(L-1)) * sum_i E0(H_block) = (N/(L-1)) * e_L
    where e_L is the ground energy of the length-L open-chain block.
    """
    if L < 2 or L > N:
        raise ValueError("Need 2 <= L <= N.")
    H_block = heisenberg_open_chain_pauli(L, J=J)
    # For block size up to ~12, sparse eigsh is fine; for very small L dense also works.
    eL = np.real(spla.eigsh(H_block, k=1, which="SA", tol=1e-12)[0][0])
    return (N / (L - 1)) * eL, eL


def best_block_lower_bound(N, J=1.0, L_min=2, L_max=None):
    """
    Compute the best (largest) lower bound among block lengths L.
    """
    if L_max is None:
        L_max = N
    best = -np.inf
    best_info = None
    for L in range(L_min, L_max + 1):
        lb, eL = sliding_block_lower_bound(N, L, J=J)
        if lb > best:
            best = lb
            best_info = (L, lb, eL)
    return best_info  # (best_L, best_lower_bound, block_ground_energy)


def main():
    N = 8
    J = 1.0

    H = heisenberg_ring_pauli(N, J=J)

    # 1) Upper bound (variational) via Lanczos/eigsh
    E_upper = variational_upper_bound_eigsh(H, ncv=80, maxiter=5000, tol=1e-14, seed=1)

    # 2) Lower bound via sliding blocks (choose best L)
    best_L, E_lower, eL = best_block_lower_bound(N, J=J, L_min=2, L_max=N)

    # 3) Very coarse universal bounds (sanity)
    coarse_lower = -3.0 * N * J  # because each bond has min eigenvalue -3
    coarse_upper = +1.0 * N * J  # because each bond has max eigenvalue +1

    print("=== Quantum XXX Heisenberg ring (Pauli form) ===")
    print(f"N={N}, J={J}")
    print()
    print("[Coarse operator bounds]")
    print(f"Lower >= {coarse_lower:.12f}")
    print(f"Upper <= {coarse_upper:.12f}")
    print()
    print("[Tighter bounds (classical computation)]")
    print(f"Lower bound (best sliding block): L={best_L},  E0 >= {E_lower:.12f}   (e_L(open)={eL:.12f})")
    print(f"Upper bound (variational eigsh):            E0 <= {E_upper:.12f}")
    print()
    print("[Interval]")
    print(f"{E_lower:.12f}  <=  E0  <=  {E_upper:.12f}")
    
    ### save data
    with open("run_classical.pkl", "wb") as f:
        pickle.dump(obj={"energy_classical_lower": E_lower,
                         "energy_classical_upper": E_upper,
                         "datetime": datetime.now(ZoneInfo("Europe/Paris")),
                        },
                    file=f)

if __name__ == "__main__":
    main()

