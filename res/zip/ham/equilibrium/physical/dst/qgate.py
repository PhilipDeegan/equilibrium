"""
DST quantum logic gates.

A qubit in DST is the complex transverse amplitude of a canal's u-field:

    |ψ⟩ = α|0⟩ + β|1⟩  ↔  canal with transverse components (α, β) ∈ ℂ²

The two basis states correspond to the two orthogonal transverse polarisations
of the u-wave.  All physics of single-qubit gates reduces to unitary rotations
of this two-component canal field.

Two-qubit gates (CNOT, CZ) use the canal between two knots: the same
non-factorisable joint canal that underlies Bell correlations.  The CNOT is
the substrate mechanism by which one knot's state conditionally updates
another's — mediated by the canal, not by direct contact.

Measurement follows Born rule / Malus's law:
    P(outcome 0) = |α|²,  P(outcome 1) = |β|²

This is Malus's law (P(+1) = cos²(θ−φ)) expressed in the computational basis.
The state collapses to the measured basis vector — the canal is re-emitted in
the projected orientation, exactly as in the Bell test polariser model.

Connection to Bell:
    bell.dst_joint_canal samples the two-qubit joint canal state (the Bell
    state |Φ+⟩ or |Ψ−⟩) and recovers C(Δθ) = −cos(2Δθ).
    The Bell state is prepared here by H(qubit 0) → CNOT(0→1).
"""

import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Single-qubit gate matrices
# (substrate operations — unitary transformations of the canal transverse field)
# ---------------------------------------------------------------------------

X = np.array([[0, 1], [1, 0]], dtype=complex)  # bit flip (λ/2 plate at 45°)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)  # bit+phase flip
Z = np.array([[1, 0], [0, -1]], dtype=complex)  # phase flip
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # beamsplitter (λ/4 + λ/2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)  # π/2 phase gate
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # π/4 phase gate


def Rx(theta: float) -> np.ndarray:
    """Rotation by θ around the X-axis of the Bloch sphere."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def Ry(theta: float) -> np.ndarray:
    """Rotation by θ around the Y-axis of the Bloch sphere."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def Rz(theta: float) -> np.ndarray:
    """Rotation by θ around the Z-axis of the Bloch sphere."""
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


# ---------------------------------------------------------------------------
# DSTCircuit — n-qubit state vector
# ---------------------------------------------------------------------------


class DSTCircuit:
    """
    n-qubit quantum circuit in the DST canal representation.

    State: complex vector of length 2^n.
    Ordering: qubit 0 is the most-significant bit.
        e.g. for n=2: index 0=|00⟩, 1=|01⟩, 2=|10⟩, 3=|11⟩

    Gates are substrate operations applied to one or two canal fields.
    CNOT is the canonical two-knot canal-mediated conditional update.
    """

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.sv = np.zeros(2**n_qubits, dtype=complex)
        self.sv[0] = 1.0  # start in |00…0⟩

    # ------------------------------------------------------------------
    # Single-qubit gate
    # ------------------------------------------------------------------

    def apply(self, gate: np.ndarray, qubit: int) -> "DSTCircuit":
        """Apply a 2×2 gate to one qubit (in-place)."""
        n, sv = self.n, self.sv
        for idx in range(2**n):
            if idx & (1 << (n - 1 - qubit)):
                continue  # only process pairs once
            partner = idx | (1 << (n - 1 - qubit))
            a, b = sv[idx], sv[partner]
            sv[idx] = gate[0, 0] * a + gate[0, 1] * b
            sv[partner] = gate[1, 0] * a + gate[1, 1] * b
        return self

    # ------------------------------------------------------------------
    # Two-qubit gates  (canal-mediated)
    # ------------------------------------------------------------------

    def cnot(self, control: int, target: int) -> "DSTCircuit":
        """
        CNOT gate: flip target qubit when control is |1⟩.

        DST mechanism: the canal from control knot to target knot propagates
        the control u-field state; when it arrives as |1⟩, the Malus projection
        on the target applies the bit-flip (X gate) to the target canal.
        """
        n, sv = self.n, self.sv
        c_bit = 1 << (n - 1 - control)
        t_bit = 1 << (n - 1 - target)
        for idx in range(2**n):
            if (idx & c_bit) and not (idx & t_bit):
                sv[idx], sv[idx | t_bit] = sv[idx | t_bit].copy(), sv[idx].copy()
        return self

    def cz(self, q0: int, q1: int) -> "DSTCircuit":
        """CZ gate: add −1 phase when both qubits are |1⟩."""
        n, sv = self.n, self.sv
        b0, b1 = 1 << (n - 1 - q0), 1 << (n - 1 - q1)
        for idx in range(2**n):
            if (idx & b0) and (idx & b1):
                sv[idx] *= -1
        return self

    def swap(self, q0: int, q1: int) -> "DSTCircuit":
        """SWAP: exchange two qubit canals."""
        n, sv = self.n, self.sv
        b0, b1 = 1 << (n - 1 - q0), 1 << (n - 1 - q1)
        for idx in range(2**n):
            if bool(idx & b0) != bool(idx & b1):  # exactly one is set
                partner = idx ^ b0 ^ b1
                if idx < partner:
                    sv[idx], sv[partner] = sv[partner].copy(), sv[idx].copy()
        return self

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def probabilities(self) -> np.ndarray:
        """Born-rule probabilities for all 2^n basis states."""
        return np.abs(self.sv) ** 2

    def measure(self, shots: int = 1024) -> np.ndarray:
        """
        Sample the circuit shots times.  Returns counts array of length 2^n.
        Each shot collapses the canal to a definite basis state (Malus's law).
        """
        probs = self.probabilities()
        outcomes = np.random.default_rng().choice(2**self.n, size=shots, p=probs)
        return np.bincount(outcomes, minlength=2**self.n)

    def state_str(self) -> str:
        """Human-readable state vector (non-negligible amplitudes only)."""
        n, sv = self.n, self.sv
        parts = []
        for i, amp in enumerate(sv):
            if abs(amp) < 1e-9:
                continue
            bits = format(i, f"0{n}b")
            parts.append(f"({amp.real:+.3f}{amp.imag:+.3f}j)|{bits}⟩")
        return "  +  ".join(parts) if parts else "0"

    def bloch(self, qubit: int) -> Tuple[float, float, float]:
        """
        Reduced Bloch vector (x,y,z) for a single qubit.
        Computed from the single-qubit reduced density matrix.
        """
        n = self.n
        bit = 1 << (n - 1 - qubit)
        rho = np.zeros((2, 2), dtype=complex)
        for i in range(2**n):
            for j in range(2**n):
                if (i ^ j) & ~bit:
                    continue  # must agree on all other qubits
                row = 0 if not (i & bit) else 1
                col = 0 if not (j & bit) else 1
                rho[row, col] += self.sv[i] * np.conj(self.sv[j])
        x = 2.0 * rho[0, 1].real
        y = 2.0 * rho[0, 1].imag
        z = float((rho[0, 0] - rho[1, 1]).real)
        return x, y, z


# ---------------------------------------------------------------------------
# Standard circuit factories
# ---------------------------------------------------------------------------


def bell_state(which: int = 0) -> DSTCircuit:
    """
    Prepare one of the four Bell states.

    The Bell state is the two-qubit non-factorisable joint canal state.
    It IS the entangled canal of dst_joint_canal in bell.py.

    which=0: |Φ+⟩ = (|00⟩ + |11⟩)/√2  — correlated
    which=1: |Φ−⟩ = (|00⟩ − |11⟩)/√2
    which=2: |Ψ+⟩ = (|01⟩ + |10⟩)/√2  — anticorrelated
    which=3: |Ψ−⟩ = (|01⟩ − |10⟩)/√2  — singlet (used in Bell test)
    """
    c = DSTCircuit(2)
    c.apply(H, 0).cnot(0, 1)
    if which == 1:
        c.apply(Z, 0)
    elif which == 2:
        c.apply(X, 0)
    elif which == 3:
        c.apply(X, 0).apply(Z, 0)
    return c


def ghz_state(n: int = 3) -> DSTCircuit:
    """
    n-qubit GHZ state: (|00…0⟩ + |11…1⟩)/√2.
    Maximal multipartite entanglement — all canals correlated.
    """
    c = DSTCircuit(n)
    c.apply(H, 0)
    for i in range(1, n):
        c.cnot(0, i)
    return c


def teleport_circuit(alpha: complex = 1.0, beta: complex = 0.0) -> DSTCircuit:
    """
    Three-qubit quantum teleportation circuit.

    Qubit 0: payload |ψ⟩ = α|0⟩ + β|1⟩  (to be teleported)
    Qubits 1,2: Bell pair (shared canal between Alice and Bob)

    After the circuit and classical corrections, qubit 2 carries |ψ⟩.
    The teleportation demonstrates that the canal (entanglement) can
    transfer a quantum state without the physical qubit crossing the canal.
    """
    norm = np.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    alpha, beta = alpha / norm, beta / norm

    c = DSTCircuit(3)
    c.sv[0] = alpha  # |000⟩ component
    c.sv[2] = beta  # |010⟩ component  (qubit 0 = |1⟩)
    # Actually set qubit 0 to (alpha,beta) directly in the tensor product
    c.sv = np.zeros(8, dtype=complex)
    c.sv[0b000] = alpha
    c.sv[0b100] = beta

    # Step 1: entangle qubits 1 and 2 (Bell pair — the canal)
    c.apply(H, 1).cnot(1, 2)
    # Step 2: Alice's Bell measurement on qubits 0,1
    c.cnot(0, 1).apply(H, 0)
    # (Classical corrections would be applied based on measuring qubits 0,1)
    return c
