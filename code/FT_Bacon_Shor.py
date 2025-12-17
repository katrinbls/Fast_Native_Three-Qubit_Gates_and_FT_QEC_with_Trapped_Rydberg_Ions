import cirq
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product
import time
import pandas as pd
from pathlib import Path

# For the BS code implementation with noise we reduce the number of Swap ancillas to one because for the simulation
# it does not play a role.

### NOISE MODEL AND SIMULATION###
class CircuitLevelNoise(cirq.NoiseModel):
    """Noise model that distinguishes 2-qubit NN vs NNN gates via tags.
    - Tag a 2-qubit op with 'nnn' (or 'NNN' / 'next_nearest') to apply the NNN error rate.
    - Untagged 2-qubit ops get the NN error rate.
    - 3-qubit ops get a separate error rate.
    - 1-qubit ops and ops with >3 qubits are passed through unchanged (adjust if needed).
    """

    def __init__(self, p_2q_nn: float = 0.0, p_2q_nnn: float = 0.0, p_3q: float = 0.0):
        self.p_2q_nn  = float(np.clip(p_2q_nn,  0.0, 1.0))
        self.p_2q_nnn = float(np.clip(p_2q_nnn, 0.0, 1.0))
        self.p_3q     = float(np.clip(p_3q,     0.0, 1.0))

    def _depol(self, n_qubits: int, p: float):
        """Return an n-qubit depolarizing channel or None if p == 0."""
        if p <= 0.0:
            return None
        return cirq.depolarize(p, n_qubits=n_qubits)

    def noisy_operation(self, op: cirq.Operation):
        # Unwrap tags if present
        if isinstance(op, cirq.TaggedOperation):
            base_op = op.sub_operation
            tags = set(op.tags)
        else:
            base_op = op
            tags = set()

        n = len(base_op.qubits)

        # Pass through 1-qubit ops and ops with >3 qubits
        if n == 1 or n > 3:
            return op

        # Two-qubit: decide NN vs NNN by tag
        if n == 2:
            is_nnn = any(t in {"nnn", "NNN", "next_nearest"} for t in tags)
            p = self.p_2q_nnn if is_nnn else self.p_2q_nn
            ch = self._depol(2, p)
            return [op, ch.on(*base_op.qubits)] if ch is not None else op

        # Three-qubit: single rate
        if n == 3:
            ch = self._depol(3, self.p_3q)
            return [op, ch.on(*base_op.qubits)] if ch is not None else op

        # Fallback
        return op

def noisy_density_matrix_simulation(err_2q_nn, err_2q_nnn, err_3q, init_state = '0', path=None):
    data_positions = [5, 1, 2, 3, 6, 4, 11, 12, 10] #positions after BC
    qubit = [cirq.LineQubit(i) for i in range(13)]
    qubit_order = qubit  # gleiche Objekte, gleiche Reihenfolge
    index_map = {q: i for i, q in enumerate(qubit_order)}
    
    #encoding circuit
    enc_circuit = encoding_circuit(qubit, encoded_state=init_state)
      
    #QEC circuit
    QEC_circuit = cirq.Circuit()
    QEC_circuit += red_readout_circuit_X(qubit)
    QEC_circuit += red_correction_circuit_Z(qubit)
    QEC_circuit += red_readout_circuit_Z(qubit)
    QEC_circuit += red_correction_circuit_X(qubit)
    
    #post-processing circuit
    post_processing_circuit = classical_post_processing_circuit(qubit)
    
    
    #simulate noise-free encoding
    enc_sim = cirq.DensityMatrixSimulator(noise=None, dtype=np.complex128)
    enc_res = enc_sim.simulate(
        enc_circuit,
        qubit_order=qubit_order
    )
    enc_rho = _normalize_density_matrix(enc_res.final_density_matrix)
  
    #noisy simulation
    noise_model = CircuitLevelNoise(err_2q_nn, err_2q_nnn, err_3q)
    sim_noise = cirq.DensityMatrixSimulator(noise=noise_model, dtype=np.complex128)
    res_noise = sim_noise.simulate(QEC_circuit, initial_state= enc_rho, qubit_order=qubit_order)
    rho_noise = _normalize_density_matrix(res_noise.final_density_matrix)

    #simulate noise-free post-processing
    sim = cirq.DensityMatrixSimulator(noise=None, dtype=np.complex128)
    res = sim.simulate(
        post_processing_circuit,
        initial_state=rho_noise,
        qubit_order=qubit_order
    )
    rho = _normalize_density_matrix(res.final_density_matrix)
    
    # get the stabilizers and logical operators
    operators = get_stabilizer_logicals(data_positions)
           
    # Compute expectation values
    expectation_vals = {
        name: float(np.round(np.real_if_close(pauli_string.expectation_from_density_matrix(rho, qubit_map=index_map)), 12))
        for name, pauli_string in operators.items()
    }
    
    print("\nExpectation values:")
    print("-" * 40)
    for name, val in expectation_vals.items():
        print(f"{name:<20s}: {val:>8.4f}")
        print("-" * 40)
        
    # --- Optional CSV persistence (one line per run) -------------------------
    if path is not None:
        path = Path(path)
        if path.is_dir() or str(path).endswith(os.sep):
            path = path / "expectation_values.csv"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Combine error rates + expectation values into a single flat dict
        row_data = {
            "init_state": init_state,
            "err_2q_nn": err_2q_nn,
            "err_2q_nnn": err_2q_nnn,
            "err_3q": err_3q,
            **expectation_vals
        }

        # If file exists, append new row; otherwise create new file
        df_row = pd.DataFrame([row_data])
        if path.exists():
            df_row.to_csv(path, mode="a", header=False, index=False)
        else:
            df_row.to_csv(path, index=False)

        print(f"\n One-row CSV saved to: {path}")

    return expectation_vals, rho

def noisy_MC_simulation(err_2q_nn, err_2q_nnn, err_3q, shots=10_000, init_state='0', path=None, seed=52):
    """
    Minimal MonteCarlo-based simulation with CSV output.
    - Encode (ideal)
    - QEC block (with mid-circuit noise)
    - Post-processing (ideal)
    - Measure each stabilizer/logical operator separately
    """
    qubits = [cirq.LineQubit(i) for i in range(13)]
    qubit_order = qubits

    # Circuits
    enc = encoding_circuit(qubits, encoded_state=init_state)
    qec = cirq.Circuit()
    qec += red_readout_circuit_X(qubits)
    qec += red_correction_circuit_Z(qubits)
    qec += red_readout_circuit_Z(qubits)
    qec += red_correction_circuit_X(qubits)
    post = classical_post_processing_circuit(qubits)

    # Noise only on QEC part
    noise_model = CircuitLevelNoise(err_2q_nn, err_2q_nnn, err_3q)
    qec_noisy = qec.with_noise(noise_model)

    # Observables (PauliStrings)
    data_positions = [5, 1, 2, 3, 6, 4, 11, 12, 10]
    operators = get_stabilizer_logicals(data_positions)

    sim = cirq.Simulator(seed=seed)
    results = {}

    for name, P in operators.items():
        c = cirq.Circuit()
        c += enc
        c += qec_noisy
        c += post

        meas_qubits = []
        for q, p in P.items():
            if p == cirq.X:
                c += cirq.H(q)
                meas_qubits.append(q)
            elif p == cirq.Y:
                c += cirq.Z(q)**-0.5  # S^\dagger
                c += cirq.H(q)
                meas_qubits.append(q)
            elif p == cirq.Z:
                meas_qubits.append(q)

        if meas_qubits:
            c += cirq.measure(*meas_qubits, key="m")

        r = sim.run(c, repetitions=shots, qubit_order=qubit_order)

        if meas_qubits:
            bits = r.measurements["m"]
            ev = (1 - 2*bits).prod(axis=1)
            results[name] = float(ev.mean())
        else:
            results[name] = float(np.real_if_close(P.coefficient))

    # --- Optional CSV output ---
    if path is not None:
        path = Path(path)
        if path.is_dir() or str(path).endswith(os.sep):
            path = path / "expectation_values_shot_based.csv"
        path.parent.mkdir(parents=True, exist_ok=True)

        row_data = {
            "init_state": init_state,
            "shots": shots,
            "err_2q_nn": err_2q_nn,
            "err_2q_nnn": err_2q_nnn,
            "err_3q": err_3q,
            **results
        }

        df_row = pd.DataFrame([row_data])
        if path.exists():
            df_row.to_csv(path, mode="a", header=False, index=False)
        else:
            df_row.to_csv(path, index=False)

        print(f"Results appended to: {path}")

    return results



### CIRCUIT ELEMENTS ###

def FT_SWAP(a, d1, d2):
    #one line needs to consist of nnn CNOT gates
    return [cirq.CNOT(a, d1), cirq.CNOT(d1, a), cirq.CNOT(a, d1),
            cirq.CNOT(d1, d2).with_tags('nnn'), cirq.CNOT(d2, d1).with_tags('nnn'), cirq.CNOT(d1, d2).with_tags('nnn'),
            cirq.CNOT(d2, a), cirq.CNOT(a, d2), cirq.CNOT(d2, a)]  
  
    
def nFT_SWAP(d1, d2):
    #non FT SWAP gate
    return [cirq.CNOT(d1, d2), cirq.CNOT(d2, d1), cirq.CNOT(d1, d2),]  


# reduced number of FT SWAPs
def red_readout_circuit_X(qubit):
    if len(qubit) != 13:
        raise ValueError("Expected 13 qubits, got {}".format(len(qubit)))
    
    # create X-stabilizer readout circuit
    x_readout = cirq.Circuit()

    # add Hadamard gates to the ancilla qubits
    x_readout.append([cirq.H(qubit[1]), cirq.H(qubit[2]), cirq.H(qubit[12])])
    
    # readout X_1
    # reduced number of FT SWAPs
    x_readout.append(
        [
            cirq.CNOT(qubit[3], qubit[2]).with_tags('nnn'), #nnn
            cirq.CNOT(qubit[2], qubit[3]).with_tags('nnn'), #nnn
            cirq.CNOT(qubit[4], qubit[3]),
            cirq.CNOT(qubit[3], qubit[4]),
            FT_SWAP(qubit[0], qubit[4], qubit[5]),
            cirq.CNOT(qubit[6], qubit[5]).with_tags('nnn'), #nnn
            cirq.CNOT(qubit[5], qubit[6]).with_tags('nnn'), #nnn
            cirq.CNOT(qubit[7], qubit[6]),
            cirq.CNOT(qubit[6], qubit[7]),
            FT_SWAP(qubit[0], qubit[7], qubit[8]),
            cirq.CNOT(qubit[9], qubit[8]).with_tags('nnn'), #nnn
            cirq.CNOT(qubit[8], qubit[9]).with_tags('nnn'), #nnn
            cirq.CNOT(qubit[9], qubit[10]),              
        ]
    )
    
    # readout X_2
    # reduced number of FT SWAPs
    x_readout.append(
        [
            cirq.CNOT(qubit[11], qubit[12]).with_tags('nnn'), #nnn
            cirq.CNOT(qubit[12], qubit[11]).with_tags('nnn'), #nnn
            cirq.CNOT(qubit[10], qubit[11]), 
            cirq.CNOT(qubit[11], qubit[10]), 
            nFT_SWAP(qubit[9], qubit[10]),
            FT_SWAP(qubit[0], qubit[8], qubit[9]),
            cirq.CNOT(qubit[7], qubit[8]), 
            cirq.CNOT(qubit[8], qubit[7]), 
            cirq.CNOT(qubit[6], qubit[7]), 
            cirq.CNOT(qubit[7], qubit[6]), 
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            cirq.CNOT(qubit[5], qubit[4]), 
            cirq.CNOT(qubit[5], qubit[3]).with_tags('nnn') #nnn
        ]
    )
    
    # readout X_3 
    x_readout.append(
        [
            cirq.CNOT(qubit[2], qubit[1]),
            cirq.CNOT(qubit[1], qubit[2]),
            nFT_SWAP(qubit[2], qubit[3]),
            cirq.CNOT(qubit[4], qubit[3]),
            cirq.CNOT(qubit[3], qubit[4]),
            nFT_SWAP(qubit[4], qubit[5]),
            cirq.CNOT(qubit[6], qubit[5]).with_tags('nnn'), #nnn
            cirq.CNOT(qubit[5], qubit[6]).with_tags('nnn'), #nnn
            nFT_SWAP(qubit[6], qubit[7]),
            cirq.CNOT(qubit[8], qubit[7]),
            cirq.CNOT(qubit[7], qubit[8]),
            cirq.CNOT(qubit[9], qubit[8]).with_tags('nnn'), #nnn
            cirq.CNOT(qubit[8], qubit[9]).with_tags('nnn'), #nnn
            FT_SWAP(qubit[0], qubit[11], qubit[12]),
            cirq.CNOT(qubit[9], qubit[11]).with_tags('nnn'), #nnn
            
        ]
    )
    
    return x_readout
 
    
# reduced number of FT SWAPs
def red_correction_circuit_Z(qubit):
    if len(qubit) != 13:
        raise ValueError("Expected 13 qubits, got {}".format(len(qubit)))
    
    # create X-stabilizer readout circuit
    z_correction = cirq.Circuit()

    # add Hadamard gates to the ancilla qubits
    z_correction.append([cirq.H(qubit[4]), cirq.H(qubit[9]), cirq.H(qubit[10])])
    
    # correction
    z_correction.append(
        [   
            nFT_SWAP(qubit[0], qubit[8]), #faulty SWAP between ancilla and data qubit
            cirq.CCZ(qubit[0], qubit[9], qubit[10]),
            nFT_SWAP(qubit[0], qubit[8]), #SWAP back
            nFT_SWAP(qubit[4], qubit[5]),
            nFT_SWAP(qubit[5], qubit[6]),  
            nFT_SWAP(qubit[8], qubit[9]),  
            nFT_SWAP(qubit[9], qubit[10]),  
            cirq.CCZ(qubit[6], qubit[7], qubit[8]),    
            nFT_SWAP(qubit[6], qubit[7]), 
            nFT_SWAP(qubit[8], qubit[9]), 
            FT_SWAP(qubit[0], qubit[5], qubit[6]),  
            cirq.CCZ(qubit[6], qubit[7], qubit[8]),   
        ]
    )
    
    #reset the ancillas
    z_correction.append([cirq.reset(qubit[7]), cirq.reset(qubit[8]), cirq.reset(qubit[9])])
    
    
    return z_correction
    

# reduced number of FT SWAPs
def red_readout_circuit_Z(qubit):
    if len(qubit) != 13:
        raise ValueError("Expected 13 qubits, got {}".format(len(qubit)))
    
    # create X-stabilizer readout circuit
    z_readout = cirq.Circuit()
    
    
    # readout Z_1
    z_readout.append(
        [
            cirq.CNOT(qubit[7], qubit[6]),
            cirq.CNOT(qubit[6], qubit[7]),
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            FT_SWAP(qubit[0], qubit[4], qubit[5]),
            FT_SWAP(qubit[0], qubit[3], qubit[4]),
            cirq.CNOT(qubit[2], qubit[3]).with_tags('nnn'), #nnn
            FT_SWAP(qubit[0], qubit[1], qubit[2]),
            cirq.CNOT(qubit[2], qubit[3]).with_tags('nnn'), #nnn
            FT_SWAP(qubit[0], qubit[3], qubit[4]),
            cirq.CNOT(qubit[5], qubit[4]),
            cirq.CNOT(qubit[3], qubit[4]),
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            cirq.CNOT(qubit[5], qubit[4]),        
        ]
    )
    
    # readout Z_2 
    z_readout.append(
        [
            cirq.CNOT(qubit[6], qubit[8]).with_tags('nnn'), #nnn
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            FT_SWAP(qubit[0], qubit[9], qubit[10]),
            cirq.CNOT(qubit[9], qubit[8]).with_tags('nnn'), #nnn
            cirq.CNOT(qubit[6], qubit[8]).with_tags('nnn'), #nnn
            FT_SWAP(qubit[0], qubit[8], qubit[9]),
            nFT_SWAP(qubit[9], qubit[10]),
            cirq.CNOT(qubit[11], qubit[10]),
            FT_SWAP(qubit[0], qubit[11], qubit[12]),
            cirq.CNOT(qubit[11], qubit[10]),
            nFT_SWAP(qubit[9], qubit[10]),
            FT_SWAP(qubit[0], qubit[8], qubit[9]),
            cirq.CNOT(qubit[7], qubit[8]),     
        ]
    )
    
    # readout Z_3
    z_readout.append(
        [
            FT_SWAP(qubit[0], qubit[1], qubit[2]),
            FT_SWAP(qubit[0], qubit[2], qubit[3]),
            cirq.CNOT(qubit[11], qubit[10]),
            FT_SWAP(qubit[0], qubit[9], qubit[10]),
            nFT_SWAP(qubit[8], qubit[9]),
            nFT_SWAP(qubit[7], qubit[8]),
            FT_SWAP(qubit[0], qubit[6], qubit[7]),
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            nFT_SWAP(qubit[4], qubit[5]),
            
            
            cirq.CNOT(qubit[3], qubit[4]),
            FT_SWAP(qubit[0], qubit[2], qubit[3]),
            cirq.CNOT(qubit[3], qubit[4]),
            FT_SWAP(qubit[0], qubit[11], qubit[12]),
            FT_SWAP(qubit[0], qubit[10], qubit[11]),
            
            nFT_SWAP(qubit[4], qubit[5]),
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            nFT_SWAP(qubit[6], qubit[7]),
            FT_SWAP(qubit[0], qubit[7], qubit[8]),
            nFT_SWAP(qubit[8], qubit[9]),
            
            cirq.CNOT(qubit[10], qubit[9]),
            cirq.CNOT(qubit[11], qubit[9]).with_tags('nnn'), #nnn
            
            FT_SWAP(qubit[0], qubit[1], qubit[2]),
            FT_SWAP(qubit[0], qubit[2], qubit[3]),
            FT_SWAP(qubit[0], qubit[3], qubit[4]),
            FT_SWAP(qubit[0], qubit[4], qubit[5]),
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            
            nFT_SWAP(qubit[8], qubit[9]),
            cirq.CNOT(qubit[6], qubit[8]).with_tags('nnn')    #nnn     
        ]
    )
    
    return z_readout    


# reduced number of FT SWAPs
def red_correction_circuit_X(qubit):
    if len(qubit) != 13:
        raise ValueError("Expected 13 qubits, got {}".format(len(qubit)))
    
    # create X-stabilizer readout circuit
    x_correction = cirq.Circuit()

    
    # correction
    x_correction.append(
        [
            cirq.CNOT(qubit[0], qubit[8]), cirq.CNOT(qubit[8], qubit[0]), cirq.CNOT(qubit[0], qubit[8]), #faulty SWAP between ancilla and data qubit
            cirq.CCNOT(qubit[0], qubit[9], qubit[10]),
            cirq.CNOT(qubit[0], qubit[8]), cirq.CNOT(qubit[8], qubit[0]), cirq.CNOT(qubit[0], qubit[8]), #SWAP back
            
            FT_SWAP(qubit[0], qubit[6], qubit[7]),
            
            nFT_SWAP(qubit[3], qubit[4]),
            nFT_SWAP(qubit[4], qubit[5]),
            nFT_SWAP(qubit[5], qubit[6]),
            
            cirq.CCNOT(qubit[6], qubit[8], qubit[7]),
            nFT_SWAP(qubit[8], qubit[9]),
            nFT_SWAP(qubit[6], qubit[7]),
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            
            cirq.CCNOT(qubit[8], qubit[7], qubit[6]),
            
        ]
    )
    
    return x_correction

    
def encoding_circuit(qubit, encoded_state='0'):
    if len(qubit) != 13:
        raise ValueError("Expected 13 qubits, got {}".format(len(qubit)))
    
    if encoded_state not in ['0', '1', '-', '+']:
        raise ValueError("encoded_state must be one of '0', '1', '-', or '+'")
    
    # stabilizer ancillas: [0][1][15]
    # SWAP ancillas: [2][6][10][14]
    encode = cirq.Circuit()
    
    if encoded_state == '1' or encoded_state == '-':
        encode.append([cirq.X(qubit[3]), cirq.X(qubit[6]), cirq.X(qubit[9])])


    for i in [3, 6, 9]:
        encode.append(cirq.H(qubit[i]))
        encode.append([cirq.CNOT(qubit[i], qubit[i+1]), cirq.CNOT(qubit[i+1], qubit[i+2])])
        if encoded_state == '1' or encoded_state == '0':
            encode.append([cirq.H(qubit[i]), cirq.H(qubit[i+1]), cirq.H(qubit[i+2])])
            
            
    #if we encode + or -, we need to swap the qubits in the correct order since we do not apply a transversal Hadamard gate
    if encoded_state == '+' or encoded_state == '-':
        encode.append([
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            FT_SWAP(qubit[0], qubit[8], qubit[9]),
            FT_SWAP(qubit[0], qubit[4], qubit[5]),
            FT_SWAP(qubit[0], qubit[6], qubit[7]),
            FT_SWAP(qubit[0], qubit[9], qubit[10]),
            FT_SWAP(qubit[0], qubit[7], qubit[8]),
            FT_SWAP(qubit[0], qubit[6], qubit[7]),
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            FT_SWAP(qubit[0], qubit[8], qubit[9])
        ])
         
    return encode


def readout_circuit_X(qubit):
    if len(qubit) != 13:
        raise ValueError("Expected 13 qubits, got {}".format(len(qubit)))
    
    # create X-stabilizer readout circuit
    x_readout = cirq.Circuit()

    # add Hadamard gates to the ancilla qubits
    x_readout.append([cirq.H(qubit[1]), cirq.H(qubit[2]), cirq.H(qubit[12])])
    
    # readout X_1 -> FT, tested
    x_readout.append(
        [
            cirq.CNOT(qubit[2], qubit[3]).with_tags('nnn'), #nnn
            FT_SWAP(qubit[0], qubit[2], qubit[3]),
            cirq.CNOT(qubit[3], qubit[4]),
            FT_SWAP(qubit[0], qubit[3], qubit[4]),
            FT_SWAP(qubit[0], qubit[4], qubit[5]),
            cirq.CNOT(qubit[5], qubit[6]).with_tags('nnn'), #nnn
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            cirq.CNOT(qubit[6], qubit[7]).with_tags('nnn'),
            FT_SWAP(qubit[0], qubit[6], qubit[7]),
            FT_SWAP(qubit[0], qubit[7], qubit[8]),
            cirq.CNOT(qubit[8], qubit[9]).with_tags('nnn'), #nnn
            FT_SWAP(qubit[0], qubit[8], qubit[9]),
            cirq.CNOT(qubit[9], qubit[10])            
        ]
    )
    
    # readout X_2
    x_readout.append(
        [
            cirq.CNOT(qubit[12], qubit[11]).with_tags('nnn'), #nnn
            FT_SWAP(qubit[0], qubit[11], qubit[12]),
            cirq.CNOT(qubit[11], qubit[10]), 
            FT_SWAP(qubit[0], qubit[10], qubit[11]),
            FT_SWAP(qubit[0], qubit[9], qubit[10]), 
            FT_SWAP(qubit[0], qubit[8], qubit[9]),   
            cirq.CNOT(qubit[8], qubit[7]),  
            FT_SWAP(qubit[0], qubit[7], qubit[8]),  
            cirq.CNOT(qubit[7], qubit[6]),  
            FT_SWAP(qubit[0], qubit[6], qubit[7]),  
            FT_SWAP(qubit[0], qubit[5], qubit[6]),   
            cirq.CNOT(qubit[5], qubit[4]), 
            cirq.CNOT(qubit[5], qubit[3]).with_tags('nnn'),  #nnn
        ]
    )
    
    # readout X_3
    x_readout.append(
        [
            cirq.CNOT(qubit[1], qubit[2]),
            FT_SWAP(qubit[0], qubit[1], qubit[2]), 
            FT_SWAP(qubit[0], qubit[2], qubit[3]), 
            cirq.CNOT(qubit[3], qubit[4]),
            FT_SWAP(qubit[0], qubit[3], qubit[4]), 
            FT_SWAP(qubit[0], qubit[4], qubit[5]), 
            cirq.CNOT(qubit[5], qubit[6]).with_tags('nnn'), #nnn
            FT_SWAP(qubit[0], qubit[5], qubit[6]), 
            FT_SWAP(qubit[0], qubit[6], qubit[7]), 
            cirq.CNOT(qubit[7], qubit[8]),
            FT_SWAP(qubit[0], qubit[7], qubit[8]), 
            cirq.CNOT(qubit[8], qubit[9]).with_tags('nnn'), #nnn
            FT_SWAP(qubit[0], qubit[8], qubit[9]), 
            FT_SWAP(qubit[0], qubit[11], qubit[12]), 
            cirq.CNOT(qubit[9], qubit[11]).with_tags('nnn'), #nnn
        ]
    )
    
    return x_readout


def correction_circuit_Z(qubit):
    if len(qubit) != 13:
        raise ValueError("Expected 13 qubits, got {}".format(len(qubit)))
    
    # create X-stabilizer readout circuit
    z_correction = cirq.Circuit()

    # add Hadamard gates to the ancilla qubits
    z_correction.append([cirq.H(qubit[4]), cirq.H(qubit[9]), cirq.H(qubit[10])])
    
    # correction
    z_correction.append(
        [
            cirq.CNOT(qubit[8], qubit[0]), cirq.CNOT(qubit[0], qubit[8]), cirq.CNOT(qubit[8], qubit[0]), #faulty SWAP between ancilla and data qubit
            cirq.CCZ(qubit[0], qubit[9], qubit[10]),
            cirq.CNOT(qubit[8], qubit[0]), cirq.CNOT(qubit[0], qubit[8]), cirq.CNOT(qubit[8], qubit[0]), #SWAP back
            FT_SWAP(qubit[0], qubit[4], qubit[5]),
            FT_SWAP(qubit[0], qubit[5], qubit[6]),  
            FT_SWAP(qubit[0], qubit[8], qubit[9]),  
            FT_SWAP(qubit[0], qubit[9], qubit[10]),  
            cirq.CCZ(qubit[6], qubit[7], qubit[8]),    
            FT_SWAP(qubit[0], qubit[6], qubit[7]), 
            FT_SWAP(qubit[0], qubit[8], qubit[9]),  
            FT_SWAP(qubit[0], qubit[5], qubit[6]),  
            cirq.CCZ(qubit[6], qubit[7], qubit[8]),   
        ]
    )
    
    #reset the ancillas
    z_correction.append([cirq.reset(qubit[7]), cirq.reset(qubit[8]), cirq.reset(qubit[9])])
    
    
    return z_correction



def readout_circuit_Z(qubit):
    if len(qubit) != 13:
        raise ValueError("Expected 13 qubits, got {}".format(len(qubit)))
    
    # create X-stabilizer readout circuit
    z_readout = cirq.Circuit()
    
    
    # readout Z_1
    z_readout.append(
        [
            cirq.CNOT(qubit[6], qubit[7]),
            FT_SWAP(qubit[0], qubit[6], qubit[7]),
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            FT_SWAP(qubit[0], qubit[4], qubit[5]),
            FT_SWAP(qubit[0], qubit[3], qubit[4]),
            cirq.CNOT(qubit[2], qubit[3]).with_tags('nnn'), #nnn
            FT_SWAP(qubit[0], qubit[1], qubit[2]),
            cirq.CNOT(qubit[2], qubit[3]).with_tags('nnn'), #nnn
            FT_SWAP(qubit[0], qubit[3], qubit[4]),
            cirq.CNOT(qubit[5], qubit[4]),
            cirq.CNOT(qubit[3], qubit[4]),
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            cirq.CNOT(qubit[5], qubit[4]),        
        ]
    )
    
    # readout Z_2 
    z_readout.append(
        [
            cirq.CNOT(qubit[6], qubit[8]).with_tags('nnn'), #nnn
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            FT_SWAP(qubit[0], qubit[9], qubit[10]),
            cirq.CNOT(qubit[9], qubit[8]).with_tags('nnn'), #nnn
            cirq.CNOT(qubit[6], qubit[8]).with_tags('nnn'), #nnn
            FT_SWAP(qubit[0], qubit[8], qubit[9]),
            FT_SWAP(qubit[0], qubit[9], qubit[10]),
            cirq.CNOT(qubit[11], qubit[10]),
            FT_SWAP(qubit[0], qubit[11], qubit[12]),
            cirq.CNOT(qubit[11], qubit[10]),
            FT_SWAP(qubit[0], qubit[9], qubit[10]),
            FT_SWAP(qubit[0], qubit[8], qubit[9]),
            cirq.CNOT(qubit[7], qubit[8]),     
        ]
    )
    
    # readout Z_3
    z_readout.append(
        [
            FT_SWAP(qubit[0], qubit[1], qubit[2]),
            FT_SWAP(qubit[0], qubit[2], qubit[3]),
            cirq.CNOT(qubit[11], qubit[10]),
            FT_SWAP(qubit[0], qubit[9], qubit[10]),
            FT_SWAP(qubit[0], qubit[8], qubit[9]),
            FT_SWAP(qubit[0], qubit[7], qubit[8]),
            FT_SWAP(qubit[0], qubit[6], qubit[7]),
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            FT_SWAP(qubit[0], qubit[4], qubit[5]),
            
            
            cirq.CNOT(qubit[3], qubit[4]),
            FT_SWAP(qubit[0], qubit[2], qubit[3]),
            cirq.CNOT(qubit[3], qubit[4]),
            FT_SWAP(qubit[0], qubit[11], qubit[12]),
            FT_SWAP(qubit[0], qubit[10], qubit[11]),
            
            FT_SWAP(qubit[0], qubit[4], qubit[5]),
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            FT_SWAP(qubit[0], qubit[6], qubit[7]),
            FT_SWAP(qubit[0], qubit[7], qubit[8]),
            FT_SWAP(qubit[0], qubit[8], qubit[9]),
            
            cirq.CNOT(qubit[10], qubit[9]),
            cirq.CNOT(qubit[11], qubit[9]).with_tags('nnn'), #nnn
            
            FT_SWAP(qubit[0], qubit[1], qubit[2]),
            FT_SWAP(qubit[0], qubit[2], qubit[3]),
            FT_SWAP(qubit[0], qubit[3], qubit[4]),
            FT_SWAP(qubit[0], qubit[4], qubit[5]),
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            
            FT_SWAP(qubit[0], qubit[8], qubit[9]),
            cirq.CNOT(qubit[6], qubit[8]).with_tags('nnn')    #nnn     
        ]
    )
    
    return z_readout


def correction_circuit_X(qubit):
    if len(qubit) != 13:
        raise ValueError("Expected 13 qubits, got {}".format(len(qubit)))
    
    # create X-stabilizer readout circuit
    x_correction = cirq.Circuit()

    
    # correction
    x_correction.append(
        [
            cirq.CNOT(qubit[0], qubit[8]), cirq.CNOT(qubit[8], qubit[0]), cirq.CNOT(qubit[0], qubit[8]), #faulty SWAP between ancilla and data qubit
            cirq.CCNOT(qubit[0], qubit[9], qubit[10]),
            cirq.CNOT(qubit[0], qubit[8]), cirq.CNOT(qubit[8], qubit[0]), cirq.CNOT(qubit[0], qubit[8]), #SWAP back
            
            FT_SWAP(qubit[0], qubit[6], qubit[7]),
            
            FT_SWAP(qubit[0], qubit[3], qubit[4]),
            FT_SWAP(qubit[0], qubit[4], qubit[5]),
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            
            cirq.CCNOT(qubit[6], qubit[8], qubit[7]),
            FT_SWAP(qubit[0], qubit[8], qubit[9]),
            FT_SWAP(qubit[0], qubit[6], qubit[7]),
            FT_SWAP(qubit[0], qubit[5], qubit[6]),
            
            cirq.CCNOT(qubit[8], qubit[7], qubit[6]),
            
        ]
    )
    
    return x_correction


#checked
def classical_post_processing_circuit(qubit): 
    if len(qubit) != 13:
        raise ValueError("Expected 13 qubits, got {}".format(len(qubit)))
    
    post_proc_circuit = cirq.Circuit()
    
    # X stabilizer readout
    post_proc_circuit.append([cirq.reset(qubit[7]), cirq.reset(qubit[8]), cirq.reset(qubit[9])])
    post_proc_circuit.append([cirq.H(qubit[7]), cirq.H(qubit[8]), cirq.H(qubit[9])])
    post_proc_circuit.append([cirq.CNOT(qubit[9], qubit[5]), cirq.CNOT(qubit[9], qubit[1]), cirq.CNOT(qubit[9], qubit[3]), cirq.CNOT(qubit[9], qubit[6]), cirq.CNOT(qubit[9], qubit[11]), cirq.CNOT(qubit[9], qubit[12])])
    post_proc_circuit.append([cirq.CNOT(qubit[8], qubit[1]), cirq.CNOT(qubit[8], qubit[2]), cirq.CNOT(qubit[8], qubit[6]), cirq.CNOT(qubit[8], qubit[4]), cirq.CNOT(qubit[8], qubit[12]), cirq.CNOT(qubit[8], qubit[10])])
    post_proc_circuit.append([cirq.CNOT(qubit[7], qubit[5]), cirq.CNOT(qubit[7], qubit[2]), cirq.CNOT(qubit[7], qubit[3]), cirq.CNOT(qubit[7], qubit[4]), cirq.CNOT(qubit[7], qubit[11]), cirq.CNOT(qubit[7], qubit[10])])
    
    # correction
    post_proc_circuit.append([cirq.H(qubit[7]), cirq.H(qubit[8]), cirq.H(qubit[9])])
    post_proc_circuit.append([cirq.CCZ(qubit[6], qubit[8], qubit[9]), cirq.CCZ(qubit[7], qubit[8], qubit[10]), cirq.CCZ(qubit[5], qubit[7], qubit[9])])
    post_proc_circuit.append([cirq.reset(qubit[7]), cirq.reset(qubit[8]), cirq.reset(qubit[9])])
    
    # Z-stabilizer readout
    post_proc_circuit.append([cirq.CNOT(qubit[5], qubit[9]), cirq.CNOT(qubit[1], qubit[9]), cirq.CNOT(qubit[2], qubit[9]), cirq.CNOT(qubit[3], qubit[9]), cirq.CNOT(qubit[6], qubit[9]), cirq.CNOT(qubit[4], qubit[9])])
    post_proc_circuit.append([cirq.CNOT(qubit[3], qubit[8]), cirq.CNOT(qubit[6], qubit[8]), cirq.CNOT(qubit[4], qubit[8]), cirq.CNOT(qubit[11], qubit[8]), cirq.CNOT(qubit[12], qubit[8]), cirq.CNOT(qubit[10], qubit[8])])
    post_proc_circuit.append([cirq.CNOT(qubit[5], qubit[7]), cirq.CNOT(qubit[1], qubit[7]), cirq.CNOT(qubit[2], qubit[7]), cirq.CNOT(qubit[11], qubit[7]), cirq.CNOT(qubit[12], qubit[7]), cirq.CNOT(qubit[10], qubit[7])])
    
    # correction
    post_proc_circuit.append([cirq.CCNOT(qubit[9], qubit[8], qubit[6]), cirq.CCNOT(qubit[7], qubit[8], qubit[10]), cirq.CCNOT(qubit[9], qubit[7], qubit[5])])

    return post_proc_circuit
  
    

### TEST FAULT TOLERANCE ###

def get_stabilizer_logicals(data_positions):
    qubit = [cirq.LineQubit(i) for i in range(13)]
    stabilizers = {
        'Z_1': cirq.PauliString({qubit[data_positions[0]]: cirq.Z, qubit[data_positions[1]]: cirq.Z, qubit[data_positions[2]]: cirq.Z, qubit[data_positions[3]]: cirq.Z, qubit[data_positions[4]]: cirq.Z, qubit[data_positions[5]]: cirq.Z}),
        'Z_2': cirq.PauliString({qubit[data_positions[3]]: cirq.Z, qubit[data_positions[4]]: cirq.Z, qubit[data_positions[5]]: cirq.Z, qubit[data_positions[6]]: cirq.Z, qubit[data_positions[7]]: cirq.Z, qubit[data_positions[8]]: cirq.Z}),
        'X_1': cirq.PauliString({qubit[data_positions[0]]: cirq.X, qubit[data_positions[1]]: cirq.X, qubit[data_positions[3]]: cirq.X, qubit[data_positions[4]]: cirq.X, qubit[data_positions[6]]: cirq.X, qubit[data_positions[7]]: cirq.X}),
        'X_2': cirq.PauliString({qubit[data_positions[1]]: cirq.X, qubit[data_positions[2]]: cirq.X, qubit[data_positions[4]]: cirq.X, qubit[data_positions[5]]: cirq.X, qubit[data_positions[7]]: cirq.X, qubit[data_positions[8]]: cirq.X})}
    
    logicals = {
        'X_L': cirq.PauliString({qubit[data_positions[0]]: cirq.X, qubit[data_positions[3]]: cirq.X, qubit[data_positions[6]]: cirq.X}),
        'Z_L': cirq.PauliString({qubit[data_positions[0]]: cirq.Z, qubit[data_positions[1]]: cirq.Z, qubit[data_positions[2]]: cirq.Z})}
    
    return stabilizers | logicals


def circuits_with_single_pauli_errors(base_circuit):
    """Return circuits with Pauli errors inserted after multi-qubit gates.

    Args:
        base_circuit: The circuit that should be decorated with mid-circuit errors.

    Returns:
        A list of new circuits. Each circuit contains the original operations with an
        extra moment that inserts all combinations of X, Y, and Z errors on every subset
        of at least two qubits touched by a multi-qubit gate, placed immediately after
        that gate.
    """

    error_circuits = []
    qubits = list(base_circuit.all_qubits())

    if not qubits:
        return error_circuits

    error_generators = (cirq.X, cirq.Y, cirq.Z)
    num_moments = len(base_circuit)
    for moment_index, moment in enumerate(base_circuit):
        for op in moment.operations:
            touched_qubits = op.qubits
            if len(touched_qubits) <= 1:
                continue
            qubit_list = list(touched_qubits)

            for subset_size in range(2, len(qubit_list) + 1):
                for qubit_subset in combinations(qubit_list, subset_size):
                    for error_ops in product(error_generators, repeat=subset_size):
                        new_circuit = cirq.Circuit(base_circuit)
                        pauli_ops = [gate(qubit) for gate, qubit in zip(error_ops, qubit_subset)]
                        insertion_index = min(moment_index + 1, num_moments)
                        new_circuit.insert(insertion_index, pauli_ops)
                        error_circuits.append(new_circuit)
                
    print(f"Generated {len(error_circuits)} circuits containing all possible weight>1 mid-circuit Pauli errors")

    return error_circuits


def simulate_stabilizer_expectations(circuit, data_positions, display = False):

    # Collect all qubits that need to be included in the simulation
    
    circuit_qubits = sorted(circuit.all_qubits())

    qubit_set = set(circuit_qubits)
    qubit_set.update(circuit.all_qubits())

    qubit_order = sorted(qubit_set)
    

    # Simulate the circuit
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    state = result.final_state_vector

    # Build index_map consistent with qubit_order
    index_map = {q: i for i, q in enumerate(qubit_order)}
    
    # get the stabilizers and logical operators
    operators = get_stabilizer_logicals(data_positions)

    # Compute expectation values
    expectation_vals = {
        name: float(np.round(np.real_if_close(pauli_string.expectation_from_state_vector(state, index_map)), 12))
        for name, pauli_string in operators.items()
    }

    # Optional printout
    if display:
        def is_logical(name: str) -> bool:
            lowered = name.lower()
            return lowered.endswith('_l') or lowered.startswith('l_') or 'logical' in lowered

        stabilizer_names = sorted(name for name in expectation_vals if not is_logical(name))
        logical_names = sorted(name for name in expectation_vals if is_logical(name))

        if stabilizer_names:
            print("Stabilizer expectations:")
            for name in stabilizer_names:
                print(f"  {name} ({operators[name]}): {expectation_vals[name]}")

        if logical_names:
            print("Logical operator expectations:")
            for name in logical_names:
                print(f"  {name} ({operators[name]}): {expectation_vals[name]}")

    return expectation_vals, state


def classical_error_correction(circuit, data_positions, display = False):
    qubit = [cirq.LineQubit(i) for i in range(13)]
    exp_vals, state = simulate_stabilizer_expectations(circuit, data_positions, display = False)
    
    if exp_vals['X_1'] == 1.0 and exp_vals['X_2'] == 1.0 and exp_vals['Z_1'] == 1.0 and exp_vals['Z_2'] == 1.0:
        return exp_vals, state
    
    if data_positions is None:
        data_positions = list(range(13))
        
    
    if exp_vals['X_1'] == -1.0 and exp_vals['X_2'] == 1.0:
        circuit.append(cirq.Z(qubit[data_positions[0]]))
    elif exp_vals['X_1'] == 1.0 and exp_vals['X_2'] == -1.0:
        circuit.append(cirq.Z(qubit[data_positions[2]]))
    elif exp_vals['X_1'] == -1.0 and exp_vals['X_2'] == -1.0:
        circuit.append(cirq.Z(qubit[data_positions[1]]))
        
    if exp_vals['Z_1'] == -1.0 and exp_vals['Z_2'] == 1.0:
        circuit.append(cirq.X(qubit[data_positions[0]]))
    elif exp_vals['Z_1'] == 1.0 and exp_vals['Z_2'] == -1.0:
        circuit.append(cirq.X(qubit[data_positions[6]]))
    elif exp_vals['Z_1'] == -1.0 and exp_vals['Z_2'] == -1.0:
        circuit.append(cirq.X(qubit[data_positions[3]])) 
        
    exp_vals, state = simulate_stabilizer_expectations(circuit, data_positions, display = False)
     
    if display:    
        print(f'corrected circuit: \n {circuit}')
        print(f'corrected expectation values {exp_vals}')
     
    return exp_vals, state
        

def check_fault_tolerance_BC(circuit, data_positions, display = False):
    qubit = [cirq.LineQubit(i) for i in range(13)]
    #get the error-free expectation values
    error_free_exp, error_free_state = simulate_stabilizer_expectations(circuit, data_positions, display = False)  
    print("\nExpectation values:")
    print("-" * 40)
    for name, val in error_free_exp.items():
        print(f"{name:<20s}: {val:>8.4f}")
        print("-" * 40)
    
    error_circuits = circuits_with_single_pauli_errors(circuit)
    
    # test with post processing circuit
    combined_circuits = [c + classical_post_processing_circuit(qubit) for c in error_circuits]
    
    
    faulty_exp = []
    mismatched_indices = []
    for i in range(len(error_circuits)):
        #faulty_exp_val, faulty_state = classical_error_correction(error_circuits[i], data_positions)
        faulty_exp_val, faulty_state  = simulate_stabilizer_expectations(combined_circuits[i], data_positions, display = False)
        faulty_exp.append(faulty_exp_val)
        expectations_match = all(
            np.isclose(faulty_exp_val[key], error_free_exp[key], atol=1e-6)
            for key in error_free_exp
        )
        if not expectations_match:
            mismatched_indices.append(i)
            
            
    if not mismatched_indices:
        print(f"The circuit is fault tolerant: {len(error_circuits)-len(mismatched_indices)} of {len(error_circuits)} error circuits did not result in an incorrectable error.")
    else:
        print(f"The circuit is not fault tolerant. {len(mismatched_indices)} of {len(error_circuits)} error circuits resulted in an incorrectable error.")
        if display:
            def is_stabilizer(name: str) -> bool:
                parts = name.split("_", 1)
                return len(parts) == 2 and parts[1].isdigit()
            def is_logical(name: str) -> bool:
                lowered = name.lower()
                return lowered.endswith('_l') or lowered.startswith('l_') or 'logical' in lowered

            stabilizer_names = sorted(name for name in error_free_exp if is_stabilizer(name))
            logical_names = sorted(name for name in error_free_exp if is_logical(name))

            display_columns = stabilizer_names + logical_names

            if display_columns:
                headers = ["error_index"] + display_columns
                rows = []
                for index in mismatched_indices:
                    row = [str(index)]
                    for observable in display_columns:
                        value = faulty_exp[index].get(observable, "")
                        if isinstance(value, (float, np.floating)):
                            row.append(f"{value:.2f}")
                        else:
                            row.append(str(value))
                    rows.append(row)

                if rows:
                    col_widths = [
                        max(len(header), *(len(row[i]) for row in rows))
                        for i, header in enumerate(headers)
                    ]

                    header_line = " | ".join(
                        header.ljust(col_widths[i]) for i, header in enumerate(headers)
                    )
                    separator_line = "-+-".join("-" * width for width in col_widths)

                    print("Expectation values for deviating error circuits (stabilizer/logical):")
                    print(header_line)
                    print(separator_line)
                    for row in rows:
                        print(" | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))
            else:
                print("No stabilizer or logical expectation values available for display.")

    return len(mismatched_indices) == 0


def circuit_element_FT_test(circuit, data_positions, encodet_state = '+', display = False):
    qubit = [cirq.LineQubit(i) for i in range(13)]
    enc_circuit = encoding_circuit(qubit, encoded_state=encodet_state)
    test_circuit = enc_circuit + circuit
    check_fault_tolerance_BC(test_circuit, data_positions, display = display)
    
    
def full_BC_fT_test():
    qubit = [cirq.LineQubit(i) for i in range(13)]
    test_circuit = cirq.Circuit()
    test_circuit += red_readout_circuit_X(qubit)
    test_circuit += red_correction_circuit_Z(qubit)
    test_circuit += red_readout_circuit_Z(qubit)
    test_circuit += red_correction_circuit_X(qubit)
    data_positions = [5, 1, 2, 3, 6, 4, 11, 12, 10] # after X correction/entire circuit
    
    start_time = time.time()
    circuit_element_FT_test(test_circuit, data_positions, encodet_state = '+', display = True)
    circuit_element_FT_test(test_circuit, data_positions, encodet_state = '0', display = True)
    end_time = time.time()
    runtime = (end_time - start_time)/60
    print(f"Runtime: {runtime:.2f} minutes")
    

### HELPFUL FUNCTIONS ###

def save_cirq_pdf(circuit, path="circuit.pdf", transpose=False):
    """Render Cirq's text diagram into a vector PDF with minimal code."""
    # Get compact (Unicode) text diagram; transpose=True for vertical layout
    diag = circuit.to_text_diagram(use_unicode_characters=True, transpose=transpose)
    lines = diag.splitlines()

    # Simple figure sizing from content: width by longest line, height by #lines
    fig_w = max(6, 0.12 * max((len(l) for l in lines), default=1))   # inches
    fig_h = max(2, 0.22 * len(lines) + 0.5)                          # inches

    plt.figure(figsize=(fig_w, fig_h))
    plt.axis("off")
    plt.text(0, 1, diag, family="DejaVu Sans Mono", fontsize=8, va="top")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def get_p_from_Fidelity(num_qubits, fidelity):
    """Convert fidelity to depolarizing error rate p."""
    if not np.all((fidelity >= 0) & (fidelity <= 1)):
        raise ValueError("Fidelity must be between 0 and 1.")
    if num_qubits < 1:
        raise ValueError("Number of qubits must be at least 1.")

    d = 2**num_qubits
    p = (d + 1) / d * (1 - fidelity)

    return p


def get_Fidelity_from_p(num_qubits, p):
    """Convert  depolarizing error rate p to fidelity."""
    if not np.all((p >= 0) & (p <= 1)):
        raise ValueError("Error rate must be between 0 and 1.")
    if num_qubits < 1:
        raise ValueError("Number of qubits must be at least 1.")

    d = 2**num_qubits
    F = 1 - d / (d + 1) * p

    return F

def _normalize_density_matrix(rho, eps=1e-15):
    """Normalize and enforce Hermiticity/PSD within numerical precision."""
    rho = np.asarray(rho, dtype=np.complex128)
    # Symmetrize to remove numerical anti-Hermitian noise
    rho = 0.5 * (rho + rho.conj().T)
    # Clip tiny negative eigenvalues if present
    w, v = np.linalg.eigh(rho)
    w = np.clip(w, 0.0, None)
    rho = v @ np.diag(w) @ v.conj().T
    # Renormalize trace
    tr = np.trace(rho)
    if np.isclose(tr, 0.0, atol=eps):
        raise ValueError("Density matrix trace is zero; cannot normalize.")
    rho /= tr
    return rho


if __name__ == "__main__":
    #qubit = [cirq.LineQubit(i) for i in range(13)]
    
    ##positions of qubits after each circuit element
    #data_positions = [3, 4, 5, 6, 7, 8, 9, 10, 11] # after encoding
    #data_positions = [1, 2, 3, 5, 6, 7, 8, 12, 11] # after X stabilizer readout
    #data_positions = [1, 2, 3, 4, 6, 5, 10, 12, 11] # after Z correction 
    #data_positions = [6, 1, 2, 4, 7, 5, 11, 12, 10] # after Z stabilizer readout
    #data_positions = [5, 1, 2, 3, 6, 4, 11, 12, 10] # after X correction/entire circuit

    #test_circuit += red_readout_circuit_X(qubit)
    #test_circuit += red_correction_circuit_Z(qubit)
    #test_circuit += red_readout_circuit_Z(qubit)
    #test_circuit += red_correction_circuit_X(qubit)
    
    ## here the FT of the circuit can be tested
    #start_time = time.time()
    #circuit_element_FT_test(test_circuit, data_positions, encodet_state = '+', display = True)
    #circuit_element_FT_test(test_circuit, data_positions, encodet_state = '0', display = True)
    #end_time = time.time()
    #runtime = (end_time - start_time)/60
    #print(f"Runtime: {runtime:.2f} minutes")
    
    #full_BC_fT_test()
    
    
    ## NOISY SIMULATION ##
    F_CZ_NN = 0.9975
    F_CZ_NNN = 0.9861
    F_CCZ = 0.9742
    
    err_2q_nn = get_p_from_Fidelity(2, F_CZ_NN) # nearest-neighbor two-qubit gate error rate
    err_2q_nnn = get_p_from_Fidelity(2, F_CZ_NNN) # next-nearest-neighbor two-qubit gate error rate
    err_3q = get_p_from_Fidelity(3, F_CCZ) # three qubit gate error rate

    
    lam = [10**-0.5, 10**-1.5, 10**-2.5, 10**-3.5]
    print(lam)
    
    for l in lam:
        start_time = time.time() 
        noisy_density_matrix_simulation(l*err_2q_nn, l*err_2q_nnn, l*err_3q, init_state = '0', path="./data/logical_error_rates_0.csv")
        end_time = time.time()
        runtime = (end_time - start_time)/60
        print(f"Runtime state |0>, lambda = {l}: {runtime:.2f} minutes")
    
    
        start_time = time.time() 
        noisy_density_matrix_simulation(l*err_2q_nn, l*err_2q_nnn, l*err_3q, init_state = '+', path="./data/logical_error_rates_+.csv")
        end_time = time.time()
        runtime = (end_time - start_time)/60
        print(f"Runtime state |+>, lambda = {l}: {runtime:.2f} minutes")


    

    
    

    
    
    


    
