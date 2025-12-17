import numpy as np
import qutip as qt
import PulseHamiltonian
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import time
from scipy import integrate


def params_dict_to_array(params_dict: dict):
    values = params_dict.values()
    params_array = np.array(list(values))
    return params_array

def params_array_to_dict(params_array, keys):
    params_dict = {key: params_array[i] for i, key in enumerate(keys)}
    return params_dict


def renormalize_phases(phases):  
    counter = 0
    renormalized_phases = [phases[0]]
    for i in range(len(phases) - 1):
        if phases[i + 1] - phases[i] > 1:
            counter += -1
        elif phases[i + 1] - phases[i] > -1:
            counter += 0
        else:
            counter += 1
        renormalized_phases.append(phases[i + 1] + 2 * counter)
    phases = np.array(renormalized_phases)  
    return phases

#function that gives you the end popupation and phase for all four states
def Final_Pop_and_Phases(H, t, params):
    t = np.linspace(0, params['tau'], 1000)
    psi = [qt.tensor(qt.basis(4, i), qt.basis(4, j), qt.basis(4, k)) for i in [0, 1] for j in [0, 1] for k in [0, 1]]
    result = [qt.sesolve(H, psi[i], t, e_ops = [], args=params, options=qt.Options(store_final_state=True)) for i in range(8)]
    state = [result[i].final_state for i in range(8)]
    pops = [qt.expect(psi[i].proj(), state[i]) for i in range(8)]
    phases = [np.angle(psi[i].dag()*state[i]) for i in range(8)]
    return pops[1:], phases[1:]


def Intermediate_Pop_and_Phases(H, t, params):
    psi = [qt.tensor(qt.basis(4, i), qt.basis(4, j), qt.basis(4, k)) for i in [0, 1] for j in [0, 1] for k in [0, 1]]
    results = [qt.sesolve(H, psi[i], t, e_ops = [], args=params, options=qt.Options(store_final_state=True, store_states = True, normalize_output=False)) for i in range(8)]
    coeffs = [[(psi[i].dag()*j) for j in results[i].states] for i in range(8)]
    pops = [qt.expect(psi[i].proj(), results[i].states) for i in range(8)]
    phases = [np.angle(coeffs[i]) for i in range(8)]
    phases = [renormalize_phases(phases[i]/np.pi) for i in range(8)]
    return pops, phases

#Bell state fidelity with Single Qubit Rotations for a CCZ
def Bell_state_Fidelity_SQR(H, params, t):
    #initial state
    psi = (1/(2*np.sqrt(2)) * (qt.tensor(qt.basis(4, 0), qt.basis(4, 0), qt.basis(4, 0)) + 
                              qt.tensor(qt.basis(4, 0), qt.basis(4, 0), qt.basis(4, 1)) +
                              qt.tensor(qt.basis(4, 0), qt.basis(4, 1), qt.basis(4, 0)) +
                              qt.tensor(qt.basis(4, 0), qt.basis(4, 1), qt.basis(4, 1)) +
                              qt.tensor(qt.basis(4, 1), qt.basis(4, 0), qt.basis(4, 0)) +
                              qt.tensor(qt.basis(4, 1), qt.basis(4, 0), qt.basis(4, 1)) +
                              qt.tensor(qt.basis(4, 1), qt.basis(4, 1), qt.basis(4, 0)) +
                              qt.tensor(qt.basis(4, 1), qt.basis(4, 1), qt.basis(4, 1))))
    
    #auxiliary state
    aux = [qt.tensor(qt.basis(4, i), qt.basis(4, j), qt.basis(4, k)) for i in [0, 1] for j in [0, 1] for k in [0, 1]]
    
    if 'gamma' in params and params['gamma'] !=0 :
        res = qt.sesolve(H, psi, t, e_ops=None, args=params, options={'store_final_state': True, 'normalize_output': False, 'atol' : 1e-10, 'rtol' : 1e-8})
    else:
        res = qt.sesolve(H, psi, t, e_ops=None, args=params, options={'store_final_state': True, 'normalize_output': True, 'atol' : 1e-10, 'rtol' : 1e-8})
        
    state = res.final_state
    phase = [np.angle(aux[i].dag()*state) for i in range(8)]
    
    #Single Qubit Rotation
    Rot100 = qt.tensor(qt.basis(4, 0)).proj() + np.exp(-1j*phase[4])*qt.tensor(qt.basis(4, 1)).proj() + qt.tensor(qt.basis(4, 2)).proj() + qt.tensor(qt.basis(4, 3)).proj()
    Rot010 = qt.tensor(qt.basis(4, 0)).proj() + np.exp(-1j*phase[2])*qt.tensor(qt.basis(4, 1)).proj() + qt.tensor(qt.basis(4, 2)).proj() + qt.tensor(qt.basis(4, 3)).proj()
    Rot001 = qt.tensor(qt.basis(4, 0)).proj() + np.exp(-1j*phase[1])*qt.tensor(qt.basis(4, 1)).proj() + qt.tensor(qt.basis(4, 2)).proj() + qt.tensor(qt.basis(4, 3)).proj()
    Rot_total = qt.tensor(Rot100, Rot010, Rot001)


    psi_t = (1/(2*np.sqrt(2)) * (qt.tensor(qt.basis(4, 0), qt.basis(4, 0), qt.basis(4, 0)) + 
                                qt.tensor(qt.basis(4, 0), qt.basis(4, 0), qt.basis(4, 1)) +
                                qt.tensor(qt.basis(4, 0), qt.basis(4, 1), qt.basis(4, 0)) +
                                qt.tensor(qt.basis(4, 0), qt.basis(4, 1), qt.basis(4, 1)) +
                                qt.tensor(qt.basis(4, 1), qt.basis(4, 0), qt.basis(4, 0)) +
                                qt.tensor(qt.basis(4, 1), qt.basis(4, 0), qt.basis(4, 1)) +
                                qt.tensor(qt.basis(4, 1), qt.basis(4, 1), qt.basis(4, 0)) -
                                qt.tensor(qt.basis(4, 1), qt.basis(4, 1), qt.basis(4, 1))))
    psi_t = Rot_total.dag()*psi_t

    fidelity = abs(res.final_state.overlap(psi_t))**2
    return fidelity

#Bell state fidelity without Single Qubit Rotations for a CCZ
def Bell_state_Fidelity(H, params, t):
    #initial state
    psi = (1/(2*np.sqrt(2)) * (qt.tensor(qt.basis(4, 0), qt.basis(4, 0), qt.basis(4, 0)) + 
                              qt.tensor(qt.basis(4, 0), qt.basis(4, 0), qt.basis(4, 1)) +
                              qt.tensor(qt.basis(4, 0), qt.basis(4, 1), qt.basis(4, 0)) +
                              qt.tensor(qt.basis(4, 0), qt.basis(4, 1), qt.basis(4, 1)) +
                              qt.tensor(qt.basis(4, 1), qt.basis(4, 0), qt.basis(4, 0)) +
                              qt.tensor(qt.basis(4, 1), qt.basis(4, 0), qt.basis(4, 1)) +
                              qt.tensor(qt.basis(4, 1), qt.basis(4, 1), qt.basis(4, 0)) +
                              qt.tensor(qt.basis(4, 1), qt.basis(4, 1), qt.basis(4, 1))))
    
    if 'gamma' in params and params['gamma'] !=0 :
        res = qt.sesolve(H, psi, t, e_ops=None, args=params, options={'store_final_state': True, 'normalize_output': False, 'atol' : 1e-10, 'rtol' : 1e-8})
    else:
        res = qt.sesolve(H, psi, t, e_ops=None, args=params, options={'store_final_state': True, 'normalize_output': True, 'atol' : 1e-10, 'rtol' : 1e-8})

    psi_t = (1/(2*np.sqrt(2)) * (qt.tensor(qt.basis(4, 0), qt.basis(4, 0), qt.basis(4, 0)) + 
                                qt.tensor(qt.basis(4, 0), qt.basis(4, 0), qt.basis(4, 1)) +
                                qt.tensor(qt.basis(4, 0), qt.basis(4, 1), qt.basis(4, 0)) +
                                qt.tensor(qt.basis(4, 0), qt.basis(4, 1), qt.basis(4, 1)) +
                                qt.tensor(qt.basis(4, 1), qt.basis(4, 0), qt.basis(4, 0)) +
                                qt.tensor(qt.basis(4, 1), qt.basis(4, 0), qt.basis(4, 1)) +
                                qt.tensor(qt.basis(4, 1), qt.basis(4, 1), qt.basis(4, 0)) -
                                qt.tensor(qt.basis(4, 1), qt.basis(4, 1), qt.basis(4, 1))))


    fidelity = abs(res.final_state.overlap(psi_t))**2
    return fidelity

#Bell state fidelity with Single Qubit Rotations for a C_1Z_3 gate
def Bell_state_Fidelity_SQR_C1Z3(H, params, t):
    #initial state
    psi = (1/(2*np.sqrt(2)) * (qt.tensor(qt.basis(4, 0), qt.basis(4, 0), qt.basis(4, 0)) + 
                              qt.tensor(qt.basis(4, 0), qt.basis(4, 0), qt.basis(4, 1)) +
                              qt.tensor(qt.basis(4, 0), qt.basis(4, 1), qt.basis(4, 0)) +
                              qt.tensor(qt.basis(4, 0), qt.basis(4, 1), qt.basis(4, 1)) +
                              qt.tensor(qt.basis(4, 1), qt.basis(4, 0), qt.basis(4, 0)) +
                              qt.tensor(qt.basis(4, 1), qt.basis(4, 0), qt.basis(4, 1)) +
                              qt.tensor(qt.basis(4, 1), qt.basis(4, 1), qt.basis(4, 0)) +
                              qt.tensor(qt.basis(4, 1), qt.basis(4, 1), qt.basis(4, 1))))
    
    #auxiliary state
    aux = [qt.tensor(qt.basis(4, i), qt.basis(4, j), qt.basis(4, k)) for i in [0, 1] for j in [0, 1] for k in [0, 1]]
    
    if 'gamma' in params and params['gamma'] !=0 :
        res = qt.sesolve(H, psi, t, e_ops=None, args=params, options={'store_final_state': True, 'normalize_output': False, 'atol' : 1e-10, 'rtol' : 1e-8})
    else:
        res = qt.sesolve(H, psi, t, e_ops=None, args=params, options={'store_final_state': True, 'normalize_output': True, 'atol' : 1e-10, 'rtol' : 1e-8})
        
    state = res.final_state
    phase = [np.angle(aux[i].dag()*state) for i in range(8)]
    
    #Single Qubit Rotation
    Rot100 = qt.tensor(qt.basis(4, 0)).proj() + np.exp(-1j*phase[4])*qt.tensor(qt.basis(4, 1)).proj() + qt.tensor(qt.basis(4, 2)).proj() + qt.tensor(qt.basis(4, 3)).proj()
    Rot010 = qt.tensor(qt.basis(4, 0)).proj() + np.exp(-1j*phase[2])*qt.tensor(qt.basis(4, 1)).proj() + qt.tensor(qt.basis(4, 2)).proj() + qt.tensor(qt.basis(4, 3)).proj()
    Rot001 = qt.tensor(qt.basis(4, 0)).proj() + np.exp(-1j*phase[1])*qt.tensor(qt.basis(4, 1)).proj() + qt.tensor(qt.basis(4, 2)).proj() + qt.tensor(qt.basis(4, 3)).proj()
    Rot_total = qt.tensor(Rot100, Rot010, Rot001)


    psi_t = (1/(2*np.sqrt(2)) * (qt.tensor(qt.basis(4, 0), qt.basis(4, 0), qt.basis(4, 0)) + 
                                qt.tensor(qt.basis(4, 0), qt.basis(4, 0), qt.basis(4, 1)) +
                                qt.tensor(qt.basis(4, 0), qt.basis(4, 1), qt.basis(4, 0)) +
                                qt.tensor(qt.basis(4, 0), qt.basis(4, 1), qt.basis(4, 1)) +
                                qt.tensor(qt.basis(4, 1), qt.basis(4, 0), qt.basis(4, 0)) -
                                qt.tensor(qt.basis(4, 1), qt.basis(4, 0), qt.basis(4, 1)) +
                                qt.tensor(qt.basis(4, 1), qt.basis(4, 1), qt.basis(4, 0)) -
                                qt.tensor(qt.basis(4, 1), qt.basis(4, 1), qt.basis(4, 1))))
    psi_t = Rot_total.dag()*psi_t

    fidelity = abs(res.final_state.overlap(psi_t))**2
    return fidelity


def Cost_Function_Fidelity(params_opt_vals, H, t, params_fix, params_opt_keys, single_qubit_rotation):
    params_opt = dict(zip(params_opt_keys, params_opt_vals))
    params = params_opt | params_fix
    if single_qubit_rotation is True: 
        fidelity = Bell_state_Fidelity_SQR(H, params, t)
    else: 
        fidelity = Bell_state_Fidelity(H, params, t)
    return 1-fidelity


def Cost_Function_Fidelity_C1Z3(params_opt_vals, H, t, params_fix, params_opt_keys, single_qubit_rotation):
    params_opt = dict(zip(params_opt_keys, params_opt_vals))
    params = params_opt | params_fix
    fidelity = Bell_state_Fidelity_SQR_C1Z3(H, params, t)
    return 1-fidelity



def Optimization_global(H, t, params_fix, params_opt, bounds, Cost_Function, single_qubit_rotation = True, init_guess = None, seed0 = 123):
    params0 = list(params_opt.values())
    params_opt_keys = list(params_opt.keys())
    start = time.time()
    
    res = differential_evolution(Cost_Function, bounds, args = (H, t, params_fix, params_opt_keys, single_qubit_rotation), strategy = "rand1bin", seed = seed0, disp=True, workers = -1, maxiter = 2000, popsize=30, mutation=(0.7, 1.2), tol = 0.0001, init='sobol', x0 = init_guess)
    
    end = time.time()
    print(res)
    elapsed_time = end - start
    print('\nElapsed time: {t:.3f} seconds'.format(t=elapsed_time))
    params_opt_vals = res.x
    params_opt = dict(zip(params_opt_keys, params_opt_vals))
    params = params_opt | params_fix
    pops, phases = Intermediate_Pop_and_Phases(H, t, params)
    return pops, phases, params

    
# analytic formula to calculate the fidelity loss via integral, the general population error and the total phase error and the phase error for each state
def gate_errors(params):
    plus = qt.basis(4, 3)
    minus = qt.basis(4, 2)
    
    #define timesteps
    t = np.linspace(0, params['tau'], 1000)
    
    # check if there is decay
    if 'gamma' in params and params['gamma'] != 0:
        decay = True
    else:
        decay = False
    
    # define pulse functions
    if 'Omega0_2' in params:
        pulse_functions = {'Delta': PulseHamiltonian.Delta_sin, 'Omega_Re': PulseHamiltonian.Omega_sin_double, 'Omega_MW': PulseHamiltonian.Omega_MW_const}
        shape = 'double'
    else:
        pulse_functions = {'Delta': PulseHamiltonian.Delta_sin, 'Omega_Re': PulseHamiltonian.Omega_sin, 'Omega_MW': PulseHamiltonian.Omega_MW_const}
        shape = 'single'
    
    # define Hamiltonian
    H = PulseHamiltonian._get_Hamiltonian(params, pulse_functions, decay)

    #initial state
    psi = (1/(2*np.sqrt(2)) * (qt.tensor(qt.basis(4, 0), qt.basis(4, 0), qt.basis(4, 0)) + 
                          qt.tensor(qt.basis(4, 0), qt.basis(4, 0), qt.basis(4, 1)) +
                          qt.tensor(qt.basis(4, 0), qt.basis(4, 1), qt.basis(4, 0)) +
                          qt.tensor(qt.basis(4, 0), qt.basis(4, 1), qt.basis(4, 1)) +
                          qt.tensor(qt.basis(4, 1), qt.basis(4, 0), qt.basis(4, 0)) +
                          qt.tensor(qt.basis(4, 1), qt.basis(4, 0), qt.basis(4, 1)) +
                          qt.tensor(qt.basis(4, 1), qt.basis(4, 1), qt.basis(4, 0)) +
                          qt.tensor(qt.basis(4, 1), qt.basis(4, 1), qt.basis(4, 1))))
    
    e_ops_psi = [qt.tensor(qt.basis(4, i), qt.basis(4, j), qt.basis(4, k)) for i in [0, 1] for j in [0, 1] for k in [0, 1]]
    

    #state's populations 
    st_p1 = qt.tensor(plus.proj(), qt.qeye(4), qt.qeye(4))
    st_p2 = qt.tensor(qt.qeye(4), plus.proj(), qt.qeye(4))
    st_p3 = qt.tensor(qt.qeye(4), qt.qeye(4), plus.proj())
    st_m1 = qt.tensor(minus.proj(), qt.qeye(4), qt.qeye(4))
    st_m2 = qt.tensor(qt.qeye(4), minus.proj(), qt.qeye(4))
    st_m3 = qt.tensor(qt.qeye(4), qt.qeye(4), minus.proj())
    
    if decay == True:
        res = qt.sesolve(H, psi, t, e_ops=[st_p1, st_p2, st_p3, st_m1, st_m2, st_m3], args=params, options={'store_states': True, 'normalize_output': False, 'atol' : 1e-10, 'rtol' : 1e-8})
    else:
        res = qt.sesolve(H, psi, t, e_ops=[st_p1, st_p2, st_p3, st_m1, st_m2, st_m3], args=params, options={'store_states': True, 'normalize_output': True, 'atol' : 1e-10, 'rtol' : 1e-8})
    
    
    population_p1 = res.expect[0]
    population_p2 = res.expect[1]
    population_p3 = res.expect[2]
    population_m1 = res.expect[3]
    population_m2 = res.expect[4]
    population_m3 = res.expect[5]
    
    
    time_in_R = integrate.simpson(population_p1 + population_p2 + population_p3 + population_m1 + population_m2 + population_m3, dx=t[1])
    integral_pop_error = 1-(np.abs(1-params['gamma']/2*time_in_R))**2

    #calculate population and phase error 
    coeffs = [[(e_ops_psi[i].dag() * state) for state in res.states] for i in range(8)]
    
    #computational basis states population
    #com_basis_populations = [qt.expect(e_ops_psi[i].proj(), res.states) for i in range(8)]
    
    #phases
    phases = [np.angle(coeffs[i]) for i in range(8)]
    phases = [renormalize_phases(phases[i]/np.pi) for i in range(8)]
    
    #population error
    population_error = 1 - 1/8 * sum(np.abs([coeff[-1] for coeff in coeffs]))**2
    
    #total phase error
    #define entangling phases 
    ent_phase_111 = phases[7]- 3*phases[1]
    ent_phase_101 = phases[5]- 2*phases[1]
    ent_phase_110 = phases[6]- 2*phases[1]
    total_phase_error = 1 - 1/64 * np.abs(4 + 2*np.exp(1j * ent_phase_110[-1]*np.pi) + np.exp(1j * ent_phase_101[-1]*np.pi) - np.exp(1j * ent_phase_111[-1]*np.pi))**2
    
    #101 phase error
    phase_error_101 = 1 - 1/64 * np.abs(7 + np.exp(1j * ent_phase_101[-1]*np.pi))**2
    
    return integral_pop_error, population_error, total_phase_error, phase_error_101
    

if __name__ == '__main__':
    ## EXAMPLE USAGE ##
    #bounds for parameter: Delta0, delta0, Omega0, Omega0_2
    bounds = ([-2*np.pi*250, 2*np.pi*250], [0, 2*np.pi*250], [0,2*np.pi*100])
    #parameter which are optimized
    params_opt = {'Delta0' : 2*np.pi*50, 'delta0' : 2*np.pi*100, 'Omega0' : 2*np.pi*20} 
    params_fix = {'tau' : 1, 'V': 2*np.pi*25, 'gamma' : 0.128, 'Omega_MW' : 2*np.pi*500}
    
    params = params_opt | params_fix
    t = np.linspace(0, params['tau'], 1000)
    
    pulse_functions = {'Delta': PulseHamiltonian.Delta_sin, 'Omega_Re': PulseHamiltonian.Omega_sin, 'Omega_MW' : PulseHamiltonian.Omega_MW_const}

    H = PulseHamiltonian._get_Hamiltonian(params, pulse_functions, True)
    
    pops, phases, params = Optimization_global(H, t, params_fix, params_opt, bounds,Cost_Function_Fidelity, True, seed0=123)

    Fidelity = Bell_state_Fidelity_SQR(H, params, t)
    
    print(f"The optimized gate has a Fidelity of F = {Fidelity}")
    print(f"and the optimized parameters are: {params}")
