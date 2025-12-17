import numpy as np
import qutip as qt


## HAMILTONIAN ##
def _get_Hamiltonian(params, pulse_functions, decay = False, alpha_ = 1/8):
    plus = qt.basis(4, 3)
    minus = qt.basis(4, 2)
    one = qt.basis(4, 1)
    
    if decay == True:
            H_const = (params['V']/2 * (qt.tensor(plus.proj() - minus.proj(), plus.proj() - minus.proj(), qt.qeye(4))- qt.tensor(minus*plus.dag() - plus*minus.dag(), minus*plus.dag() - plus*minus.dag(), qt.qeye(4)))
               + params['V']/2 * (qt.tensor(qt.qeye(4), plus.proj() - minus.proj(), plus.proj() - minus.proj())- qt.tensor(qt.qeye(4), minus*plus.dag() - plus*minus.dag(), minus*plus.dag() - plus*minus.dag())) 
               + alpha_ * params['V']/2 * (qt.tensor(plus.proj() - minus.proj(), qt.qeye(4), plus.proj() - minus.proj())- qt.tensor(minus*plus.dag() - plus*minus.dag(), qt.qeye(4), minus*plus.dag() - plus*minus.dag()))
               - 1j/2*params['gamma']*(qt.tensor(minus.proj(), qt.qeye(4), qt.qeye(4)) + qt.tensor(qt.qeye(4), minus.proj(), qt.qeye(4)) + qt.tensor(qt.qeye(4), qt.qeye(4), minus.proj()))
               - 1j/2*params['gamma']*(qt.tensor(plus.proj(), qt.qeye(4), qt.qeye(4)) + qt.tensor(qt.qeye(4), plus.proj(), qt.qeye(4)) + qt.tensor(qt.qeye(4), qt.qeye(4), plus.proj()))) 
    else:
            H_const = (params['V']/2 * (qt.tensor(plus.proj() - minus.proj(), plus.proj() - minus.proj(), qt.qeye(4))- qt.tensor(minus*plus.dag() - plus*minus.dag(), minus*plus.dag() - plus*minus.dag(), qt.qeye(4)))
               + params['V']/2 * (qt.tensor(qt.qeye(4), plus.proj() - minus.proj(), plus.proj() - minus.proj())- qt.tensor(qt.qeye(4), minus*plus.dag() - plus*minus.dag(), minus*plus.dag() - plus*minus.dag())) 
               + alpha_ * params['V']/2 * (qt.tensor(plus.proj() - minus.proj(), qt.qeye(4), plus.proj() - minus.proj())- qt.tensor(minus*plus.dag() - plus*minus.dag(), qt.qeye(4), minus*plus.dag() - plus*minus.dag()))) 
        
                     
    H = [H_const,
         [1/2 * (qt.tensor(plus.proj() - minus.proj(), qt.qeye(4), qt.qeye(4)) + qt.tensor(qt.qeye(4), plus.proj() - minus.proj(), qt.qeye(4)) + qt.tensor(qt.qeye(4), qt.qeye(4), plus.proj() - minus.proj())), pulse_functions['Omega_MW']],
         [qt.tensor(plus.proj() + minus.proj(), qt.qeye(4), qt.qeye(4)) + qt.tensor(qt.qeye(4), plus.proj() + minus.proj(), qt.qeye(4)) + qt.tensor(qt.qeye(4), qt.qeye(4), plus.proj() + minus.proj()), pulse_functions['Delta']],
         [(qt.tensor(minus*one.dag() + plus*one.dag() +one*minus.dag() + one*plus.dag(), qt.qeye(4), qt.qeye(4)) + qt.tensor(qt.qeye(4), minus*one.dag() + plus*one.dag() +one*minus.dag() + one*plus.dag(), qt.qeye(4)) + qt.tensor(qt.qeye(4), qt.qeye(4), minus*one.dag() + plus*one.dag() +one*minus.dag() + one*plus.dag()))/(2*np.sqrt(2)) , pulse_functions['Omega_Re']]]
    
    return H



## PULSE SHAPES ##
def Delta_sin(t, args):
        return args['delta0'] - args['Delta0'] * np.sin(np.pi * t / args['tau']) ** 2
    
def Omega_sin(t, args):
    return args['Omega0'] * np.sin(np.pi * t / args['tau']) ** 2

def Omega_MW_const(t, args):
    return args['Omega_MW'] + t * 0

    

if __name__ == '__main__':
    print('nothing to do')
    