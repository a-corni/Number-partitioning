import json
from qat.lang.AQASM import Program, RX, RY, RZ, H
from qat.core import Observable, Term, Variable
from qat.plugins import  ScipyMinimizePlugin
#from qat.qpus import PyLinalg
from qat.qpus import get_default_qpu

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch


def bipartition(s, p):
    
    ###Define a variational circuit
    #As a Program
    prog = Program()
    
    #With n qubits
    nqbits = len(s)
    qbits = prog.qalloc(nqbits)
    
    #Define the variational states (einsatz)
    #|psi(theta)> = Rx1(theta1x)...Rxn(theta1n)|0,...,0>
    
    for i in range(0, nqbits):
        
        for j in range(1):
            #H(qbits[i])
            thetaij = prog.new_var(float, "theta"+str(i)+str(j))    
            if j == 0 :
                RX(np.pi*thetaij)(qbits[i])
                #RZ(thetaij)(qbits[i]) #H Rz H method
            #elif j ==1 :
            #    RY(thetaij)(qbits[i])
            #elif j == 2:
            #    RZ(thetaij)(qbits[i])
            # elif j == 3 :
            #     RY(thetaij)(qbits[i])
            # elif j == 4 :
            #     RX(thetaij)(qbits[i])
            #H(qbits[i])
    
    #export program into a quntum circuit
    circuit = prog.to_circ()
    
    #print created variables"
    print("Variables:", circuit.get_variables())
    
    ###Define an observable
    #Initialization
    obs = Observable(nqbits)
    
    #Observable = Hamiltonian we want to minimize
    # => Reformulate bi-partition problem in Hamiltonian minimalization problem :
    #Bi-partition problem : Find A1,A2 such that A1 u A2 = s and minimizes |sum_{n1 in A1}n1 - sum_{n2 in A2}n2|
    #Optimization Problem : Find e = {ei} in {-1,1}^n minimizing |sum_{1<=i<=n}ei.si|
    #Non-Linear Integer Programming Optimization Problem Find e = {ei} in {-1,1}^n minimizing (sum_{1<=i<=n}ei.si)^2
    #Ising problem : Find e = {ei} in {-1,1}^n minimizing sum_{1<=i<=n}(si)^2 + sum_{1<=i<=n, 1<=i<j<=n} (si.sj).ei.ej)^2 (si^2 = 1)
    
    J = [[si*sj for si in s] for sj in s] 
    b = [si**2 for si in s]
    
    obs += sum(b)*Observable(nqbits, constant_coeff = sum(b))
    
    for i in range(nqbits):
        for j in range(i-1):
        
            obs += Observable(nqbits, pauli_terms = [Term(J[i][j], "ZZ", [i, j])]) #Term(coefficient, tensorprod, [qubits]) 
        
    ###Create a quantum job
    #Made of a circuit and and an observable
    job = circuit.to_job(observable = obs) #nbshots=inf
    
    ###Define classical optimizer : scipy.optimia.minimize
    method_name = "COBYLA" #"Nelder-Mead", "Powell", "CG",...
    optimize = ScipyMinimizePlugin(method = method_name, tol = 1e-3, options = {"maxiter" : p})
    
    ###Define qpu on which to run quantum job
    stack = optimize | get_default_qpu()
    
    optimizer_args = {"method": method_name, "tol":1e-3, "options":{"maxiter":p}}
    
    result = stack.submit(job, meta_data ={"ScipyMinimizePlugin": json.dumps(optimizer_args)})
    
    ###Get characteristics of best parametrized circuit found
    final_energy = result.value
    parameters = json.loads(result.meta_data['parameters'])
    
    print("final energy:", final_energy)
    print('best parameters:', parameters)
    #print('trace:', result.meta_data['optimization_trace'])
    
    ###Deduce the most probable state => solution of the problem
    #Probability to measure qubit i in state |0> : cos(theta_i)^2
    #Probability to measure qubit i in state |1> : sin(theta_i)^2
    states = {}
    
    best_state = ''
    max_proba = 0
    
    for i in range(2**nqbits):
        
        state = bin(i)[2:].zfill(nqbits)
        proba = 1

        for i in range(nqbits):
            proba = proba/2*((1+(-1)**int(state[nqbits-i-1]))*np.cos(parameters[i]*np.pi/2)+(1+(-1)**(int(state[nqbits-i-1])+1))*np.sin(parameters[i]*np.pi/2))
        
        if max_proba < proba:
            max_proba = proba
            best_state = state
        
        states[state] = proba
    
    print(states)
    
    A1 = []
    A2 = []
    
    for i in range(nqbits):
    
        if best_state[i] == '0':
            A1.append(s[i])
    
        elif best_state[i] == '1':
            A2.append(s[i])
    
    return A1, A2