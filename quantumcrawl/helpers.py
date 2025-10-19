import numpy as np
import scipy

def real_to_complex(real):
    return np.array([real[i] + real[i+1] * 1j for i in range(0, len(real), 2)], dtype=complex)

def complex_to_real(comp):
    result = []
    for i in comp:
        result.append(i.real)
        result.append(i.imag)
    return np.array(result, dtype=float)

def macierz_gestosci_vec(vec):
    vec /= np.linalg.norm(vec)
    return np.outer(np.conjugate(vec), vec)

def trB(A):
    return np.array([[A[0,0]+A[1,1], A[0,2]+A[1,3]],
                     [A[2,0]+A[3,1], A[2,2]+A[3,3]]])

def von_neuman_entropy(A):
    roA = trB(A)
    entropy = -np.trace(roA @ scipy.linalg.logm(roA)) / np.log(2)
    return entropy.real