import numpy as np
import scipy

def real_to_complex(real):
    '''
    Converts a list of 2n real numbers into a list of n complex numbers
    Args:
        real: list or np array of real numbers of length 2n
    Returns:
        np array of complex numbers of length n
    '''
    return np.array([real[i] + real[i+1] * 1j for i in range(0, len(real), 2)], dtype=complex)

def complex_to_real(comp):
    '''
    Converts a list of n complex numbers into a list of 2n real numbers
    Args:
        comp: list or np array of complex numbers of length n
    Returns:
        np array of real numbers of length 2n
    '''
    result = []
    for i in comp:
        result.append(i.real)
        result.append(i.imag)
    return np.array(result, dtype=float)

def macierz_gestosci_vec(vec):
    '''
    Normalizez the input vector and returns its density matrix
    Args:
        vec: np array of complex numbers
    Returns:
        np array representing the density matrix
    '''
    vec /= np.linalg.norm(vec)
    return np.outer(np.conjugate(vec), vec)

def trB(A):
    '''
    Partial trace over the second subsystem for a 4x4 density matrix A
    Args:
        A: np array representing the 4x4 density matrix
    Returns:
        np array representing the 2x2 reduced density matrix'''
    return np.array([[A[0,0]+A[1,1], A[0,2]+A[1,3]],
                     [A[2,0]+A[3,1], A[2,2]+A[3,3]]])

def von_neuman_entropy(A):
    '''
    Calculates the Von Neumann entropy of a density matrix A
    Args:
        A: np array representing the density matrix
    Returns:
        float representing the Von Neumann entropy
    '''
    roA = trB(A)
    entropy = -np.trace(roA @ scipy.linalg.logm(roA)) / np.log(2)
    return entropy.real