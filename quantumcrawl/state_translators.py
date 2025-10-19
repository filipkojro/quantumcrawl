import numpy as np

from . import helpers

def normal(starting_state):
    ''' array with 8 floats thats will be converted to vector containing 4 complex numbers (normalized)
    Args:
        starting_state:
            np.array([a0, a1, a2, a3, a4, a5,a 6, a7])
    Returns:
        np.array([a0 + a1j, a2 + a3j, a4 + a5j, a6 + a7j])
    '''
    starting_state_prepared = helpers.real_to_complex(starting_state)
    starting_state_prepared /= np.linalg.norm(starting_state_prepared)
    return starting_state_prepared

def phase(starting_state):
    ''' array with 8 floats thats will be converted to vector containing 4 complex numbers (normalized)
    Args:
        starting_state: 
            np.array([a0, a1, a2, a3, a4, a5,a 6, a7])
    Returns:
        np.array([a0 * e^(a4 * 1j), a1 * e^(a5* 1j), a2 * e^(a6 * 1j), a3 * e^(a7 * 1j)])
    '''
    starting_state_prepared = starting_state[:4] * np.pow(np.e, starting_state[4:] * 1j)
    starting_state_prepared /= np.linalg.norm(starting_state_prepared)
    return starting_state_prepared

def kron(starting_state):
    ''' array with 8 floats thats will be converted to vector containing 4 complex numbers (normalized)
    Args:
        starting_state:
            np.array([a0, a1, a2, a3, a4, a5,a 6, a7])
    Returns:
        np.kron([a0 + a1j, a2 + a3j], [a4 + a5j, a6 + a7j])
    '''
    subsystem0 = helpers.real_to_complex(starting_state[:4])
    subsystem0 /= np.linalg.norm(subsystem0)
    subsystem1 = helpers.real_to_complex(starting_state[4:])
    subsystem1 /= np.linalg.norm(subsystem1)

    starting_state_prepared = np.kron(subsystem0, subsystem1)

    return starting_state_prepared