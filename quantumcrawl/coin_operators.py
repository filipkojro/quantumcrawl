import numpy as np

# H \otimes H
Hgate = np.array([[1,1],[1,-1]]) / np.sqrt(2)
H2gate = np.kron(Hgate, Hgate)

# moneta fouriera
# N = 4
# n = np.arange(N)
# k = n.reshape((N, 1))
# omega = np.exp(-2j * np.pi / N)
# F = omega ** (k * n)
F = np.array([[1,1,1,1],
              [1,-1j,-1,1j],
              [1,-1,1,-1],
              [1,1j,-1,-1j]])
F /= np.sqrt(np.linalg.norm(F))

# moneta groovera?
G = np.array([[-1,1,1,1],
              [1,-1,1,1],
              [1,1,-1,1],
              [1,1,1,-1]]) / 2

coinH = H2gate
coinF = F
coinG = G


N = 4                     # lub N = 2**d
n = np.arange(N)
k = n.reshape((N, 1))

# tak, jak na obrazku: plus w wykładniku i normalizacja 1/sqrt(N)
omega = np.exp(2j * np.pi / N)
Fminus = (1 / np.sqrt(N)) * (omega ** (k * n))
