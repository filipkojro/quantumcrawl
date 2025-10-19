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


# proper coin opaerators names
coinH = H2gate
coinF = F
coinG = G