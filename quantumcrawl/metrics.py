import numpy as np
import ot

def all_rots(A):
    '''
    Generates all 4 rotations of a matrix and returns them as a list.
    Args:
        A: np array representing the 2D distribution
    Returns:
        list of np arrays representing all 4 rotations of A
    '''
    rots = [A]

    for i in range(3):
        rots.append(np.rot90(rots[i]))
    return rots

def every_symmetry_symmetry(A):
    '''
    Calculates normalized sum of differences between all symmetric versions of 2D distribution A using sum of absolute difference of every value.
    Args:
        A: np array representing the 2D distribution
    Returns:
        float representing the difference between all symmetric versions of A
    '''
    rots = all_rots(A) + all_rots(np.flip(A, 0)) # + all_rots(np.flip(A, 1))

    result = 0
    comps = 0

    for i in range(len(rots)):
        for j in range(len(rots)):
            if i != j:
                result += np.sum(np.abs(rots[i] - rots[j]))
                comps += 1
    return result / comps

def TVD2d(A):
    '''
    Calculates sum of differences between axial symmetric versions of 2D distribution A using sum of absolute difference of every value.
    Args:
        A: np array representing the 2D distribution
    Returns:
        float representing the difference between A and its axial symmetric versions
    '''
    Ay_flip = np.flip(A, 0)
    Ax_flip = np.flip(A, 1)

    Ay_diff = np.abs(A - Ay_flip)
    Ax_diff = np.abs(A - Ax_flip)

    y_diff_sum = np.sum(np.sum(Ay_diff, 0))
    x_diff_sum = np.sum(np.sum(Ax_diff, 1))

    return x_diff_sum + y_diff_sum

def point_symmetry(A):
    '''
    Calculates sum of differences between point symmetric version of 2D distribution A using sum of absolute difference of every value.
    Args:
        A: np array representing the 2D distribution
    Returns:
        float representing the difference between A and its point symmetric version
    '''
    rotated_A = np.rot90(A,2)
    return np.sum(np.abs(A - rotated_A))

def rot_symmetry(A):
    '''
    Calculates sum of differences between all rotated versions of 2D distribution A using sum of absolute difference of every value.
    Args:
        A: np array representing the 2D distribution
    Returns:
        float representing the difference between all rotated versions of A
    '''
    rot_A = A.copy()
    result = 0
    for _ in range(3):
        rot_A = np.rot90(rot_A)
        result += np.sum(np.abs(A - rot_A))

    return result

def wasserstein_distance(A, B):
    '''
    Calculates the Wasserstein distance between two 2D distributions A and B using cityblock metric.
    Args:
        A: np array representing the first 2D distribution
        B: np array representing the second 2D distribution
    Returns:
        float representing the Wasserstein distance between A and B
    '''
    AF = A.flatten()
    BF = B.flatten()

    coords = np.indices(A.shape).reshape(2, -1).T

    M = ot.dist(coords, metric='cityblock')

    distance = ot.emd2(AF, BF, M)

    return distance

def every_symmetry_symmetry_wasserstein(A):
    '''
    Calculates normalized sum of differences between all symmetric versions of 2D distribution A using Wasserstein distance.
    Args:
        A: np array representing the 2D distribution
    Returns:
        float representing the difference between all symmetric versions of A
    '''
    rots = all_rots(A) + all_rots(np.flip(A, 0)) # + all_rots(np.flip(A, 1))

    result = 0
    comps = 0

    for i in range(len(rots)):
        for j in range(len(rots)):
            if i != j:
                result += wasserstein_distance(rots[i], rots[j])
                # result += np.sum(np.abs(rots[i] - rots[j]))
                comps += 1
    return result / comps