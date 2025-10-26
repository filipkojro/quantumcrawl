import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
from tqdm import tqdm
import json

from . import coin_operators
from . import state_translators
from . import metrics
from . import helpers

def singleC2d(states, coins):
    '''
    Applies coin operator to each position in the 2D grid.
    Args:
        states: 3D numpy array of shape (X, Y, 4) representing the quantum states at each position.
        coins: 4D numpy array of shape (X, Y, 4, 4) representing the coin operators at each position.
    Returns:
        a 3D numpy array of the same shape as states after applying the coin operators.
    '''
    next_states = np.zeros_like(states)

    next_states = np.einsum('xyqp,xyp->xyq', coins, states)
    
    return next_states

# old slower functions
# def C2d(states, coins):
#     next_states = np.zeros_like(states)

#     for x in range(len(states)):
#         for y in range(len(states)):
#             next_states[x,y] = coins[x,y] @ states[x,y]
    
#     return next_states

def singleS2d(states):
    '''
    Applies shift operator to the 2D grid.
    Args:
        states: 3D numpy array of shape (X, Y, 4) representing the quantum states at each position.
    Returns:
        a 3D numpy array of the same shape as states after applying the shift operator.
    '''
    next_states = np.zeros_like(states)

    # (y→y–1)
    next_states[: , :-1 , 0] += states[: , 1:   , 0]

    # (x→x+1)
    next_states[1:  , :, 1] += states[:-1 , :, 1]

    # (x→x–1)
    next_states[:-1 , :, 2] += states[1:  , :, 2]

    # (y→y+1)
    next_states[: , 1:  , 3] += states[: , :-1 , 3]

    return next_states

def singleS2ddiag(states):
    '''
    Applies shift operator to the 2D grid. **Same as singleS2d but for diagonal movement.**
    Args:
        states: 3D numpy array of shape (X, Y, 4) representing the quantum states at each position.
    Returns:
        a 3D numpy array of the same shape as states after applying the shift operator.
    '''
    next_states = np.zeros_like(states)

    # left down
    next_states[:-1 , 1:, 0] += states[1:  , :-1, 0]

    # right down
    next_states[1: , 1:  , 1] += states[:-1 , :-1 , 1]

    # left up
    next_states[:-1 , :-1, 2] += states[1:  , 1:, 2]

    # right up
    next_states[1:  , :-1, 3] += states[:-1 , 1:, 3]

    return next_states

# old slower functions
# def S2d(states):
#     next_states = np.zeros_like(states)

#     for x in range(len(states[0])):
#         for y in range(len(states)):
#             if x != len(states[0])-1:
#                 next_states[x + 1, y, 1] += states[x, y, 1]
#             if x != 0:
#                 next_states[x - 1, y, 3] += states[x, y, 3]
#             if y != len(states)-1:
#                 next_states[x, y + 1, 2] += states[x, y, 2]
#             if y != 0:
#                 next_states[x, y - 1, 0] += states[x, y, 0]
#     return next_states

class Walk:
    '''
    Class representing a 2D quantum walk with coin operators and optimization capabilities.
    Args:
        num_steps: Number of steps in the quantum walk.
        coin4all: Coin operator to be used at all positions.
        starting_state: Initial state of the quantum walker in position (num_steps / 2, num_steps / 2).
        state_translator: Function to translate real-valued parameters into complex starting states.
        metric_fun: Function to evaluate the performance of the walk based on the final probability distribution.
        bounds: Bounds for the optimization of the starting state parameters.
        verbose: If True, prints additional information during optimization.
        diag: If True, uses diagonal shift operator instead of standard shift operator.
    '''
    def __init__(self,
                 num_steps = 50,
                 coin4all = coin_operators.coinH,
                 starting_state = np.random.rand(8),
                 state_translator = state_translators.normal,
                 metric_fun = metrics.every_symmetry_symmetry,
                 bounds = None,
                 verbose = False,
                 diag = False):
        
        self.num_steps = num_steps
        self.coin4all = coin4all
        self.starting_state = state_translator(starting_state)
        self.state_translator = state_translator
        self.metric_fun = metric_fun
        self.verbose = verbose
        self.history = np.array([])
        self.diag = diag


        if self.state_translator == state_translators.normal:
            self.state_translator_string = "normal"
            self.bounds = [(-1,1) for _ in range(8)]
        elif self.state_translator == state_translators.phase:
            self.state_translator_string = "phase"
            self.bounds = [(-1,1) for _ in range(4)] + [(0,2 * np.pi) for _ in range(4)]
        elif self.state_translator == state_translators.kron:
            self.state_translator_string = "kron"
            self.bounds = [(-1,1) for _ in range(8)]


        if bounds != None:
            self.bounds = bounds

        self.generate_coin_ops()

    def generate_coin_ops(self):
        '''
        Generates the coin operators for each position in the 2D grid based on the provided coin4all operator.
        '''
        self.coins_op = np.zeros((self.num_steps * 2 + 1, self.num_steps * 2 + 1, 4,4), dtype=complex)

        for i in range(self.num_steps * 2 + 1):
            for j in range(self.num_steps * 2 + 1):
                self.coins_op[i,j] = self.coin4all

    def walk(self):
        '''
        Performs the 2d quantum walk using initialized starting state and coin operator and stores probability history. In loop applies coin and shift operaators.
        '''
        states = np.zeros((self.num_steps * 2 + 1, self.num_steps * 2 + 1, 4), dtype=complex)
        
        states[self.num_steps, self.num_steps] = self.starting_state
        
        self.history = np.zeros((self.num_steps, self.num_steps * 2 + 1, self.num_steps * 2 + 1))

        for i in tqdm(range(self.num_steps)) if self.verbose else range(self.num_steps):
            states = singleC2d(states, self.coins_op)
            if self.diag:
                states = singleS2ddiag(states)
            else:
                states = singleS2d(states)

            self.history[i] = np.sum(np.abs(states)**2,axis=2)

    def revwalk(self):
        '''
        Performs the 2d quantum walk using initialized starting state and coin operator and stores probability history. In loop applies shift and coin operaators.
        '''
        states = np.zeros((self.num_steps * 2 + 1, self.num_steps * 2 + 1, 4), dtype=complex)
        
        states[self.num_steps, self.num_steps] = self.starting_state
        
        self.history = np.zeros((self.num_steps, self.num_steps * 2 + 1, self.num_steps * 2 + 1))

        for i in tqdm(range(self.num_steps)) if self.verbose else range(self.num_steps):
            if self.diag:
                states = singleS2ddiag(states)
            else:
                states = singleS2d(states)
            states = singleC2d(states, self.coins_op)

            self.history[i] = np.sum(np.abs(states)**2,axis=2)

        
    def walk_fun(self, starting_state):
        '''
        Fuctions to be minimized during optimization. Performs the quantum walk for a given starting state and evaluates the metric.
        Args:
            starting_state: Real-valued parameters representing the starting state.
        Returns:
            float representing the metric value to be minimized.
        '''
        states = np.zeros((self.num_steps * 2 + 1, self.num_steps * 2 + 1, 4), dtype=complex)

        starting_state_prepared = self.state_translator(starting_state)
        
        states[self.num_steps, self.num_steps] = starting_state_prepared.copy()

        for i in range(self.num_steps):
            states = singleC2d(states, self.coins_op)
            if self.diag:
                states = singleS2ddiag(states)
            else:
                states = singleS2d(states)

        distribution = np.sum(np.abs(states)**2,axis=2)

        metric = self.metric_fun(distribution)

        self.state_history.append(starting_state_prepared)
        self.m_history.append(metric)

        if self.verbose:
            print(metric, starting_state_prepared)

        return metric

    def optimize(self, initial_guess = None):
        '''
        Optimizes the starting state parameters to minimize the metric function using scipy's minimize function.
        Args:
            initial_guess: Optional initial guess for the starting state parameters. If None, a random guess is used.
        '''
        self.state_history = []
        self.m_history = []
        if initial_guess == None:
            initial_guess = np.random.rand(8)

        self.opt_res = scipy.optimize.minimize(fun=self.walk_fun, x0=initial_guess,bounds=self.bounds)

    def preety_print_optimize_results(self):
        '''
        Prints the results of the optimization in nice format.
        '''

        print('=' * 111)

        self.starting_state = self.state_translator(self.opt_res.x)

        print("\nscipy response mess", self.opt_res.message)

        print("\nlast try metric:", self.opt_res.fun)

        print("\nstarting table", self.opt_res.x)

        print("\ntranslator:")
        print("[a0, a1, a2, a3, a4, a5,a 6, a7]")
        if self.state_translator == state_translators.normal:
            print("[a0 + a1j, a2 + a3j, a4 + a5j, a6 + a7j]")
        elif self.state_translator == state_translators.phase:
            print("[a0 * e^(a4 * 1j), a1 * e^(a5* 1j), a2 * e^(a6 * 1j), a3 * e^(a7 * 1j)]")
        elif self.state_translator == state_translators.kron:
            print("np.kron([a0 + a1j, a2 + a3j], [a4 + a5j, a6 + a7j])")

        print("\nstarting state", self.starting_state)

        print('\nvon neumann entropy:', helpers.von_neuman_entropy(helpers.macierz_gestosci_vec(self.starting_state)))

        print("")
        print('=' * 111)

        self.walk()

        if self.verbose:
            plt.plot(np.arange(len(self.m_history)), self.m_history)
            plt.title("metric history")
            plt.show()

        plt.imshow(self.history[-1], cmap='turbo')
        plt.title("probabilities on last step")
        plt.autoscale()
        plt.colorbar()
        plt.show()
    
    def save_history_to_file(self, filename):
        '''
        Saves optimization history and results to a JSON file.
        Args:
            filename: Name of the file to save the results.
        '''
        try:
            with open(filename, "w") as fp:
                prepared_dict = {
                    "state_translator": self.state_translator_string,
                    "starting_table": self.opt_res.x.tolist(),
                    "last_try_metric": self.opt_res.fun,
                    "response_message": self.opt_res.message,
                    "number_of_steps": self.num_steps
                }

                if self.verbose:
                    print(json.dumps(prepared_dict, indent=4))

                json.dump(prepared_dict, fp, indent=4)
                print(f"saved to {filename}")
        except Exception as e:
            print(f"saving failed: {e}")