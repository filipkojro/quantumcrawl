# quantumcrawl
Python libraty meant to simplify simulations of 2d quantum walks on discrete grid and search of a most symmetric distibution of given parameters.

## Installation (using pip)

### main branch

```bash
pip install git+https://github.com/filipkojro/quantumcrawl.git
```
Versions of libraries in dependencies was copied from my current workstation. Other versions may or may not work.

### testing branch

If you want most up to date version but **not yet tested** you can install from testing branch.
```bash
pip install git+https://github.com/filipkojro/quantumcrawl.git@testing
```

## Example usage

### Optimization
```python
import quantumcrawl.walk
import quantumcrawl.coin_operators

walkFnormal = quantumcrawl.walk.Walk(coin4all=quantumcrawl.coin_operators.coinH, diag=True)
walkFnormal.optimize()
walkFnormal.preety_print_optimize_results()
```

### Only simulation
```python
import quantumcrawl.coin_operators
import quantumcrawl.walk
import matplotlib.pyplot as plt

walk = quantumcrawl.walk.Walk(coin4all=quantumcrawl.coin_operators.coinF, starting_state=[ 0, 0, 0, 0, 0, 0, 1, 0], num_steps=100)
walk.walk()

plt.imshow(walk.history[-1], cmap='turbo', )
plt.title("probabilities on last step")
plt.autoscale()
plt.colorbar()
plt.show()
```