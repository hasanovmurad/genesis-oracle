# Ascension Report

## Exercise 2: Speedup Factor

Legacy simulation time: 2.766298 seconds

JAX second run time: 0.040943 seconds

Speedup factor:

2.766298 / 0.040943 = 67.56

The first execution of a JIT-compiled function is slower because JAX must trace the Python function and compile it using XLA. The second execution reuses the already compiled machine code and therefore executes much faster.

## Exercise 3: Automatic Differentiation

Optimized initial velocity: 29.999977 m/s

jax.grad computes derivatives through automatic differentiation of the computational graph. Finite differences approximate derivatives numerically using expressions such as (f(x+h)-f(x))/h, which depend on the choice of h and can introduce numerical errors.

## Exercise 4: Explicit State Management in Flax

Flax separates model architecture from model parameters. The architecture is defined by a Module, while parameters are created explicitly using model.init together with a jax.random.PRNGKey.

The forward pass is executed through model.apply, where the parameter dictionary is passed explicitly. This design makes state management explicit and functional. In contrast, Keras usually stores weights internally inside the model object, making state management more implicit.
