# ADCME Documentation

ADCME is suitable for conducting inverse modeling in scientific computing. The purpose of the package is to: (1) provide differentiable programming framework for scientific computing based on TensorFlow automatic differentiation (AD) backend; (2) adapt syntax to facilitate implementing scientific computing, particularly for numerical PDE discretization schemes; (3) supply missing functionalities in the backend (TensorFlow) that are important for engineering, such as sparse linear algebra, constrained optimization, etc. Applications include

- full wavelength inversion

- reduced order modeling in solid mechanics

- learning hidden geophysical dynamics

- physics based machine learning

- parameter estimation in stochastic processes

The package inherents the scalability and efficiency from the well-optimized backend TensorFlow. Meanwhile, it provides access to incooperate existing C/C++ codes via the custom operators. For example, some functionalities for sparse matrices are implemented in this way and serve as extendable "plugins" for ADCME. 

# Getting Started 

To install ADCME, use the following command:
```julia
using Pkg
Pkg.add("ADCME")
```
to load the package, use
```julia
using ADCME
```
The building process will check the dependencies (`tensorflow`, `tensorflow-probability`, etc.) If the install is not successful, check your system and make sure `tensorflow==1.14` and `tensorflow-probability==0.7` are properly installed.

