

# Overview

> ADCME: Your Gateway to Inverse Modeling with Physics Based Machine Learning

ADCME is an open-source Julia package for inverse modeling in scientific computing using automatic differentiation. The backend of ADCME is the high performance deep learning framework, TensorFlow, which provides parallel computing and automatic differentiation features based on computational graph, but  ADCME augments TensorFlow by functionalities---like sparse linear algebra---essential for scientific computing. ADCME leverages the Julia environment for maximum efficiency of computing. Additionally, the syntax of ADCME is designed from the beginning to be compatible with the Julia syntax, which is friendly for scientific computing. 

**Prerequisites**

The tutorial does not assume readers with experience in deep learning. However, basic knowledge of scientific computing in Julia is required. 

**Tutorial Series**

[What is ADCME? Computational Graph, Automatic Differentiation & TensorFlow](./tu_whatis.md)

[ADCME Basics: Tensor, Type, Operator, Session & Kernel](./tu_basic.md)

[PDE Constrained Optimization](./tu_optimization.md)

[Sparse Linear Algebra in ADCME](./tu_sparse.md)

[Numerical Scheme in ADCME: Finite Difference Example](./tu_fd.md)

[Numerical Scheme in ADCME: Finite Element Example](./tu_fem.md)

[Inverse Modeling in ADCME](./tu_inv.md)

[Inverse Modeling Recipe](./tu_recipe.md)

[Combining NN with Numerical Schemes](./tu_nn.md)

[Advanced: Automatic Differentiation for Implicit Operations](./tu_implicit.md)

[Advanced: Custom Operators](./tu_customop.md)

[Advanced: Debugging and Profiling](./tu_debug.md)

[Exercise](./exercise.md)[^exercise]

[^exercise]: If you want to discuss or check your exercise solutions, you are welcome to send an email to kailaix@hotmail.com.





