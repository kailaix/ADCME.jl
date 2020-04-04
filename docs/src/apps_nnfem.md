# Symmetric Positive Definite Neural Networks (SPD-NN) 

---

Kailai Xu (**co-first author**), Huang, Daniel Z. (**co-first author**), and Eric Darve. "Learning Constitutive Relations using Symmetric Positive Definite Neural Networks"

[Project Website](https://github.com/kailaix/NNFEM.jl)

---

Material modeling aims to construct constitutive models to describe the relationship between strain and stress, in which the relationship may be hysteresis. The constitutive relations can be derived from microscopic interactions between multiscale structures or between atoms. However, the first-principles simulations, which resolve all these interactions, remain prohibitively expensive. This motivates to learn a data-driven constitutive relation that expresses the mapping from strain tensors (possibly with historic information) to stress tensors. 

In a [previous application](https://kailaix.github.io/ADCME.jl/dev/apps_constitutive_law/), we showed how to learn a constitutive relation, which is a (nonlinear) map from strain tensors to stress tensors, from state variables in a static equation. However, many constitutive relations also depend on the historic information, such as (elasto-)plasticity and viscosity. The constitutive relation is much more complex since they have the form 

$${\sigma}(t) = \mathcal{M}({\epsilon}(t), \mathcal{I}(t))$$

where $t$ is the time, $\sigma$, $\epsilon$ are strain and stress tensors,  $\mathcal{I}$ contains the historic information in $[0,t)$, and $\mathcal{M}$ is an unknown function. This is a high dimensional mapping and traditional methods, such as piecewise linear functions, suffer from the curse of dimensionality problem. This issue motivates us to use neural network as surrogate models. 

However, typically we cannot measure the stress directly and therefore a strain-stress pair training data set is not available. The idea is to plug the neural network based constitutive relation into physical laws, i.e., the kinematic and kinetic equations of motion, and obtain a hypothetical displacement $u$. $u$ is a quantity which we can measure. We can find the optimal weights and biases of the neural network by minimizing the discrepancy between $u$ and observed displacement. This procedure is done using automatic differentiation, where the gradients are back-propagated through both the numerical solver and the neural network. 

The challenge here is that **the numerical solver is unstable if we plug in a random neural network based constitutive relation**. Indeed, numerical solvers are developed based on certain physical assumptions, and a  neural network from random choices may go wild and can be quite ill-behaved. The idea is to **add physical constraints** to the neural network. The solution we proposed is the symmetric positive definite neural network (SPD-NN): instead of modeling the constitutive relation directly, we model the tangent stiffness matrix. To be more specific,

$$\Delta {\sigma} =\mathsf{L}_{\theta}\mathsf{L}_{\theta}^T \Delta {\epsilon}$$

where $\mathsf L_\theta$ is a Cholesky factor and therefore $\mathsf{L}_{\theta}\mathsf{L}_{\theta}^T$ is SPD. The formulation preserves both time consistency and weak convexity of the strain energy. In specific applications, the formulation is further customized. For example, 

$${\sigma}^{n+1} = \mathsf{M}_{\theta}({\epsilon}^{n+1}, {\epsilon}^{n}, {\sigma}^{n}) := \left\{\begin{matrix}
& \mathsf{C}_{\theta}{\epsilon}^{n+1}  & \text{Linear Elasticity}\\
&\mathsf{L}_{\theta}({\epsilon}^{n+1} )\mathsf{L}_{\theta}({\epsilon}^{n+1})^T({\epsilon}^{n+1} -  {\epsilon}^{n})  + {\sigma}^{n} & \text{Nonlinear Elasticity}\\
& (1 - D(\sigma^{n}, \tilde{\sigma}_Y)) \sigma_{\mathrm{elasticity}}^{n+1} + D(\sigma^{n}, \tilde{\sigma}_Y) \sigma_{\mathrm{plasticity}}^{n+1} & \text{Elasto-Plasticity}
\end{matrix}\right.$$

As a final remark, the challenge in learning plasticity behavior is that we have to capture the loading and unloading transitions. In this case, the tangent stiffness matrix exhibits a discontinuity. To alleviate the problem, we adjust the neural network by using a transition function $D$ in the elasto-plasticity case. 