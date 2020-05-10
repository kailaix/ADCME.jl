# General Seismic Inversion using Automatic Differentiation

---

Weiqiang Zhu (**co-first author**), Kailai Xu (**co-first author**), Eric Darve, and Gregory C. Beroza

[Project Website](https://github.com/kailaix/ADSeismic.jl)

---

Imaging Earth structure or seismic sources from seismic data involves minimizing a target misfit function, and is commonly solved through gradient-based optimization. The adjoint-state method has been developed to compute the gradient efficiently; however, its implementation can be time-consuming and difficult. We develop a general seismic inversion framework to calculate gradients using reverse-mode automatic differentiation. The central idea is that adjoint-state methods and reverse-mode automatic differentiation are mathematically equivalent. The mapping between numerical PDE simulation and deep learning allows us to build a seismic inverse modeling library, ADSeismic, based on deep learning frameworks, which supports high performance reverse-mode automatic differentiation on CPUs and GPUs. We demonstrate the performance of ADSeismic on inverse problems related to velocity model estimation, rupture imaging, earthquake location, and source time function retrieval. ADSeismic has the potential to solve a wide variety of inverse modeling applications within a unified framework.

| Connection Between the Adjoint-State Method and Automatic Differentiation | Remarkable Multi-GPU Acceleration               | Earthquake Location and Source-Time Function Inversion       |
| ------------------------------------------------------------ | ----------------------------------------------- | ------------------------------------------------------------ |
| ![compare-NN-PDE](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/compare-NN-PDE.png?raw=true)               | ![image-20200313110921108](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/cpugpu.png?raw=true) | ![image-20200313111045121](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/earthquake.png?raw=true) |



