# Install ADCME on Windows

The following sections provide instructions on installing ADCME on Windows computers. 

## Install Julia

Windows users can install ADCME following these [instructions](https://julialang.org/downloads/). Choose your version of Windows (32-bit or 64-bit).

[Detailed instructions to install Julia on Windows](https://julialang.org/downloads/platform/#windows)

For Windows users, you can press the Windows button or click the Windows icon (usually located in the lower left of your screen) and type `julia`. Open the Desktop App `Julia` and you will see a Julia prompt. 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/windows_julia.png?raw=true)


## Install C/C++ Compilers

To use and build custom operators, you need a C/C++ compiler that is compatible with the TensorFlow backend. The prebuilt TensorFlow shipped with ADCME was built using Microsoft Visual Studio 2017 15. Therefore, you need to install this specific version. 

1. Download and install from [here](https://visualstudio.microsoft.com/vs/older-downloads/)

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/vs2017.png?raw=true)

Note that this is an older version of Visual Studio. It's not the one from 2019 but the previous version from 2017.

2. Double click the installer that you just downloaded. You will see the following image:

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/install1.png?raw=true)

A free community version is available (**Visual Studio Community 2017**). Click **install** and a window will pop up. 

3. Make sure the following two checkboxes are checked:



- In the **Workloads** tab, **Desktop development with C++** is checked.

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/install2.png?raw=true)

- In the **Indivisual components** tab, **MSBuild** is checked.

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/install3.png?raw=true)

4. Click **install** on the lower right corner. You will see the following window. 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/install4.png?raw=true)

5. The installation may take some time. Once the installation is finished, you can safely close the installer. 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/install5.png?raw=true)

## Configure Paths

In order to locate shared libraries and executable files provided by ADCME, you also need to set an extra set of PATH environment variables. Please add the following environment variables to your system path (my user name is `kaila`; please replace it with yours!)

```
C:\Users\kaila\.julia\adcme\Scripts
C:\Users\kaila\.julia\adcme\Library\bin
C:\Users\kaila\.julia\adcme\
```

Here is how you can add these environment paths:

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/windows_install.png?raw=true)

## Install ADCME

Now you can install ADCME via 
```julia
using Pkg
Pkg.add("ADCME")
```


