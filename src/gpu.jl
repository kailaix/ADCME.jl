export gpu_info, get_gpu, has_gpu

"""
    gpu_info()

Returns the CUDA and GPU information. 
"""
function gpu_info()
    NVCC = "missing"
    CUDALIB = "missing"
    v = "missing"
    u = "missing"
    dcount = 0
    try 
        NVCC = strip(String(read(`which nvcc`)))
        CUDALIB = abspath(joinpath(NVCC, "../../lib64"))
        v = zeros(Int32,1)
        @eval ccall((:cudaRuntimeGetVersion, $(joinpath(CUDALIB, "libcudart.so"))), Cvoid, (Ref{Cint},), $v)
        u = zeros(Int32,1)
        @eval ccall((:cudaDriverGetVersion, $(joinpath(CUDALIB, "libcudart.so"))), Cvoid, (Ref{Cint},), $u)
        dcount = zeros(Int32, 1)
        @eval ccall((:cudaGetDeviceCount, $(joinpath(CUDALIB, "libcudart.so"))), Cvoid, (Ref{Cint},), $dcount)
        v = v[1]
        if v==0
            v = "missing"
        end
        u = u[1]
        dcount = dcount[1]
    catch
        
    end
    
    println("- NVCC: ", NVCC)
    println("- CUDA library directories: ", CUDALIB)
    println("- Latest supported version of CUDA: ", u)
    println("- CUDA runtime version: ", v)
    println("- CUDA include_directories: ", length(ADCME.CUDA_INC)==0 ? "missing" : ADCME.CUDA_INC)
    println("- CUDA toolkit directories: ", length(ADCME.LIBCUDA)==0 ? "missing" : ADCME.LIBCUDA)
    println("- Number of GPUs: ", dcount)

    if NVCC == "missing"
        println("\nTips: nvcc is not found in the path. Please add nvcc to your environment path if you intend to use GPUs.")
    end

    if length(ADCME.CUDA_INC)==0
        println("\nTips: ADCME is not configured to use GPUs. See https://kailaix.github.io/ADCME.jl/latest/tu_customop/#Install-GPU-enabled-TensorFlow-(Linux-and-Windows) for instructions.")
    end

    if dcount==0
        println("\nTips: No GPU resources found. Do you have access to GPUs?")
    end
end

"""
    get_gpu()   

Returns the compiler information for GPUs. 
"""
function get_gpu()
    NVCC = missing 
    CUDALIB = missing 
    CUDAINC = missing 
    try 
        NVCC = strip(String(read(`which nvcc`)))
        CUDALIB = abspath(joinpath(NVCC, "../../lib64"))
        CUDAINC = abspath(joinpath(NVCC, "../../include"))
        if length(ADCME.CUDA_INC)>0 && CUDAINC!=ADCME.CUDA_INC
            @warn """
Inconsistency detected:
ADCME CUDAINC: $(ADCME.CUDA_INC)
Implied CUDAINC: $CUDAINC
"""
        end
    catch
    end
    if length(ADCME.LIBCUDA)>0
        CUDATOOLKIT = split(ADCME.LIBCUDA, ":")[1]
        CUDNN = split(ADCME.LIBCUDA, ":")[2]
    else 
        CUDATOOLKIT = missing 
        CUDANN = missing
    end
    return (NVCC=NVCC, LIB=CUDALIB, INC=CUDAINC, TOOLKIT=CUDATOOLKIT, CUDNN=CUDNN)
end

"""
    has_gpu()

Check if the TensorFlow backend is using CUDA GPUs. Operators that have GPU implementations will be executed on GPU devices. 
See also [`get_gpu`](@ref)

!!! note
    ADCME will use GPU automatically if GPU is available. To disable GPU, set the environment variable `ENV["CUDA_VISIBLE_DEVICES"]=""` before importing ADCME 
"""
function has_gpu()
    s = tf.test.gpu_device_name()
    if length(s)==0
        return false
    else
        return true
    end
end


