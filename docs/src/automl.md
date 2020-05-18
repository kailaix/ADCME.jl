# AutoML: Adaptive Deep Learning

## Introduction

One of the key ingredients to the success of today's deep learning is deep neural networks (DNN). DNNs have very powerful expressivity, and can approximate very complex and high dimensional functions surprisingly well. However, one annoying problem is that the effectiveness of a deep learning model relies on DNN's architectures, which are very challenging to design. Over the years, researchers strive to design neural network architectures that work for specific domains. For example, VGGNet, AlexNet, ResNet, and etc., are successful models in computer visions. Recurrent neural networks, transformers, attention, and etc., are more specific to natural language processing. Finding the right architectures is challenging, yet is very promising to solve long-standing problems.

This challenge calls for an "algorithm" that can find an appropriate architecture **automatically**. Such a topic is known as **autoML** (or **NAS**, **meta learning**, etc.) in the machine learning community. For scientific computational researchers, autoML can be compared with adaptive methods, such as $hp$-finite element method, where a set of appropriate basis functions (or mesh) is found algorithmically. Once we have an algorithm that can search for an good neural network architecture, we free ourselves from trial and error and throw all the computational work to today's high performance computing system. 

## Method

In this article, we introduce ADCME's autoML module, which implements an evolutionary algorithm for autoML. The basic idea is that we maintain an optimal average ensemble of subnetworks, which can be as simple as a fully connected neural network. Mathematically, the ensembled neural network is 

$$h(\theta) = \frac{1}{N}\sum_{i=1}^N h_i(\theta)$$

where $h$ is the ensembled neural network, $h_i$ is a subnetwork. 

At each iteration, given the set of subnetwork $\{h_i\}_{i=1}^N$, we propose a set of candidate subnetworks based on the most recent added subnetwork $h_N$

$$\{h_i'\}_{i=1}^M = g(h_N)$$

where $g$ is called **generator function**. 

There are two possible next step to try:

1. Replace $h_N(\theta)$ with a new subnetwork in $\{h_i\}_{i=1}^N$, i.e.,

$$h^*_j(\theta) = \frac{1}{N}\left(\sum_{i}^{N-1} h_i(\theta) + h_j'\right)\quad j = 1,2, \ldots, M$$

2. for each $i\in\{1,2,\ldots, M\}$, we train the neural network $h_i'$ using whatever optimization method. In the end, we can compute a loss function on the test set for new neural network ensembles

$$h^*_j(\theta) = \frac{1}{N+1}\left(\sum_{i}^{N} h_i(\theta) + h_j'\right)\quad j = 1,2, \ldots, M$$

If all the computed loss functions are larger than the loss function for $h(\theta)$, we move to the next iteration and generate new subnetworks.

Otherwise, we add the optimal subnetwork to the ensemble set 

$$j = \arg\min_j h_j^* (\theta)$$

That is, the new ensembled neural network becomes 


$$h^*(\theta) = \frac{1}{N}\left(\sum_{i}^{N-1} h_i(\theta) + h_j'\right)\quad j = 1,2, \ldots, M$$


or

$$h^*(\theta) =  \frac{1}{N+1}\left(\sum_{i}^{N} h_i(\theta) + h_{j}'\right)$$



## Example

To implement the above idea in ADCME, let us consider a simple example: train a neural network to approximate 

$$y = \sin (2\pi x)$$

We need to implement two files: a driver file (`automl.jl`) and a neural network training file (`automl_helper.jl`). 

### Implementing the Driver File 

```julia
using Revise
using ADCME

function generate_subnetwork(automl::AutoML)
    if length(automl.subnetworks)==0
        return ["4";"5";"6"]
    end
    hidden_size = maximum(parse.(Int64, automl.most_recent_subnetworks))
    return [
        string(hidden_size);
        string(hidden_size+1)
    ]
end

function execute_subnetwork(s::String, rep::Bool)
    output = String(read(`julia automl_helper.jl $(automl.WORKSPACE) $(s) $(rep)`)) 
    l = parse(Float64, match(r"standalone loss >>> (.*?) <<<", output)[1])
    le = parse(Float64, match(r"ensemble loss >>> (.*?) <<<", output)[1])
    return l, le
end


ADCME.options.automl.max_iter = 20
automl = AutoML(generate_subnetwork, execute_subnetwork)
run(automl)s
```

In this driver file, we need to implement two functions that serve as inputs to [`AutoML`](@ref)

* `generate_subnetwork` specifies the rule to generate new subnetworks. Every subnetwork is described by a string, and programmers are responsible to convert string to neural networks and vice versa. 
* `execute_subnetwork` takes the neural network name (a string) as input and outputs two loss function values: the standalone loss function value, which is the loss function when using just the subnetwork, and the ensemble loss function value, which is the loss function value when this subnetwork $h_j'(\theta)$ is ensembled with the current best ensembled one $h(\theta)$. In the implementation, to run all the training in parallel, we launch the training by running external program. 


### Implementing the Training File 
```julia
using ADCME
reset_default_graph()

WORKSPACE = ARGS[1]
name = ARGS[2]
rep = parse(Bool, ARGS[3])
previous_ensembles = strip.(readlines("$WORKSPACE/ensembles.txt"))

@info  WORKSPACE, name
@info previous_ensembles


x = reshape(Array(LinRange(-1,1,100)), :, 1)
y = @. sin(2x)

x0 = reshape(rand(100)*2 .- 1, :, 1)
y0 = @. sin(2x0)

function create_neural_network(name)
    num_layers = parse(Int64, name)
    nn = x->fc(x, [num_layers,num_layers,num_layers,1], "nn$name")
    return nn 
end

function compute_loss_function(xp, yp, nn)
    xp, yp = constant(xp), constant(yp)
    loss = sum((nn(xp) - yp)^2)
    return loss
end

nn = create_neural_network(name)
loss = compute_loss_function(x, y, nn)
sess = Session(); init(sess)
if !isfile("$WORKSPACE/$name/data.mat")
    global loss0 = BFGS!(sess, loss)
    ADCME.save(sess, "$WORKSPACE/$name/data.mat")
else
    @info "\"$WORKSPACE/$name/data.mat\" exists." 
end

ensemble_names = filter(x->length(x)>0, String[previous_ensembles;name])
if rep 
    ensemble_names = ensemble_names[1:end-1]
end
@info "ensemble_names = ", ensemble_names
loss2 = average_ensemble(sess, ensemble_names, create_neural_network, nn->compute_loss_function(x0, y0, nn), WORKSPACE)
loss0 = run(sess, loss)
loss_ = run(sess, loss2)
println("standalone loss >>> $(loss0[end]) <<<")
println("ensemble loss >>> $(loss_) <<<")
```

The training file basically trains the subnetwork and ensembles it with the current best neural network. Two notes for the file is 

1. The trained weights and biases must be stored in `$WORKSPACE/$name/data.mat`
2. `average_ensemble` is an `ADCME` function


### Launching the Training

Finally, we can launch autoML with 

```
julia> include("automl.jl")
```



![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/automl.png?raw=true)