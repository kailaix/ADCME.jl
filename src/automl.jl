export AutoML, average_ensemble

"""
    mutable struct AutoML 
        WORKSPACE::String
        candidates::Array{<:SubNetwork}
        losses::Array{Float64}
        subnetworks::Array{<:SubNetwork}
        generate_subnetwork::Function
        execute_subnetwork::Function
    end 

An autoML struct. 

- `WORKSPACE`: the temporary directory for storing data 
- `candidates`: candidates neural networks proposed by `generate_subnetwork`
- `losses`: historical smallest losses 
- `subnetworks`: historical best neural networks 
- `generate_subnetwork`: a function that is used for generating new subnetworks
- `execute_subnetwork`: a function that executes the new subnetwork and returns the ensembled loss function 
"""
mutable struct AutoML 
    WORKSPACE::String
    candidates::Array
    losses::Array{Float64}
    subnetworks::Array
    subnetworks_history::Dict
    generate_subnetwork::Function
    execute_subnetwork::Function
    status::Int64 # =-1: initial, 0: normal, >0 has not found better candidate in the last status iterations
    most_recent_subnetworks::Union{Missing, Array{String}}
end

function AutoML(generate_subnetwork::Function, execute_subnetwork::Function) 
    WORKSPACE = joinpath(tempdir(), randstring(10))
    mkdir(WORKSPACE)
    candidates = String[]
    losses = Float64[]
    subnetworks = String[]
    subnetworks_history = Dict{String, Float64}()
    AutoML(WORKSPACE, candidates, losses, subnetworks, subnetworks_history, generate_subnetwork, execute_subnetwork, -1, missing)
end


function execute_subnetwork(automl::AutoML, rep::Int64)
    candidates = Channel{Tuple}(length(automl.candidates))
    losses = zeros(length(automl.candidates))
    for k = 1:length(automl.candidates)
        c = automl.candidates[k]
        if !isdir(joinpath(automl.WORKSPACE, automl.candidates[k]))
            mkdir(joinpath(automl.WORKSPACE, automl.candidates[k]))
        end
        put!(candidates, (k,c))
    end
    function exec(n)
        k, c = take!(candidates)
        options.automl.verbose && @info "[AutoML] Preparing candidates $k ($(c)), replacing? $(rep>0)"
        l, el = automl.execute_subnetwork(c, rep)
        return (k, l, el)
    end
    results = asyncmap(exec, 1:length(candidates.data))
    
    for (i,l, el) in results
        losses[i] = el
        automl.subnetworks_history[automl.candidates[i]] = l 
    end
    for i = 1:length(automl.candidates)
        options.automl.verbose && @info "Candidates $i (name = $(automl.candidates[i])) standalone loss = $(automl.subnetworks_history[automl.candidates[i]]), ensemble loss = $(losses[i])"
    end

    i = argmin(losses)
    options.automl.verbose && @info "Network ensembled with subnetwork $(automl.candidates[i]) has the smallest ensemble loss $(losses[i])"
    close(candidates)
    ret = automl.candidates[i]
    empty!(automl.candidates)
    return ret, losses[i]
end

function Base.:run(automl::AutoML)
    open(joinpath(automl.WORKSPACE, "ensembles.txt"), "w") do io 
        write(io, "")
    end
    for i = 1:options.automl.max_iter
        println("------------------------------------ Iteration = $i ------------------------------------ ")
        automl.candidates = automl.generate_subnetwork(automl)
        automl.most_recent_subnetworks = copy(automl.candidates)
        options.automl.verbose && @info "[AutoML] Generating new neural networks: $([s for s in automl.candidates])"
        if automl.status == 0 
            rep = rand(1:length(automl.subnetworks))
            options.automl.verbose && @info "[AutoML] Replacing subnetwork $(rep) (name = $(automl.subnetworks[rep]))"
        else 
            rep = 0
        end
        t = @elapsed subnetwork, score = execute_subnetwork(automl, rep)
        if length(automl.losses)==0 || score < minimum(automl.losses)
            options.automl.verbose && printstyled("[AutoML] Updating automl.losses ($score) and subnetwork (name = $(subnetwork)); time elapsed $t sec\n", color=:green)
            push!(automl.losses, score)
            if automl.status == 0
                if rep>0
                    automl.subnetworks[rep] = subnetwork
                else 
                    automl.subnetworks[end] = subnetwork
                end
            else
                push!(automl.subnetworks, subnetwork)
            end
            automl.most_recent_subnetworks = [subnetwork]
            open(joinpath(automl.WORKSPACE, "ensembles.txt"), "w") do io 
                write(io, join(automl.subnetworks, "\n"))
            end
            options.automl.verbose && printstyled("Current ensemble model: $(join([s for s in automl.subnetworks], " + "))\n", color=:green, bold=true)
            automl.status = 0 
        else
            push!(automl.losses, automl.losses[end])
            options.automl.verbose && printstyled("[AutoML] Better architecture not found, current best = $(automl.losses[end]), network = $(join([s for s in automl.subnetworks], " + ")); time elapsed $t sec\n", color=:red)
            options.automl.verbose && printstyled("Current ensemble model: $(join([s for s in automl.subnetworks], " + "))\n", color=:red, bold=true)
            automl.status += 1
        end

        open(joinpath(automl.WORKSPACE, "losses.txt"), "w") do io 
            write(io, join(map(string, automl.losses), "\n"))
        end
        open(joinpath(automl.WORKSPACE, "subnetworks$i.txt"), "w") do io 
            write(io, join(automl.subnetworks, "\n"))
        end
        open(joinpath(automl.WORKSPACE, "subnetworks_history.txt"), "a") do io 
            for (k, v) in automl.subnetworks_history
                write(io, "$k $v\n")
            end
        end
    end
    options.automl.verbose && @info "[AutoML] AutoML completed successfully!"
    options.automl.verbose && @info "Best neural network architecture (loss = $(automl.losses[end])): $(join([s for s in automl.subnetworks], " + "))"
    options.automl.verbose && @info "Loss history: $(automl.losses)"
end

function average_ensemble(sess::PyObject, ensemble_names::Array{String}, create_neural_network, compute_loss_function, WORKSPACE::String)
    function nn(x)
        sum([create_neural_network(s)(x) for s in ensemble_names])/length(ensemble_names)
    end
    loss = compute_loss_function(nn)
    init(sess)
    for s in ensemble_names
        ADCME.load(sess, "$(WORKSPACE)/$s/data.mat", nowarn=true)
    end
    loss
end


function generate_subnetwork(automl::AutoML)
    if length(automl.subnetworks)==0
        return ["5_3";"5_4";"5_5"]
    end
    c = split(rand(automl.most_recent_subnetworks), '_')
    hidden_size = parse(Int64, c[1])
    num_layers = parse(Int64, c[2])
    return [
        string(hidden_size+1)*"_"*string(num_layers+1);
        string(hidden_size+1)*"_"*string(num_layers);
        string(hidden_size)*"_"*string(num_layers+1)
    ]
end

function create_neural_network(name::String, output_dim::Int64)
    c = split(name, '_')
    hidden_size = parse(Int64, c[1])
    num_layers = parse(Int64, c[2])
    nn = x->fc(x, [hidden_size*ones(Int64, num_layers)...,output_dim], "nn$name")
    return nn 
end