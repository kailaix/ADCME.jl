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
run(automl)