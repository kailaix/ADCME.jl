if haskey(ENV, "DOCKER_BUILD")
    include("build-docker.jl")
else 
    include("build-standard.jl")
end