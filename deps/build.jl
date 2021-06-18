if haskey(ENV, "docker_build")
    include("build-docker.jl")
else 
    include("build-standard.jl")
end