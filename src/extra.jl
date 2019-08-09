using Random
export customop,
torchexample,
xavier_init,
gan,
klgan,
wgan,
rklgan,
lsgan,
test_customop


"""
    customop(torch=false)

Create a new custom operator.

# example
```julia-repl
julia> customop() # create an editable `customop.txt` file
[ Info: Custom operator wrapper generated; Torch is disabled

julia> customop() # after editing `customop.txt`, call it again to generate interface files.
[ Info: Custom operator wrapper generated; Torch is disabled
```
The option `torch` adds support for `PyTorch` backend in `CMakeLists.txt`
"""
function customop(torch=false)
    py_dir = "$(@__DIR__)/../examples/custom_op/template"
    if !("custom_op.txt" in readdir("."))
        cp("$(py_dir)/custom_op.example", "custom_op.txt")
        @info "Edit custom_op.txt for custom operators"
    else
        run(`python $(py_dir)/customop.py custom_op.txt $py_dir $(torch ? "" : "# ")`)
        @info "Custom operator wrapper generated; Torch is $(torch ? "enabled" : "disabled")"
    end
end

function torchexample()
    filename = "$(@__DIR__)/../examples/torch/laexample.cpp"
    s = read(filename, String)
    println(s)
end

function xavier_init(size, dtype=Float64)
    in_dim = size[1]
    xavier_stddev = 1. / sqrt(in_dim / 2.)
    return randn(dtype, size...)*xavier_stddev
end

"""
D_loss, G_loss = klgan(P::PyObject, Q::PyObject, discriminator::Function)

return discriminator loss and generator loss for KL divergence
`P` is the real distribution, `Q` is the generated distribution, 
`discriminator` is a critic function that outputs values in (0,1) (e.g. the last activation function is sigmoid)
"""
function klgan(P::PyObject, Q::PyObject, discriminator::Function)
    D_real = discriminator(P)
    D_fake = discriminator(Q)
    D_loss = -mean(log(D_real) + log(1-D_fake))
    G_loss = mean(log((1-D_fake)/D_fake))
    D_loss, G_loss
end

"""
D_loss, G_loss = gan(P::PyObject, Q::PyObject, discriminator::Function)

return discriminator loss and generator loss for JS divergence
`P` is the real distribution, `Q` is the generated distribution, 
`discriminator` is a critic function that outputs values in (0,1) (e.g. the last activation function is sigmoid)
"""
function gan(P::PyObject, Q::PyObject, discriminator::Function)
    D_real = discriminator(P)
    D_fake = discriminator(Q)
    D_loss = -mean(log(D_real) + log(1-D_fake))
    G_loss = -mean(log(D_fake))
    D_loss, G_loss
end

"""
D_loss, G_loss = wgan(P::PyObject, Q::PyObject, discriminator::Function)

return discriminator loss and generator loss for 1 Wasserstein
`P` is the real distribution, `Q` is the generated distribution, 
No constraint is imposed on discriminator
`clamp` is required for the discriminator weights
"""
function wgan(P::PyObject, Q::PyObject, discriminator::Function)
    D_real = discriminator(P)
    D_fake = discriminator(Q)
    D_loss = mean(D_fake)-mean(D_real)
    G_loss = -mean(D_fake)
    D_loss, G_loss
end

"""
D_loss, G_loss = rklgan(P::PyObject, Q::PyObject, discriminator::Function)

return discriminator loss and generator loss for reverse KL divergence
`P` is the real distribution, `Q` is the generated distribution, 
`discriminator` is a critic function that outputs values in (0,1) (e.g. the last activation function is sigmoid)
"""
function rklgan(P::PyObject, Q::PyObject, discriminator::Function)
    D_real = discriminator(P)
    D_fake = discriminator(Q)
    G_loss = mean(log((1-D_fake)/D_fake))
    D_loss = -mean(log(D_fake)+log(1-D_real))
    D_loss, G_loss
end


"""
D_loss, G_loss = lsgan(P::PyObject, Q::PyObject, discriminator::Function)

return discriminator loss and generator loss for least square
`P` is the real distribution, `Q` is the generated distribution, 
`discriminator` is a critic function that outputs values in (0,1) (e.g. the last activation function is sigmoid)
1 for real, 0 for fake
"""
function lsgan(P::PyObject, Q::PyObject, discriminator::Function)
    D_real = discriminator(P)
    D_fake = discriminator(Q)
    D_loss = mean((D_real-1)^2+D_fake^2)
    G_loss = mean((D_fake-1)^2)
    D_loss, G_loss
end

export traintestdev
function traintestdev(n::Int64, train::Float64=0.64, test::Float64=0.2)
    rn = randperm(n)
    if train+test>1 || train<0 || test<0
        error("invalid train and test set size")
    end
    dev = 1 - train-test
    return rn[1:Int64(round(train*n))], rn[Int64(round(train*n))+1:Int64(round((train+test)*n))],
                rn[Int64(round((train+test)*n)):end]
end


function test_customop()
    dir = pwd()
    @info "Test: $(@__DIR__)/../examples/while_loop/DirichletBD"
    cd("$(@__DIR__)/../examples/while_loop/DirichletBD")
    if !isdir("build")
        mkdir("build")
    end
    cd("build")
    run(`cmake ..`)
    run(`make -j`)
    cd("..")
    include("gradtest.jl")
    

    @info "Test: $(@__DIR__)/../examples/while_loop/SparseSolver"
    cd("$(@__DIR__)/../examples/while_loop/SparseSolver")
    if !isdir("build")
        mkdir("build")
    end
    cd("build")
    run(`cmake ..`)
    run(`make -j`)
    cd("..")
    include("gradtest.jl")

    cd(dir)
end 