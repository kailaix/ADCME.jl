using Statistics, MAT

export cifar10,
mnist,
mnist_fashion,
cifar100,
cifar10_aug

function cifar10()
    if !isdir(ENV["HOME"]*"/.julia/datadeps/cifar-10-batches-mat")
        dir = ENV["HOME"]*"/.julia/datadeps/"
        run(`wget https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz`)
        run(`tar -xvzf cifar-10-matlab.tar.gz`)
        mv("cifar-10-batches-mat", dir*"cifar-10-batches-mat")
        rm("cifar-10-matlab.tar.gz")
    end
    dir = ENV["HOME"]*"/.julia/datadeps/cifar-10-batches-mat"
    x_train = zeros(Float32, 50000, 3072)
    y_train = zeros(Int32, 50000)
    for i = 1:5
        file = matopen("$dir/data_batch_$i.mat")
        x_train[(i-1)*10000+1:i*10000,:] = Float32.(read(file, "data"))
        y_train[(i-1)*10000+1:i*10000] = Int32.(read(file, "labels"))
        close(file)
    end
    file = matopen("$dir/test_batch.mat")
    x_test = Float32.(read(file, "data"))
    y_test = Int32.(read(file, "labels"))
    close(file)
    d = [x_train;x_test]
    μ = mean(d); σ = std(d)
    x_train = (x_train .- μ)/(σ .+ 1e-7)
    x_test = (x_test .- μ)/(σ .+ 1e-7)
    x_train = reshape(x_train, 50000, 32, 32, 3)
    x_test = reshape(x_test, 10000, 32, 32, 3)
    x_train, y_train, x_test, y_test
end

function cifar10_aug()
    if !isdir(ENV["HOME"]*"/.julia/datadeps/cifar-10-aug")
        dir = ENV["HOME"]*"/.julia/datadeps/"
        mkdir(dir*"cifar-10-aug")
        pd = pwd()
        cd(dir*"cifar-10-aug")
        @info "download script..."
        run(`git clone https://github.com/kailaix/Cifar-10-Data-Augmentation`)
        @info "augmenting images..."
        run(`python Cifar-10-Data-Augmentation/generate_data.py`)
        @info "done!"
        rm("Cifar-10-Data-Augmentation", force=true, recursive=true)
        cd(pd)
    end
    dir = ENV["HOME"]*"/.julia/datadeps/cifar-10-aug"
    @info "loading data..."
    file = matopen("$dir/train.mat")
    x_train = Float32.(read(file, "x"))
    y_train = Int32.(read(file, "y"))
    close(file)
    file = matopen("$dir/test.mat")
    x_test = Float32.(read(file, "x"))
    y_test = Int32.(read(file, "y"))
    close(file)
    x_train, y_train, x_test, y_test
end


function mnist()
    dir = ENV["HOME"]*"/.julia/datadeps/mnist/"
    if !isdir(ENV["HOME"]*"/.julia/datadeps/mnist")
        mkpath(ENV["HOME"]*"/.julia/datadeps/mnist/")
        run(`wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O trainx.gz`)
        run(`wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O trainy.gz`)
        run(`wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O testx.gz`)
        run(`wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O testy.gz`)
        for file in ["trainx", "trainy", "testx", "testy"]
            run(`gunzip $file.gz`)
            mv(file, dir*file)
        end
    end

    IMAGEOFFSET = 16
    LABELOFFSET = 8

    NROWS = 28
    NCOLS = 28

    TRAINIMAGES = joinpath(dir, "trainx")
    TRAINLABELS = joinpath(dir, "trainy")
    TESTIMAGES = joinpath(dir, "testx")
    TESTLABELS = joinpath(dir, "testy")


    function imageheader(io::IO)
        magic_number = bswap(read(io, UInt32))
        total_items = bswap(read(io, UInt32))
        nrows = bswap(read(io, UInt32))
        ncols = bswap(read(io, UInt32))
        return magic_number, Int(total_items), Int(nrows), Int(ncols)
    end

    function labelheader(io::IO)
        magic_number = bswap(read(io, UInt32))
        total_items = bswap(read(io, UInt32))
        return magic_number, Int(total_items)
    end

    function rawimage(io::IO)
        img = Array{Int32}(undef, NCOLS, NROWS)
        for i in 1:NCOLS, j in 1:NROWS
          img[i, j] = Int32(read(io, UInt8))
        end
        return img
    end
      
    function rawimage(io::IO, index::Integer)
        seek(io, IMAGEOFFSET + NROWS * NCOLS * (index - 1))
        return rawimage(io)
    end
      
    rawlabel(io::IO) = Int(read(io, UInt8))
    
    function rawlabel(io::IO, index::Integer)
    seek(io, LABELOFFSET + (index - 1))
    return rawlabel(io)
    end
    
    getfeatures(io::IO, index::Integer) = vec(getimage(io, index))
      
    function images(set = :train)
        io = IOBuffer(read(set == :train ? TRAINIMAGES : TESTIMAGES))
        _, N, nrows, ncols = imageheader(io)
        [rawimage(io) for _ in 1:N]
    end
      
      """
          labels()
          labels(:test)
      Load the labels corresponding to each of the images returned from `images()`.
      Each label is a number from 0-9.
      Returns the 60,000 training labels by default; pass `:test` to retreive the
      10,000 test labels.
      """
    function labels(set = :train)
        io = IOBuffer(read(set == :train ? TRAINLABELS : TESTLABELS))
        _, N = labelheader(io)
        [rawlabel(io) for _ = 1:N]
    end

    X = images(:train)
    Y = images(:test)
    train_x = zeros(Int32, 60000, 28, 28, 1)
    test_x = zeros(Int32, 10000, 28, 28, 1)
    for i = 1:60000
        train_x[i, :,:,1] = X[i]
    end
    for i = 1:10000
        test_x[i, :,:,1] = Y[i]
    end    

    train_x, labels(:train), test_x, labels(:test)
end

function mnist_fashion()
    dir = ENV["HOME"]*"/.julia/datadeps/mnist_fashion/"
    if !isdir(ENV["HOME"]*"/.julia/datadeps/mnist_fashion")
        mkpath(ENV["HOME"]*"/.julia/datadeps/mnist_fashion/")
        run(`wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz -O trainx.gz`)
        run(`wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz -O trainy.gz`)
        run(`wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz -O testx.gz`)
        run(`wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz -O testy.gz`)
        for file in ["trainx", "trainy", "testx", "testy"]
            run(`gunzip $file.gz`)
            mv(file, dir*file)
        end
    end

    IMAGEOFFSET = 16
    LABELOFFSET = 8

    NROWS = 28
    NCOLS = 28

    TRAINIMAGES = joinpath(dir, "trainx")
    TRAINLABELS = joinpath(dir, "trainy")
    TESTIMAGES = joinpath(dir, "testx")
    TESTLABELS = joinpath(dir, "testy")


    function imageheader(io::IO)
        magic_number = bswap(read(io, UInt32))
        total_items = bswap(read(io, UInt32))
        nrows = bswap(read(io, UInt32))
        ncols = bswap(read(io, UInt32))
        return magic_number, Int(total_items), Int(nrows), Int(ncols)
    end

    function labelheader(io::IO)
        magic_number = bswap(read(io, UInt32))
        total_items = bswap(read(io, UInt32))
        return magic_number, Int(total_items)
    end

    function rawimage(io::IO)
        img = Array{Int32}(undef, NCOLS, NROWS)
        for i in 1:NCOLS, j in 1:NROWS
          img[i, j] = Int32(read(io, UInt8))
        end
        return img
    end
      
    function rawimage(io::IO, index::Integer)
        seek(io, IMAGEOFFSET + NROWS * NCOLS * (index - 1))
        return rawimage(io)
    end
      
    rawlabel(io::IO) = Int(read(io, UInt8))
    
    function rawlabel(io::IO, index::Integer)
    seek(io, LABELOFFSET + (index - 1))
    return rawlabel(io)
    end
    
    getfeatures(io::IO, index::Integer) = vec(getimage(io, index))
      
    function images(set = :train)
        io = IOBuffer(read(set == :train ? TRAINIMAGES : TESTIMAGES))
        _, N, nrows, ncols = imageheader(io)
        [rawimage(io) for _ in 1:N]
    end
      
      """
          labels()
          labels(:test)
      Load the labels corresponding to each of the images returned from `images()`.
      Each label is a number from 0-9.
      Returns the 60,000 training labels by default; pass `:test` to retreive the
      10,000 test labels.
      """
    function labels(set = :train)
        io = IOBuffer(read(set == :train ? TRAINLABELS : TESTLABELS))
        _, N = labelheader(io)
        [rawlabel(io) for _ = 1:N]
    end

    X = images(:train)
    Y = images(:test)
    train_x = zeros(Int32, 60000, 28, 28, 1)
    test_x = zeros(Int32, 10000, 28, 28, 1)
    for i = 1:60000
        train_x[i, :,:,1] = X[i]
    end
    for i = 1:10000
        test_x[i, :,:,1] = Y[i]
    end    

    train_x, labels(:train), test_x, labels(:test)
end

function cifar100()
    if !isdir(ENV["HOME"]*"/.julia/datadeps/cifar-10-batches-mat")
        dir = ENV["HOME"]*"/.julia/datadeps/"
        run(`wget https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz`)
        run(`tar -xvzf cifar-10-matlab.tar.gz`)
        mv("cifar-100-matlab", dir*"cifar-100-matlab")
        rm("cifar-10-matlab.tar.gz")
    end
    dir = ENV["HOME"]*"/.julia/datadeps/cifar-100-matlab"
    
    file = matopen("$dir/train.mat")
    x_train = Float32.(read(file, "data"))
    y_train = Int32.(read(file, "fine_labels"))
    close(file)

    file = matopen("$dir/test.mat")
    x_test = Float32.(read(file, "data"))
    y_test = Int32.(read(file, "fine_labels"))
    close(file)

    d = [x_train;x_test]
    μ = mean(d); σ = std(d)
    x_train = (x_train .- μ)/(σ .+ 1e-7)
    x_test = (x_test .- μ)/(σ .+ 1e-7)
    x_train = reshape(x_train, 50000, 32, 32, 3)
    x_test = reshape(x_test, 10000, 32, 32, 3)
    x_train, y_train, x_test, y_test
end