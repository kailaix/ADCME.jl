# This file implements radial basis functions
# Inspired by MATLAB code: https://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/
export kMeans, evaluateRBFN,computeRBFBetas
using Statistics
using Random


                            #####################
                            # k-means utilities #
                            #####################

function computeCentroids(X::Array{Float64}, prev_centroids::Array{Float64}, memberships::Array{T}, k::T) where T<:Integer
    m,n = size(X)
    centroids = zeros(k, n)
    
    # For each centroid...
    for i = 1 : k
        # If no points are assigned to the centroid, don't move it.
        if !any(memberships .== i)
            centroids[i, :] = prev_centroids[i, :]
        # Otherwise, compute the cluster's new centroid.
        else
            # Select the data points assigned to centroid k.
            points = X[memberships .== i, :]
    
            # Compute the new centroid as the mean of the data points.
            centroids[i, :] = mean(points, dims=1)    
        end
    end
    centroids
end
    
function findClosestCentroids(X::Array{Float64}, centroids::Array{Float64})
    #  Set 'k' to the number of centers.
    k = size(centroids, 1)
    
    #  Set 'm' to the number of data points.
    m = size(X, 1)
    
    #  'memberships' will hold the cluster numbers for each example.
    memberships = zeros(m)
    
    #  Create a matrix to hold the distances between each data point and
    #  each cluster center.
    distances = zeros(m, k)
    
    diffs = zeros(m)
    #  For each cluster...
    for i = 1 : k
        
     
        #  Subtract centroid i from all data points.
        for j = 1:m
            diffs[j] = sum((X[j,:] - centroids[i, :]).^2)
        end
        
        
        #  Take the sum of the squared differences.
        distances[:, i] = diffs
    
    end
    
    #  Find the minimum distance value, also set the index of 
    #  the minimum distance value (in this case the indeces are 
    #  equal to the cluster numbers).
    memberships = argmin(distances, dims=2)
    memberships = [memberships[j].I[2] for j = 1:length(memberships)]
end


function kMeans(X::Array{Float64}, initial_centroids::Array{Float64}, max_iters::Int64)
    local memberships
    
    #  Get 'k' from the size of 'initial_centroids'.
    k = size(initial_centroids, 1)
    
    centroids = initial_centroids
    prevCentroids = centroids
    
    #  Run K-Means
    for i = 1 : max_iters
        
        # #  Output progress
        # println("K-Means iteration $i / $max_iters")
        
        #  For each example in X, assign it to the closest centroid
        memberships = findClosestCentroids(X, centroids)
            
        #  Given the memberships, compute new centroids
        centroids = computeCentroids(X, centroids, memberships, k)
        
        #  Check for convergence. If the centroids haven't changed since
        #  last iteration, we've converged.
        if prevCentroids == centroids
            # println("  Stopping after $i iterations.")
            break
        end
    
        #  Update the 'previous' centroids.
        prevCentroids = centroids
        if i==max_iters
          error("kMeans does not converge. Increase `max_iter`.")
        end
    end

    centroids, memberships
end


function kMeansInitCentroids(X::Array{Float64}, k::Int64)
    
    centroids = zeros(k, size(X, 2))
    
    #  Randomly reorder the indices of examples
    randidx = randperm(size(X, 1))
    
    #  Take the first k examples as centroids
    centroids = X[randidx[1:k], :]
    
end

function kMeans(X::Array{Float64}, k::Int64, max_iters::Int64=100)
  initial_centroids = kMeansInitCentroids(X, k)
  centroids, memberships = kMeans(X, initial_centroids, max_iters)
end
    
    

                            #####################
                            #   RBF utilities   #
                            #####################


function computeRBFBetas(X::Array{Float64}, centroids::Array{Float64}, memberships::Array{T}) where T<:Integer
        
        numRBFNeurons = size(centroids, 1);
    
        #  Compute sigma for each cluster.
        sigmas = zeros(numRBFNeurons);
        
        #  For each cluster...
        for i = 1 : numRBFNeurons
            #  Select the next cluster centroid.
            center = centroids[i, :];
    
            #  Select all of the members of this cluster.
            members = X[(memberships .== i), :];
    
            #  Compute the average L2 distance to all of the members. 
        
            #  Subtract the center vector from each of the member vectors.
            sqrdDiffs = zeros(size(members,1))
            for i = 1:size(members,1)
                sqrdDiffs[i] = sum((members[i,:] - center).^2)
            end
                        
            #  Take the square root to get the L2 (Euclidean) distance.
            distances = sqrt.(sqrdDiffs);
    
            #  Compute the average L2 distance, and use this as sigma.
            sigmas[i] = mean(distances);
        end
    
        #  Verify no sigmas are 0.
        if (any(sigmas .== 0))
            error("One of the sigma values is zero!");
        end
        
        #  Compute the beta values from the sigmas.
        betas = 1 ./ (2 .* sigmas .^ 2);
        
end


function getRBFActivations(centers::PyObject, betas::PyObject, input::PyObject)  
      diffs = centers - input
      sqrdDists = sum(diffs^2, dims = 2)
      z = exp(-betas .* sqrdDists)  
  end

function evaluateRBFN_(Centers::PyObject, betas::PyObject, Theta::PyObject, input::PyObject, normalize::Bool=false)
  
    #  Compute the activations for each RBF neuron for this input.
    phis = getRBFActivations(Centers, betas, input);
    if normalize
      phis = phis/sum(phis)
    end
  
    z = Theta[1] + sum(Theta[2:end] .* phis)
end

function evaluateRBFN(Centers::PyObject, betas::PyObject, Theta::PyObject, input::PyObject, normalize::Bool=false)
    @assert length(size(betas))==1
    @assert length(size(Centers))==2
    @assert length(size(Theta))==1
    @assert length(betas)==size(Centers,1)
    @assert length(Theta)==size(Centers,1)+1
  
    if length(size(input))==1 
        @assert length(input)==size(Centers,2)
        return evaluateRBFN_(Centers, betas, Theta, input, normalize)
    elseif  size(input,1)==1
        @assert length(input)==size(Centers,2)
        v = evaluateRBFN_(Centers, betas, Theta, input[1], normalize)
        return reshape(v, 1)
    end
    @assert size(input,2)==size(Centers,2)

    N = size(input,1)
    ta = TensorArray(N)
    ta = write(ta, 1, evaluateRBFN_(Centers, betas, Theta, input[1], normalize))
    function cond0(i, ta)
      i<=N
    end
    function body(i, ta)
      ta = write(ta, i, evaluateRBFN_(Centers, betas, Theta, input[i], normalize))
      i+1, ta
    end
    i = constant(2, dtype=Int32)
    _, out = while_loop(cond0, body, [i, ta]; parallel_iterations=10)
    return stack(out)
end

