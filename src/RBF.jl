# This file implements radial basis functions
# Inspired by MATLAB code: https://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/
export kMeans, evaluateRBFN,computeRBFBetas
using Statistics
using Random


                            #####################
                            # k-means utilities #
                            #####################

@doc "COMPUTECENTROIDS Computes the centroids for the k clusters by taking the 
average value of the data points in the cluster.
   centroids = COMPUTECENTROIDS(X, memberships, k) returns the new centroids 
   by computing the means of the data points assigned to each centroid. 

   Parameters
     X           - The dataset, with one sample per row.
     memberships - The index of the centroid that the corresponding data point
                   in X belongs to (a value in the range 1 - k).
     k           - The number of clusters.

   Returns
     A matrix of centroids, with k rows where each row contains a centroid.
     X contains 'm' samples with 'n' dimensions each."->
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
    
@doc """FINDCLOSESTCENTROIDS Computes the centroid memberships for every sample in X.
memberships = FINDCLOSESTCENTROIDS (X, centroids) Returns the index of the
closest centroid for every data point in X.

In k-means clustering, data points are assigned to a cluster based on the
Euclidean distance between the data point and the cluster centroids.

Parameters
  X         - The data set, with one sample per row.
  centroids - The current centroids, one per row.

Returns
  A column vector containing the index of the closest centroid (a value
  between 1 - k) for each corresponding data point in X."""->
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
        
        #  Rather than compute the full euclidean distance, we just compute
        #  the squared distance (i.e., ommit the sqrt) since this is sufficient
        #  for performing distance comparisons.
        
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

@doc """KMEANS Run the k-means clustering algorithm on the data set X.
[centroids, memberships] = KMEANS(X, initial_centroids, max_iters)
Runs the k-means algorithm on the dataset X where 'k' is given by the
number of initial centroids in 'initial_centroids'

This function will test for convergence and stop when the centroids don't
change from one iteration to the next. It will also break after 'max_iters'
iterations.  

The initial centroids should all be unique. They are typically taken 
randomly from the data set. See the 'kMeansInitiCentroids' function for
selecting random, unique points from X as your initial centroids. Note that
the choice of initial centroids will affect the final clusters. To get
repeatable results from k-means, you need to use the same initial 
centroids.

Parameters
  X                 - The dataset, with one example per row.
  initial_centroids - The initial centroids to use, one per row (there
                      should be 'k' rows).
  max_iters         - The maximum number of iterations to run (k-means will
                      stop sooner if it converges).
Returns
  centroids    -  A k x n matrix of centroids, where n is the number of 
                 dimensions in the data points in X.
  memberships  - A column vector containing the index of the assigned 
                 cluster (a value between 1 - k) for each corresponding 
                 data point in X."""->
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

@doc """KMEANSINITCENTROIDS Randomly selects k different data points from X to use as
the initial centroids for k-Means clustering.
  centroids = KMEANSINITCENTROIDS(X, k) returns k initial centroids to be
  used with the k-Means on the dataset X

  Parameters
    X  - The dataset, one data point per row.
    k  - The number of cluster centers.

  Returns
    A matrix of centroids with k rows."""->
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


@doc """COMPUTERBFBETAS Computes the beta coefficients for all of the specified 
centroids.
  betas = computeRBFBetas(X, centroids, memberships)
  
  This function computes the beta coefficients based on the average distance
  between a cluster's data points and its center. The average distance is 
  called sigma, and beta = 1 / (2*sigma^2).

  Parameters:
    X           - Matrix of all training samples, one per row.
    centroids   - Matrix of cluster centers, one per row
    memberships - Vector specifying the cluster membership of each data point
                  in X. The membership is specified as the row index of the
                  centroid in 'centroids'.
                  
  Returns:
    A vector containing the beta coefficient for each centroid."""->
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


@doc """
 GETRBFACTIVATIONS Compute the activation values for all RBF neurons.
   z = getRBFActivations(centers, betas, input) Computes the activations for
   the given input for each RBF neuron specified by 'centers' and 'betas'.

   This function computes the RBF activation function for each of the RBF 
   neurons given the supplied 'input'. Each RBF neuron is described by a 
   prototype or "center" vector in 'centers', and a beta coefficient in
   'betas'. 

   Parameters
     centers  - Matrix of RBF neuron center vectors, one per row.
     betas    - Vector of beta coefficients for the corresponding RBF neuron.
     input    - Column vector containing the input.

   Returns
     A column vector containing the activation value (between 0 and 1) for
     each of the RBF neurons.
""" ->
function getRBFActivations(centers::PyObject, betas::PyObject, input::PyObject)  
      #  Subtract the input from all of the centers.
      #  diffs becomes an k x n matrix where k is the number of centers
      #  and n is the number of input dimensions.
      diffs = centers - input
      
      #  Take the sum of squared distances (the squared L2 distance).
      #  sqrdDists becomes a k x 1 vector where k is the number of centers.
      sqrdDists = sum(diffs^2, dims = 2)
  
      #  Apply the beta coefficient and take the negative exponent.
      z = exp(-betas .* sqrdDists)
  end

function evaluateRBFN_(Centers::PyObject, betas::PyObject, Theta::PyObject, input::PyObject, normalize::Bool=false)
  
    #  Compute the activations for each RBF neuron for this input.
    phis = getRBFActivations(Centers, betas, input);
    if normalize
      phis = phis/sum(phis)
    end
    
  #    Multiply the activations by the weights and take the sum. Do this for
  #  each category (output node). The result is a column vector with one row
  #  per output node.
  
    #  Theta = centroids x categories	  Theta' = categories x centroids
    #   phis = centroids x 1
    #      z = Theta' * phis = categories x 1
    z = Theta[1] + sum(Theta[2:end] .* phis)
end

@doc """
  EVALUATERBFN Computes the outputs of an RBF Network for the provided input.
    z = evaluateRBFN(Centers::PyObject, betas::PyObject, Theta::PyObject, input::PyObject, normalize::Bool=false) Evaluates the RBFN over
    given input using the provided parameters.

    This function computes the activation values of all of the RBFN neurons
    in the hidden layer using the provided 'Centers' and their 'betas', then
    computes the values for the output layer of the network using the 'Theta'
    coefficients

    Parameters
      Centers  - The prototype vectors for the RBF neurons.
      betas    - The beta coefficients for the corresponding prototypes.
      Theta    - The output weights to apply to the neuron activations.
      input    - The input vector to evaluate the RBFN over.
      normalize- Whether the basis function is normalized or not

    Returns
      A column vector representing the network's output value for each output
      node.
""" ->
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

