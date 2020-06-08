# Design Pattern 

Design patterns aim at providing reusable solutions for solving the challenges in the process of software development. The ultimate goal of design patterns is to avoid reinventing the wheels and making software flexible and resilient to change. Design patterns are neither concrete algorithms, nor programming templates, but ways of thinking. They are not always necessary if you can come up with very simple designs, which are actually more preferable in practice. Rather, they are "rules of thumb" that facilitates you when you have a hard time how to design the structure of your codes. 

We strive to make ADCME easily maintainable and extendable by using well-established design patterns for some design decisions. In this section, we describe some design patterns that are useful for programming ADCME. 


## Strategy Pattern 

The strategy pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable. The strategy pattern makes programmers wrap algorithms that are subject to frequent changes (e.g., extending) as **interfaces** instead of concrete implementations. For example, in ADCME, [`FlowOp`](@ref) is implemented using the strategy pattern. 

When you want to create a flow-based model, the structure [`NormalizingFlow`](@ref) has a method that performs a sequence of forward operations. The forward operations might have different combinations, which results in a large number of different normalizing flows. If we define a different normalizing flow structure for a different combination, there will be exponentially many such structures. Instead of defining a separate `forward` method for each different normalizing flow, we define an interface [`FlowOp`](@ref), which has a `forward` method. 

The interface is **implemented** with many concrete structures, which are called **algorithms** in the strategy pattern. These concrete `FlowOp`s, such as `SlowMAF` and `MAF`, have their specific `forward` implementations. Therefore, the system become easily extendable. When we have a new algorithm, we only need to add a new `FlowOp` instead of modifying `NormalizingFlow`. 


## Adaptor Pattern 


The adapter pattern converts the interface of a structure into another interface users expect. It is very useful to unify the APIs and reuses the existing functions, and thus reliefs users from memorizing many new functions. The typical structure of an adaptor has the form

```julia
struct Adaptor <: AbstractNewComponent
    o::LegacyComponent
    function new_do_this(adaptor, x)
        old_do_this(adaptor.o, x)
    end
    ...
end 
```

Here users work with `AbstractNewComponent`, whose concrete types implement a function `new_do_this`. However, we have a structure of type `LegacyComponent`, which has a function `old_do_this`. An adaptor pattern is used to match an old function call `old_do_this` to `new_do_this` in the new system.


An example of adaptor pattern is [`SparseTensor`](@ref), which wraps a `PyObject`. The operations on the `SparseTensor` is propagated to the `PyObject`, and therefore users can think in terms of the new `SparseTensor` data type. 

## Observer Pattern 

The observer pattern define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically. The basic pattern is 

**Subject**
```julia
struct Subject
    o # Observer 
    state
    function update(s)
        # some operations on `s.state`
    end
    function notify(s)
        # some operations on `s.o`
    end
end
```

**Observer**

```julia
struct Observer
    subjects::Array{Subject}
    function update(o)
        for s in o.subjects
            update(s)
        end
    end
end
```

For example, the `commitHistory` function in [NNFEM.jl](https://github.com/kailaix/NNFEM.jl) uses the observer pattern to update states from `Domain`, to `Element`, and finally to `Material`. 


## Decorator Pattern 

The decorator pattern attaches additional responsibilities to an object dynamically. Decorator patterns are very similar to adaptor patterns. The difference is that the input and output types of the decorator pattern are the same. For example, if the input is a `SparseTensor`, the output should also be a `SparseTensor`. The adaptor pattern converts `PyObject` to `SparseTensor`. Another difference is that the decorator pattern usually does not change the methods and fields of the structure. For example,

```julia
struct Juice
    cost::Float64 
end

function add_one_dollar(j::Juice)
    Juice(j.cost+1)
end
```

Then `add_one_dollar(j)` is still a `Juice` structure but the cost is increased by 1. You can also compose multiple `add_one_dollar`:

```julia
add_one_dollar(add_one_dollar(j))
```

In Julia, this can be done elegantly using macros. We do not discuss macros in this section but leave it to another section on macros. 

## Iterator Pattern

Iterator patterns provide a way to access the elements of an aggregate object sequentially without exposing its underlying representation. Julia has built-in iterator support. The `in` keyword loops through iterations and `collect` collects all the entries in the iterator. In Julia, to understand iteration, remember that the following code

```julia
for i in x
    # stuff
end
```

is a shorthand for writing

```julia
it = iterate(x)
while it !== nothing
    i, state = it
    # stuff
    it = iterate(x, state)
end
```

Therefore, we only need to implement `iterate`

```julia
iterate(iter [, state]) -> Union{Nothing, Tuple{Any, Any}}
```

## Factory Pattern

The factory pattern defines an interface for creating an object, but lets subclasses decide which class to instantiate. Simply put, we define a function that returns a specific structure

```julia
function factory(s::String)
    if s=="StructA"
        return StructA()
    elseif s=="StructB"
        return StructB()
    elseif s=="StructC"
        return StructC()
    else
        error(ArgumentError("$s is not understood"))
    end
end
```

For example, `FiniteStrainContinuum` in `NNFEM.jl` has a constructor

```julia
function FiniteStrainContinuum(coords::Array{Float64}, elnodes::Array{Int64}, props::Dict{String, Any}, ngp::Int64=2)
    eledim = 2
    dhdx, weights, hs = get2DElemShapeData( coords, ngp )
    nGauss = length(weights)
    name = props["name"]
    if name=="PlaneStrain"
        mat = [PlaneStrain(props) for i = 1:nGauss]
    elseif name=="Scalar1D"
        mat = [Scalar1D(props) for i = 1:nGauss]
    elseif name=="PlaneStress"
        mat = [PlaneStress(props) for i = 1:nGauss]
    elseif name=="PlaneStressPlasticity"
        mat = [PlaneStressPlasticity(props) for i = 1:nGauss]
    elseif name=="PlaneStrainViscoelasticityProny"
        mat = [PlaneStrainViscoelasticityProny(props) for i = 1:nGauss]
    elseif name=="PlaneStressViscoelasticityProny"
        mat = [PlaneStressViscoelasticityProny(props) for i = 1:nGauss]
    elseif name=="PlaneStressIncompressibleRivlinSaunders"
        mat = [PlaneStressIncompressibleRivlinSaunders(props) for i = 1:nGauss]
    elseif name=="NeuralNetwork2D"
        mat = [NeuralNetwork2D(props) for i = 1:nGauss]
    else
        error("Not implemented yet: $name")
    end
    strain = Array{Array{Float64}}(undef, length(weights))
    FiniteStrainContinuum(eledim, mat, elnodes, props, coords, dhdx, weights, hs, strain)
end
```

Every time we add a new material, we need to modify this structure. This is not very desirable. Instead, we can have a function

```julia
function get_element(s)
    if name=="PlaneStrain"
        PlaneStrain
    elseif name=="Scalar1D"
        Scalar1D
    ...
end
```

This can also be achieved via Julia macros, thanks to the powerful meta-programming feature in ADCME. 


## Summary

We have introduced some important design patterns that facilitate design maintainable and extendable software. Design patterns should not be viewed as rules to abide by, but they are useful principles in face of design difficulties. As mentioned, Julia provides powerful meta-programming features. These features can be used in the design patterns to simplify implementations. Meta-programming will be discussed in a future section.


