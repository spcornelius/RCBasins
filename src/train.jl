const TrainDataSet{T} = AbstractVector{<:AbstractMatrix{T}}

struct DelayEmbeddings{X <: AbstractMatrix}
    x::X
    k::Int
    s::Int
end

function delay_embeddings(x::AbstractMatrix, k::Integer, s::Integer)
    @assert k ≥ 1
    @assert s ≥ 1
    DelayEmbeddings(x, k, s)
end

@views function Base.iterate(e::DelayEmbeddings, i::Int = 0)
    @unpack x, k, s = e

    if i ≥ length(e)
        return nothing
    else
        return (x[:, i+1:s:s*(k-1)+i+1], i + 1)
    end
end

Base.IteratorSize(::Type{DelayEmbeddings}) = Base.HasLength()

Base.IteratorEltype(::Type{DelayEmbeddings}) = Base.HasEltype()

function Base.length(e::DelayEmbeddings)
    @unpack x, k, s = e
    ncol = size(x, 2)
    return ncol - (k-1)*s
end

# Each delay embedding is a view of an underlying matrix. Said
# view has a lot of type parameters, so easiest to let the compiler
# tell us the iterator element type. (This method should be evaluated
# statically, at compile time).
function Base.eltype(::DelayEmbeddings{X}) where {X}
    return Core.Compiler.return_type(view, Tuple{X, Colon, StepRange{Int, Int}})
end

# from a vector of matrices, where each Matrix i is
# n x T_i where T_i is the length of that trajectory.
# We're assuming the user has taken care to make the times evenly 
# spaced...
@views function train!(model::NGRC, x_train::TrainDataSet{T}; 
                       regularization_coeff = 0.0) where {T <: AbstractFloat}
    @unpack n, k, s = model 

    W = model.weight

    all(x -> size(x, 1) == n, x_train) || 
        error("All training data sets must have exactly n = $n rows.")

    # Last column of the warmup state (inclusive).
    # If x is a n x L training dataset, then x can produce (L - warmup) training
    # targets (and an equal number of corresponding feature vectors).
    warmup = (k-1)*s + 1

    all(x -> size(x, 2) ≥ warmup + 1, x_train) || 
        error("All training data sets must have at least (k-1) * s + 2  = $(warmup + 1) columns (data points).")

    # Each trajectory i yields a number training inputs/targets L[i]
    # equal to its length minus the number of warmup states.
    # For example if k = 3 and s = 2, the first 5 = (k-1)*s+1 
    # states are the warmup. So if we have 6 time steps, that's just enough
    # to get one training example (one set of features and one next-step 
    # target).
    L = [size(x, 2) - warmup for x in x_train]

    # The Ridge regression (below) requires single matrices containing
    # the next-step targets (ΔX) and features (𝒪) collected from all 
    # training trajectories. BlockedArrays allow this while keeping track
    # of which columns correspond to which trajectory.
    ΔX = BlockedArray{T}(undef, [n], L)
    𝒪 = BlockedArray{T}(undef, [num_features(model)], L)

    # Starting from the end of the warmup, fill in the next-step targets 
    # from each training trajectory. Each column of ΔX is a difference between two 
    # adjacent training data points from a single training trajectory.
    for (x, ΔX_) in zip(x_train, eachblock(ΔX))
        @. ΔX_ = x[:, warmup+1:end] - x[:, warmup:end-1]
    end

    # Fill in the corresponding feature vectors. Each column of 𝒪 will
    # represent the features calculated from a set of k states spaced by
    # s. For example, if k = 3 and s = 2, then the first column of 𝒪
    # will be features calculated from x(1), x(3), and x(5), which correspond
    # to the target in the first column of ΔX (namely, x(6) - x(5)).
    for (x, 𝒪_) in zip(x_train, eachblock(𝒪))
        # We go only up to end-1 because the very last state in each 
        # trajectory can't be used to calculate a feature vector; it's only
        # used to calculate the (final) next-step target!
        embeddings = DelayEmbeddings(x[:, 1:end-1], k, s)
        @assert length(embeddings) == size(𝒪_, 2)
        model.f.(eachcol(𝒪_), embeddings)
    end
    
    # Perform Ridge regression.
    W .= ((𝒪 * 𝒪' + regularization_coeff * I) \ (𝒪 * ΔX'))'

    x_predict = map(zip(x_train, eachblock(𝒪))) do (x, 𝒪_)
        # Add the (k-1)*s+1 warm-up points from x_train[i] to the beginning
        # so that x_predict[i] has the same number of columns as x_train[i].
        hcat(x[:, 1:warmup], x[:, warmup:end-1] + W * 𝒪_)
    end

    return x_predict
end

# From a single training trajectory, i.e. a n x T Matrix where
# n is the phase space dimension and T is the number of time steps.
# Note: NGRC assumes the time points are evenly spaced, which is the
# user's responsibility.
function train!(model, x_train::AbstractMatrix{<:AbstractFloat}; kw...)
    x_predict = train!(model, [x_train]; kw...)
    return only(x_predict)
end

# from a `Dataset` (from DynamicalSystems.jl).
function train!(model, d::AbstractDataset; kw...)
    # In DynamicalSystems.jl, the output of `trajectory` is a Matrix where the 
    # rows represent time and the columns the state space dimensions, so 
    # transpose to match our convention.
    x_predict = train!(model, Matrix(d)'; kw...)
    return only(x_predict)
end

# from a single ODE solution
function train!(model, sol::AbstractTimeseriesSolution; kw...)
    x_predict = train!(model, Matrix(sol); kw...)
    return only(x_predict)
end

# from an ensemble ODE solution
train!(model, x_train::AbstractEnsembleSolution; kw...) =
    train!(model, [Matrix(sol) for sol in x_train]; kw...)
