function build_feature_func(n::Integer, k::Integer, features::AbstractFeatures, bias::Bool)
    @variables x[1:n*k]

    bias_term = repeat([one(Num)], bias)
    lin_terms = x
    # when called, the AbstractFeatures subtypes expect a matrix of size d x k, so reshape
    nonlin_terms = vec(collect(features(reshape(x, n, k))))

    out = vcat(bias_term, lin_terms, nonlin_terms)

    f_oop, f_iip = build_function(out, x, expression=Val{false})
    function f(out::AbstractVector, x::AbstractVector)
        return f_iip(out, x)
    end

    function f(x::AbstractVector)
        return f_oop(x)
    end

    return f, length(out)
end

@with_kw struct NGRC{T, F}
    # dimension
    n::Int64

    # number of delays (k = 1 means no memory)
    k::Int64

    features::F

    # presence of bias term
    bias::Bool = true

    # runtime-generated function to calculate features from a n x k state
    f::Function

    # weight matrix
    weight::Matrix{T}
    
    function NGRC{T}(n::Integer, k::Integer, features::F; bias::Bool=true) where 
        {T <: Number, F <: AbstractFeatures}

        k ≥ 1 || error("must have k ≥ 1")

        f, m = build_feature_func(n, k, features, bias)

        # weights are initialized to 0 before training
        weight = zeros(T, n, m)

        new{T, F}(n, k, features, bias, f, weight)
    end
end

NGRC(args...) = NGRC{Float64}(args...)

num_features(model::NGRC) = size(model.weight, 2)
