function build_feature_func(n::Integer, k::Integer, features::AbstractFeatures, bias::Bool)
    @variables x[1:n*k]

    bias_term = repeat([one(Num)], bias)
    lin_terms = x
    # when called, the AbstractFeatures subtypes expect a matrix of size d x k, so reshape
    nonlin_terms = vec(collect(features(reshape(x, n, k))))

    out = vcat(bias_term, lin_terms, nonlin_terms)

    f_oop, f_iip = build_function(out, x, expression=Val{false})
    function f(out, x)
        return f_iip(out, x)
    end

    function f(x)
        return f_oop(x)
    end

    return f, length(out)
end

@with_kw struct NGRC{T, featuresType, fType, cacheType}
    # dimension
    n::Int

    # number of delays (k = 1 means no memory)
    k::Int

    # time skipping between delayed states
    s::Int

    features::featuresType

    # presence of bias term
    bias::Bool

    # runtime-generated function to calculate features from a n x k state
    f::fType

    # weight matrix
    weight::Matrix{T}

    # cache for storing calculated features
    cache::cacheType
    
    function NGRC{T}(n::Integer, k::Integer, features::featuresType; 
                     bias::Bool = true, s::Integer = 1) where 
        {T <: AbstractFloat, featuresType <: AbstractFeatures}

        n ≥ 1 || error("must have n ≥ 1.")

        k ≥ 1 || error("must have k ≥ 1.")

        s ≥ 1 || error("must have s ≥ 1.")

        f, m = build_feature_func(n, k, features, bias)

        # weights are initialized to 0 before training
        weight = zeros(T, n, m)
        cache = DiffCache(zeros(T, m))

        new{T, featuresType, typeof(f), typeof(cache)}(n, k, s, features, bias, f, weight, cache)
    end
end

NGRC(args...; kw...) = NGRC{Float64}(args...; kw...)

num_features(model::NGRC) = size(model.weight, 2)

function state_size(model::NGRC)
    @unpack n, k, s = model
    return n * ((k-1) * s + 1)
end