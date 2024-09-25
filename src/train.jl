TrainDataSet = AbstractVector{<:AbstractMatrix{<:Real}}

function train!(model::NGRC, x_train::TrainDataSet; 
                regularization_coeff = 0.0)
    @unpack n, k = model 

    N_traj = length(x_train)
    W = model.weight

    # check sanity
    N_train = size.(x_train, 2)

    all(size.(x_train, 1) .== n) || 
        error("All training data sets must have exactly n = $n rows.")

    all(N_train .> k) || 
        error("All training data sets must have at least k = $k columns (data points).")

    # each column is a between successive training data points
    # (i.e., the targets of the model)
    Y = @views hcat((x[:, k+1:end] - x[:, k:end-1] for x in x_train)...)

    # space for feature embeddings
    𝒪 = zeros(num_features(model), sum(N_train) - N_traj * k)

    # columns between c[i] + 1 and c[i+1] (inclusive) in 𝒪 correspond to dataset i
    c = cumsum(N_train .- k)
    pushfirst!(c, 0)

    delay_embed(x) = vcat((@view x[:, i:end - k + i - 1] for i in 1:k)...)

    x_delay = hcat((delay_embed(x) for x in x_train)...)
    
    model.f.(eachcol(𝒪), eachcol(x_delay))
    W .= ((𝒪 * 𝒪' + regularization_coeff * I) \ (𝒪 * Y'))'

    # calculate predictions for each training set using 
    # the trained weights
    iter = zip(x_train, IterTools.partition(c, 2, 1))

    x_predict = map(iter) do (x, (c1, c2))
        𝒪_ = @view 𝒪[:, c1+1:c2]
        x_ = @view x[:, k:end-1]
        x_p = x_ + W * 𝒪_
        # add the k warm-up points from x_train[i] to the beginning
        # so that x_predict[i] has the same number of rows as x_train[i]
        return hcat(x[:, 1:k], x_p)
    end

    return x_predict
end

# single training trajectory
function train!(model, x_train::AbstractMatrix{<:Real}; kw...)
    x_predict = train!(model, [x_train]; kw...)
    return x_predict[1]
end

# Dataset (from DynamicalSystems.jl).
# The rows there are time and the columns the state space dimensions,
# so transpose to match our convention
function train!(model, d::AbstractDataset; kw...)
    x_predict = train!(model, Matrix(d)'; kw...)
    return x_predict[1]
end

# single ODE solution (from solve in OrdinaryDiffEq)
function train!(model, sol::AbstractTimeseriesSolution; kw...)
    x_predict = train!(model, Matrix(sol); kw...)
    return x_predict[1]
end

# Esnembled ODE solution
train!(model, x_train::AbstractEnsembleSolution; kw...) =
    train!(model, [Matrix(sol) for sol in x_train]; kw...)
