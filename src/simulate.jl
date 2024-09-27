
function _matrix_view(x::AbstractArray, r::Integer, c::Integer)
    Base.ReshapedArray(x, (r, c), ())
end

_matrix_view(x::AbstractMatrix, ::Integer, ::Integer) = x

# If the NGRC state is contiguous, can make a strided matrix view,
# which dramatically speeds up things like x[:, 1:s:...], used later...
function _matrix_view(x::DenseArray, r::Integer, c::Integer)
    sreshape(x, r, c)
end

_matrix_view(x::DenseMatrix, ::Integer, ::Integer) = StridedView(x)

Base.@propagate_inbounds @views function predict!(x_new::AbstractArray{T}, x::AbstractArray, 
                                                  model::NGRC, t) where {T}
    @unpack n, k, s, weight = model
    n_substates = (k - 1) * s + 1

    @boundscheck begin 
        ss = state_size(model)
        length(x_new) == length(x) == ss ||
            error("Expected both x and x_new to have length $ss.")
    end

    # views where rows are dimensions (1:n) and columns are time
    x = _matrix_view(x, n, n_substates)
    x_new = _matrix_view(x_new, n, n_substates)
    g = get_tmp(model.cache, x)
    model.f(g, x[:, 1:s:n_substates])
    
    # shift the delayed states left by k
    if k > 1
        copy!(x_new[:, 1:end-1], x[:, 2:end])
    end

    copy!(x_new[:, end], x[:, end])

    # update the most recent state (last n elements)
    mul!(x_new[:, end], weight, g, one(T), one(T))
    nothing
end

@views function simulate(model, x₀::AbstractArray{T}, N) where {T}
    @unpack n, k, s = model
    n_substates = (k - 1) * s + 1
    ss = state_size(model)

    length(x₀) == ss || 
        error("Expected a warmed-up NGRC state of length $(ss). Got length(x₀) == $(length(x₀)).")

    x_save = zeros(T, n, N+1)
    @. x_save[1:ss] = x₀[:]
    x_new = zeros(T, n, n_substates)
    x = copy(reshape(x₀, n, n_substates))

    for t in (k-1)*s+1:N
        #@inbounds predict!(x_new, x_save[:, t-(k-1)*s:t], model, nothing)
        @inbounds predict!(x_new, x, model, nothing)
        # last n elements of x are the most recent state; save it
        @. x_save[:, t+1] = x_new[:, end]
        copy!(x, x_new)
    end

    return x_save
end

