function predict!(feat, x, model::NGRC)
    @unpack n, k, weight = model

    model.f(feat, x)
    
    # shift the delayed states left by k
    older_states = @view x[1:n*(k-1)]
    newer_states = @view x[end-n*(k-1)+1:end]
    copy!(older_states, newer_states)

    # update the most recent state (last n elements)
    out = @view x[end-n+1:end]
    mul!(out, weight, feat, 1., 1.)
    nothing
end

function simulate(model, x₀, N)
    @unpack n, k = model

    x_save = zeros(n, N)
    @views x_save[:, 1:k] .= x₀

    x = copy(vec(x₀))

    feat = similar(x, num_features(model))

    for t in k:N
        predict!(feat, x, model)
        @views x_save[:, t] .= x[end-n+1:end]
    end

    return x_save
end

