@with_kw struct PolynomialFeatures <: AbstractFeatures
    d_max::Int64

    function PolynomialFeatures(d_max::Integer)
        d_max ≥ 2 || error("Must have d_max ≥ 2.")
        new(d_max)
    end
end

function (feat::PolynomialFeatures)(x::AbstractMatrix{T}) where {T}
    # all unique monomials of degree between 2 and d_max formed
    # by the elements of x
    # Note: this includes "cross-terms" between different variables in 
    # different states (the k columns of x)
    return (prod(vars) for d in 2:feat.d_max for 
            vars in with_replacement_combinations(x, d))
end
