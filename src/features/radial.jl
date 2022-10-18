Idxs = Union{Integer, Colon, AbstractVector{<:Integer}}

@with_kw struct RadialBasisFeatures{F, C, I} <: AbstractFeatures
    kernel::F
    centers::Vector{Vector{C}}
    idxs::I = (:)

    function RadialBasisFeatures(kernel::F, centers::Vector{Vector{C}}, idxs::I) where 
            {F <: Function, C, I <: Idxs}
        lengths = length.(centers)
        minimum(lengths) == maximum(lengths) || 
            error("All centers must have the same length.")
        new{F, C, I}(kernel, centers, idxs)
    end
end

function (feat::RadialBasisFeatures)(x::AbstractMatrix{T}) where {T}
    return Iterators.map(product(eachcol(x), feat.centers)) do (state, c)
        r = @view state[feat.idxs]
        feat.kernel(euclidean(r, c))::T
    end
end