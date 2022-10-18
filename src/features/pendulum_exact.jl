@with_kw struct ExactMagneticPendulumFeatures <: AbstractFeatures
    p::MagneticPendulum
end

function (f::ExactMagneticPendulumFeatures)(x::AbstractMatrix{T}) where {T}
    p = f.p

    # for each combination of the states (k of them) and magnets
    # calculate the magnetic force term
    forces = Iterators.map(product(eachcol(x), p.r_mag)) do (state, rₘ)
        r = @view state[1:2]
        # magnet_force returns a LazyArray, so materialize it
        materialize(magnet_force(r, rₘ, p.h))
    end

    # each force is a 2-element vector (x and y components),
    # so flatten them
    return chain(forces...)
end

