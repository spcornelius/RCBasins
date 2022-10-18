RealVecOrTuple = Union{AbstractVector{<:Real}, NTuple{<:Any, <:Real}}

struct MagneticPendulum{T <: Real}
    ω::T
    α::T
    h::T
    r_mag::Vector{Vector{T}}

    function MagneticPendulum{T}(ω, α, h, r_mag::AbstractVector) where {T <: Real}
        isempty(r_mag) && error("Must have at least one magnet.")
        all(length.(r_mag) .== 2) || 
        error("Each element of r_mag must have length 2.")
        
        r_mag = collect.(r_mag)
        r_mag = convert.(Vector{T}, r_mag)
        new{T}(ω, α, h, r_mag)
    end
end

function MagneticPendulum(ω::T1, α::T2, h::T3, 
                          r_mag::AbstractVector) where 
                          {T1<:Real, T2<:Real, T3<:Real}
    T = promote_type(T1, T2, T3, eltype.(r_mag)...)
    MagneticPendulum{T}(ω, α, h, r_mag)
end

MagneticPendulum(ω, α, h, r::Vararg{RealVecOrTuple, N}) where {N} =
    MagneticPendulum(ω, α, h, collect(r))

@inline magnet_dist(r, rₘ, h) = sqrt(sqeuclidean(r, rₘ) + h^2)

@inline function magnet_force(r, rₘ, h)
    d³ = magnet_dist(r, rₘ, h)^3
    @~ @. (rₘ - r)/d³
end

function rhs(du, u, p::MagneticPendulum, t=0.)
    @unpack ω, α, h, r_mag = p
    r = @view u[1:2]
    v = @view u[3:4]
    dr = @view du[1:2]
    dv = @view du[3:4]
    dr .= v
    @. dv = -ω^2*r - α*v
    for rₘ in r_mag
        dv .+= magnet_force(r, rₘ, h)
    end
    nothing
end

function rhs(u, p::MagneticPendulum, t=0.)
    du = similar(u)
    rhs(du, u, p, t)
    du
end

function jac(J, u, p::MagneticPendulum, t=0.)
    @unpack ω, α,  h, r_mag = p
    r = @view u[1:2]

    J[1, 3] = 1.
    J[2, 4] = 1.
    
    J[3:4, 1:2] .= 0.
    J[3, 1] = -ω^2
    J[4, 2] = -ω^2

    J[3, 3] = -α
    J[4, 4] = -α

    x, y = r
    for rₘ in r_mag
        xₘ, yₘ = rₘ
        d = magnet_dist(r, rₘ, h)
        c1 = 1/d^3
        c2 = d^5
        c3 = 3*(xₘ - x)*(yₘ - y)/c2
        c4 = 3/c2
        J[3, 1] += c4*(xₘ - x)^2 - c1
        J[3, 2] += c3
        J[4, 1] += c3
        J[4, 2] += c4*(yₘ - y)^2 - c1
    end
end