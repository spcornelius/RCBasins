abstract type AbstractFeatures end

(f::AbstractFeatures)(x::AbstractVector) = 
    f(reshape(x, :, 1), 1)

# any sub-type of AbstractFeatures should be callable, i.e. it must implement a function
#
# function (f::AbstractFeatures)(x::AbstractVector) end
#
# ...that takes in a delay-embedded NGRC state x (with k * d elements)
# and calculates all relevant feature terms from that state. It can return
# those features either as an AbstractVector or as an Iterator.

include("./polynomial.jl")
include("./pendulum_exact.jl")
include("./radial.jl")

export AbstractFeatures

# polynomial.jl
export PolynomialFeatures

# mpfeatures.jl
export ExactMagneticPendulumFeatures

# radial.jl
export RadialBasisFeatures
