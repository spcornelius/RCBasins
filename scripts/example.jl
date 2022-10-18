using GLMakie
using Colors
using Distributed
using Distributions
using LaTeXStrings
using LinearAlgebra
using Makie
using NextGenRCBasins
using OrdinaryDiffEq
using ProgressMeter
using Random
using UnPack

############
# Parameters
############
@unpack colors, diffeq_kw, alg, t_basin = Config

# time resolution to sample training trajectories
Δt = 0.01

# dimension of phase space (d = 4 for magnetic pendulum)
n = 4

# total number of time-delayed states to use for NGRC embedding
# (k = 1 means no memory)
k = 5

# number of random initial conditions from which to generate training
# trajectories
Ntraj = 100

# number of time steps to use for training (frome each trajectory)
Ntrain = 5000

# number of time steps of the training fits to plot
# (this is just for cosmetics)
Nshow = 1000

# regularization parameter for ridge regression
# higher makes the weights of the model better-conditioned
λ = 1.0
t_train = (Ntrain + k - 1)*Δt 
train_times = 0:Δt:t_train

@unpack ω, α, h, r₁, r₂, r₃, lb, ub, x₀_range, y₀_range = Config
p = MagneticPendulum(ω, α, h, r₁, r₂, r₃)

#######################################
# Set up parallel workers (Distributed)
#######################################

setup_workers()
@everywhere using NextGenRCBasins, DifferentialEquations, ProgressMeter

########################
# Generate training data
########################
dist = Uniform(lb, ub)

# randomly generated ICs of the form (x, y, 0, 0), 
# where x and y are drawn uniformly and independenlty between lb and ub
ics = [[rand(dist, 2)..., 0., 0.] for _ in 1:Ntraj]

prob = ODEProblem(rhs, ics[1], (0.0, t_train), p)

ensemble_prob = EnsembleProblem(prob,
                                prob_func=(prob, i, repeat) -> 
                                    remake(prob, u0=ics[i]))

println("Getting training data...")
train_data = solve(ensemble_prob, alg, EnsembleDistributed(); 
                   saveat=Δt, trajectories=Ntraj, 
                   progress=true, diffeq_kw...)


#######################
# Define the NGRC model
#######################

#################################
# Example for RadialBasisFeatures
#################################
# need to provide a list of centers, the indices of the state variables 
# from which the distance (to each center) should be calculated, and the kernel 
# for the RBF.
# 
# For example, to calculate the radius using only the position coordinates (x, y), 
# from 100 randomly-chosen (2D) centers, with a kernel function similar to the
# pendulum force term, one could do:
# 
# centers = [rand(Uniform(lb, ub), 2) for _ in 1:100]
# @everywhere kernel(r) = 1/sqrt(r^2 + p.h^2)^3
# f = RadialBasisFeatures(kernel=kernel, centers=centers, idxs=1:2)
#
# Note: for use within this script, the kernel function must be defined on all workers
# hence the @everywhere.

# ###############################
# Example for PolynomialFeatures
# ###############################
# need to provide maximum degree d_max >= 2, e.g.

f = PolynomialFeatures(d_max=3)

###########################################
# Example for ExactMagneticPendulumFeatures
###########################################
# f = ExactMagneticPendulumFeatures(p=p)

model = NGRC(n, k, f)

######################
# Train the NGRC model
######################

println("Training model...")
train_time = @elapsed train_predict = train!(model, train_data; 
                                             regularization_coeff=λ)

println("Training took $train_time seconds.")
println("Condition number of model weights: $(cond(model.weight))")

#######################
# Plot the training fit
#######################

set_theme!(Config.base_theme)

# show fit to training data
fig = Figure()
t = collect(0.0:Δt:(Nshow-1)*Δt)

c = min(3, Ntraj)
g1 = fig[1, 1] = GridLayout()
axs = [Axis(g1[row, col]) for row in 1:4, col in 1:c]
ylabels = [L"x", L"y", L"v_x", L"v_y"]

for j=1:c
    axs[end, j].xlabel = L"t"
end

for (i, label) in enumerate(ylabels)
    ax = axs[i, 1]
    ax.ylabel = label
    ax.yaxis.elements[:labeltext].attributes[:rotation] = 0.0
    ax.ylabelpadding = 20.0
    # make sure all panels in a given row have the same x and y limits
    linkyaxes!(axs[i, :]...)
end

for i=1:4, j=1:c
    ax = axs[i, j]

    # only show the y tick labels on the first column
    if j > 1
        ax.yticklabelsvisible = false
    end
    lines!(ax, t, train_data[j][i, 1:Nshow])
    lines!(ax, t, train_predict[j][i, 1:Nshow])
end

#######################################################
# Get the basins for the NGRC model vs. the real system
#######################################################

println("Getting basins of real system...")
basins_real = map_basins_real(p, x₀_range, y₀_range; 
                              t_max=t_basin,
                              alg=alg, diffeq_kw...)

println("Getting basins of NGRC model...")
basins_ngrc = map_basins_ngrc(model, Δt, p, x₀_range, y₀_range; 
                              t_max=t_basin,
                              alg=alg, diffeq_kw...)


#################################################
# Plot the respective basins and highlight errors
#################################################

titles = ["Real", "NGRC", "Errors"]
g2 = fig[2, 1] = GridLayout()
axs = [Axis(g2[1, i], aspect=1, title=titles[i]) for i in 1:3]
for ax in axs
    hidespines!(ax)
    ax.xlabel = L"x_0"
end
axs[1].ylabel = L"y_0"

image!(axs[1], x₀_range, y₀_range, colors[basins_real])
image!(axs[2], x₀_range, y₀_range, colors[basins_ngrc])
basins_with_errors  = copy(basins_ngrc)
basins_with_errors[basins_ngrc .!= basins_real] .= length(colors)
image!(axs[3], x₀_range, y₀_range, colors[basins_with_errors])

pct_errors = 100 * sum(basins_with_errors .== length(colors))/length(basins_with_errors)
println("% mis-classified: $pct_errors")

display(fig)
