module RCBasins

import Configurations: from_dict

using Colors
using Combinatorics
using Configurations
using DifferentialEquations
using Distances
using Distributed
using DynamicalSystems
using IterTools
using LazyArrays
using LinearAlgebra
using Makie
using Mustache
using Parameters
using ProgressMeter
using RuntimeGeneratedFunctions
import SciMLBase: solve_batch
using SciMLBase: AbstractEnsembleSolution, AbstractTimeseriesSolution, batch_func, tighten_container_eltype
using Symbolics
using SlurmClusterManager
using UnPack
using YAML

include("util/configurations.jl")
include("util/makie.jl")
include("util/parallel.jl")

include("dynamics/dynamics.jl")

include("features/features.jl")

include("ngrc.jl")
include("train.jl")
include("simulate.jl")
include("config.jl")

include("basins/basins.jl")

# util/configurations.jl
export from_yaml

# util/makie.jl
export align_xlabels!, align_ylabels!

# util/parallel.jl
export setup_workers, is_logging

# ngrc.jl
export NGRC, num_features, build_feature_func, state_size

# train.jl
export train!

# simulate.jl
export simulate, predict!

# config.jl
export Config

end # module
