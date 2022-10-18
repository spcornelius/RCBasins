# NextGenRCBasins

## Installation
Clone this repository, enter the working directory, and start Julia:

```bash
git clone git@github.com:spcornelius/NextGenRCBasins.git
cd NextGenRCBasins
julia --project
```

Now instantiate the project (install all dependencies)

```julia
julia> using Pkg
julia> Pkg.instantiate()
```

## Package Overview
This package implements the Next Generation Reservoir Computing framework. It defines methods to train and simulate these models from arbitrary time series data. This package also implements the magnetic pendulum dynamics (see `MagneticPendulum`), and utility functions for mapping the basins of attraction of either the real ODEs or a trained NG-RC model in parallel using multiple processors.

## Getting started
You can understand how to use everything by running the example script in `/scripts`:

```bash
julia --project scripts/example.jl
```

This script does the following:
- Trains a NG-RC model from multiple trajectories of the magnetic pendulum system
- Plots a sample of the training predictions vs. the real trajectories
- Calculates the basins according to the real ODEs and according to the trianed NG-RC model
- Plots the real vs. predicted basins and highlights any mis-classifications

This script automatically parallelizes the basin computation over $n - 1$ CPUs, where $n$ is the total number of CPU cores in your system.