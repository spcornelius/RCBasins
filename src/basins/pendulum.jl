function closest_magnet(p::MagneticPendulum, r)
    m = length(p.r_mag)
    # if either x or y is Inf or NaN, return a sentinel value
    # otherwise return the index of the closest magnet
    all(isfinite.(r)) ? argmin(euclidean(r, rₘ) for rₘ in p.r_mag) : m + 1
end

# utility functions for ensemble solution of ODEs

@inline function ith(i, x₀_range, y₀_range)
    idxs = CartesianIndices((1:length(x₀_range), 1:length(y₀_range)))
    ix, iy = idxs[i].I
    return (x₀_range[ix], y₀_range[iy])
end

function prob_func(prob, i, x₀_range, y₀_range)
    # create a new ODEProblem from prob with the initial conditions of
    # (x, y) set to the i-th value from the cartesian product
    # x₀_range × y₀_range
    # Initial conditions for the velocities are usually zero and 
    # are left unchanged)
    @views prob.u0[1:2] .= ith(i, x₀_range, y₀_range)
    prob
end

function output_func(sol, p::MagneticPendulum, pbar::Progress)
    # return only the closest magnet at the end of the solution,
    # instead of the full solution itself (default)
    next!(pbar)
    idx = closest_magnet(p, @view sol[1:2, end])
    (idx, false)
end

function get_basin(x₀, model::NGRC, p::MagneticPendulum,  N)
    trajectory = simulate(model, x₀, N)
    closest_magnet(p, @view trajectory[1:2, end])
end

function map_basins_real(p::MagneticPendulum, x₀_range, y₀_range;
                         t_max=100.0, alg=Vern9(),
                         show_progress::Bool=!is_logging(stderr), 
                         kw...)
    trajectories = length(x₀_range) * length(y₀_range)

    prob = ODEProblem(rhs, zeros(4), (0., t_max), p)
    pbar = Progress(trajectories; enabled=show_progress)
    prob_func_(prob, i, _) = prob_func(prob, i, x₀_range, y₀_range)
    output_func_(sol, _) = output_func(sol, p, pbar)

    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func_,
                                    output_func=output_func_)
            
    basins = solve(ensemble_prob, alg, EnsembleThreads(),
                    trajectories=trajectories; 
                    save_everystep=false, kw...)

    # ensemble solution is flat; reshape into a matrix with the same size
    # as the Cartesian product x₀_range × y₀_range
    return reshape(basins[:], length.((x₀_range, y₀_range))...)
end

function map_basins_ngrc(model::NGRC, Δt,
                         p::MagneticPendulum, x₀_range, y₀_range;
                         t_max=100.0, alg=Vern9(),
                         show_progress::Bool=!is_logging(stderr), 
                         kw...)
    @unpack k, s = model
    N = Int64(ceil(t_max / Δt))

    trajectories = length(x₀_range) * length(y₀_range)

    # Get warm-up states for all points. The NGRC state consists
    # of (k-1) * s + 1 substates, each of dimension n. So we need to
    # integrate for (k-1) * s additional time steps to augment
    # the initial condition.
    prob = ODEProblem(rhs, zeros(4), (0., (k-1) * s * Δt), p)

    prob_func_(prob, i, _) = prob_func(prob, i, x₀_range, y₀_range)
    output_func_(sol, i) = (Matrix(sol), false)
    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func_,
                                    output_func=output_func_)

    sols = solve(ensemble_prob, alg, EnsembleThreads(),
                 trajectories=trajectories; saveat=Δt, 
                 kw...)
    ics = [s for s in sols]

    # simulate the NGRC model from each warm-up state, in parallel
    pbar = Progress(trajectories; enabled=show_progress)

    # the model contains a cache vector for storing calculated features;
    # to avoid race conditions, each thread needs its own copy
    num_threads = Threads.nthreads()
    models = [deepcopy(model) for _ in 1:num_threads]
    ids = Channel{Int}(num_threads)
    for id in 1:num_threads
        put!(ids, id)
    end

    get_basin_(x₀) = begin
        id = take!(ids)
        model = models[id]
        b = get_basin(x₀, model, p, N)
        next!(pbar)
        put!(ids, id)
        b
    end

    basins = ThreadsX.map(get_basin_, ics)
    # reshape into a matrix with the same size as the Cartesian 
    # product x₀_range × y₀_range
    return reshape(basins, length.((x₀_range, y₀_range))...)
end