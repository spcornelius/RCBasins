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

function output_func(sol, p::MagneticPendulum)
    # return only the closest magnet at the end of the solution,
    # instead of the full solution itself (default)
    idx = closest_magnet(p, @view sol[1:2, end])
    (idx, false)
end

function get_basin(model, sol, N)
    x = @view sol[:, 1:model.k]
    test_predict = simulate(model, x, N)
    u_last = @view test_predict[:, end]
    closest_magnet(sol.prob.p, @view u_last[1:2])
end

function map_basins_real(p::MagneticPendulum, x₀_range, y₀_range;
                         t_max=100.0, alg=Vern9(),
                         ensemblealg=EnsembleDistributed(),
                         progress::Union{Bool, Progress}=!is_logging(stderr), 
                         kw...)
    trajectories = length(x₀_range) * length(y₀_range)

    prob = ODEProblem(rhs, zeros(4), (0., t_max), p)

    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=(prob, i, _) -> 
                                        prob_func(prob, i, 
                                                  x₀_range, y₀_range),
                                    output_func=(sol, _) -> output_func(sol, p)
                                    )
            
    basins = solve(ensemble_prob, alg, ensemblealg,
                    trajectories=trajectories; 
                    save_everystep=false, progress=progress, kw...)

    # ensemble solution is flat; reshape into a matrix with the same size
    # as the Cartesian product x₀_range × y₀_range
    return reshape(basins[:], length.((x₀_range, y₀_range))...)
end

function map_basins_ngrc(model::NGRC, Δt,
                         p::MagneticPendulum, x₀_range, y₀_range;
                         t_max=100.0, alg=Vern9(),
                         ensemblealg=EnsembleDistributed(),
                         progress::Union{Bool, Progress}=!is_logging(stderr), 
                         kw...)

    k = model.k
    N = Int64(ceil(t_max / Δt))

    trajectories = length(x₀_range) * length(y₀_range)

    # get warm-up states for all points
    prob = ODEProblem(rhs, zeros(4), (0., (k - 1) * Δt), p)
    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=(prob, i, _) -> 
                                        prob_func(prob, i, 
                                                  x₀_range, y₀_range)
                                    )
    sols = solve(ensemble_prob, alg, ensemblealg,
                 trajectories=trajectories; saveat=Δt, 
                 kw...)

    # simulate the NGRC model from each warm-up state, in parallel
    progress = isa(progress, Progress) ? progress : Progress(trajectories, enabled=progress)
        basins = progress_pmap(s -> get_basin(model, s, N), sols, progress=progress)

    # reshape into a matrix with the same size as the Cartesian 
    # product x₀_range × y₀_range
    return reshape(basins, length.((x₀_range, y₀_range))...)
end