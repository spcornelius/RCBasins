# tell if a given output stream is a TTY or not
# used for disabling progress bars on, e.g., HPC clusters
is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")

function SciMLBase.solve_batch(prob::EnsembleProblem, alg, ensemblealg::EnsembleDistributed, 
    II, pmap_batch_size; progress::Union{Bool, Progress}=false, kwargs...)
    
    wp = CachingPool(workers())

    progress = isa(progress, Progress) ? progress : Progress(length(II), enabled=progress)
    batch_data = progress_pmap(wp, II, progress=progress, 
                               batch_size = pmap_batch_size) do i
        batch_func(i, prob, alg; kwargs...)
    end
    tighten_container_eltype(batch_data)
end

function setup_workers()
    sysimage_file = unsafe_string(Base.JLOptions().image_file)
    project =  unsafe_string(Base.JLOptions().project)
    exeflags = ["--project=$project", "--sysimage=$sysimage_file"]

    # if local, add all but one CPU core as workers
    total_procs = length(Sys.cpu_info())
    n = max(total_procs - 1 - nprocs(), 0)

    mgr = haskey(ENV, "SLURM_JOBID") ? SlurmManager() : 
        Distributed.LocalManager(n, true)
    addprocs(mgr, exeflags=exeflags, topology=:master_worker)
end