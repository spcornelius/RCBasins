
module Config
    # make sure Revise tracks changes to constants
    __revise_mode__ = :eval

    using Colors
    using Makie
    using OrdinaryDiffEq

    # default values of pendulum parameters
    ω = 0.5
    α = 0.2
    h = 0.2

    # (x, y) coordinates of pendulum magnets
    r₁ = [1/√3, 0]
    r₂ = [-1/(2√3), 1/2]
    r₃ = [-1/(2√3), -1/2]

    # portion of (x, y) phase space to consider (for basins)
    lb = -1.5
    ub = 1.5
    grid_size = 200
    x₀_range = range(lb, ub, grid_size)
    y₀_range = range(lb, ub, grid_size)

    # how much time to integrate before deciding
    # which attractor a given IC went to
    t_basin = 100.0

    colors = [colorant"#B80D48",
              colorant"#F29724",
              colorant"#2B6A6C",
              colorant"#000000"]

    abstol = 1e-10
    reltol = 1e-10
    alg = Vern9()

    diffeq_kw = (abstol=abstol, reltol=reltol)

    almost_black = colorant"#262626"
    project_root = pkgdir(@__MODULE__)
    fig_root = joinpath(project_root, "figures")
    data_root = joinpath(project_root, "data")

    base_theme = Theme(
        font = "Arial", 
        fontsize = 10,
        fontcolor = almost_black,
        textcolor = almost_black,
        # figure_padding = (0, 0, 0, 0),
        backgroundcolor = :white,
        Axis = (
            titlesize = 12,
            xlabelsize = 12,
            ylabelsize = 12,
            topspinecolor = almost_black,
            bottomspinecolor = almost_black,
            rightspinecolor = almost_black,
            leftspinecolor = almost_black,
            xtickcolor = almost_black,
            ytickcolor = almost_black,
            yticklabelcolor = almost_black,
            xticklabelcolor = almost_black,
            xticksize=2,
            yticksize=2,
            xtickwidth=0.5,
            ytickwidth=0.5,
            spinewidth=0.75
        ),
        Label = (
            font = "Arial Bold",
            textsize = 16,
            color = almost_black,
            halign = :center
        ),
        Legend = (
            framevisible=false,
        )
    )
end


if !isdir(Config.data_root)
    mkdir(Config.data_root)
end

if !isdir(Config.fig_root)
    mkdir(Config.fig_root)
end