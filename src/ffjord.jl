export FFJORDModel, ffjord_loss_4obj, ffjord_loss_1obj

default_tspan = (0.0, 1.0)
default_sensealg = InterpolatingAdjoint(
    ;
    autodiff=true,
    chunk_size=0,
    autojacvec=ZygoteVJP(),
)

function ffjord_loss_part_gvar(mdl::DiffEqFlux.CNFLayer, data::Matrix{Float64}, move::MLJFlux.Mover;
        regularize::Bool=false, monte_carlo::Bool=false, rλ₁::Float64=0.01, rλ₂::Float64=0.01, batch_size::Int64=32)
    global vl, itr = nothing, Base.Iterators.cycle(broadcast(x -> hcat(x...), Base.Iterators.partition(eachcol(data), batch_size)))
    function p_loss(θ::Vector)
        global vl, itr = Base.Iterators.peel(itr)
        logpx, λ₁, λ₂ = mdl(vl, θ, move(randn(eltype(vl), size(vl))); regularize, monte_carlo)
        mean(-logpx .+ rλ₁ * λ₁ .+ rλ₂ * λ₂)
    end
    p_loss
end

function ffjord_loss_part_arr(mdl::DiffEqFlux.CNFLayer, data::Matrix{Float64}, move::MLJFlux.Mover;
        regularize::Bool=false, monte_carlo::Bool=false, rλ₁::Float64=0.01, rλ₂::Float64=0.01, batch_size::Int64=32)
    vl = Any[nothing]
    itr = Any[nothing]
    vl[], itr[] = nothing, Base.Iterators.cycle(broadcast(x -> hcat(x...), Base.Iterators.partition(eachcol(data), batch_size)))
    function p_loss(θ::Vector)
        vl[], itr[] = Base.Iterators.peel(itr[])
        logpx, λ₁, λ₂ = mdl(vl[], θ, move(randn(eltype(vl[]), size(vl[]))); regularize, monte_carlo)
        mean(-logpx .+ rλ₁ * λ₁ .+ rλ₂ * λ₂)
    end
    p_loss
end

function ffjord_loss_rand(mdl::DiffEqFlux.CNFLayer, data::Matrix{Float64}, move::MLJFlux.Mover;
        regularize::Bool=false, monte_carlo::Bool=false, rλ₁::Float64=0.01, rλ₂::Float64=0.01, batch_size::Int64=32)
    function p_loss(θ::Vector)
        batch_data = hcat(rand(collect(eachcol(data)), batch_size)...)
        logpx, λ₁, λ₂ = mdl(batch_data, θ, move(randn(eltype(batch_data), size(batch_data))); regularize, monte_carlo)
        mean(-logpx .+ rλ₁ * λ₁ .+ rλ₂ * λ₂)
    end
    p_loss
end

MLJBase.@mlj_model mutable struct FFJORDModel <: PDFEstimator
    n_hidden_ratio::Int64 = 2::(_ > 0)
    tspan::Tuple{Float64, Float64} = default_tspan
    actv::Function = tanh
    basedist::Union{Distribution, Nothing} = nothing::(isnothing(_) || eltype(_) == model.dtype)
    rλ₁::Float64 = 0.01
    rλ₂::Float64 = 0.01

    sol_met::OrdinaryDiffEqAlgorithm = Tsit5()

    adtype::SciMLBase.AbstractADType = GalacticOptim.AutoZygote()
    sensealg::SciMLBase.AbstractSensitivityAlgorithm = default_sensealg

    optms::Vector{OptM} = short_optms

    regularize::Bool = true
    monte_carlo::Bool = true

    loss::Function = ffjord_loss_rand
    batch_size::Int64 = 32

    dtype::Union{Type{Float64}, Type{Float32}, Type{Float16}} = Float64
    acceleration::AbstractResource = CPU1()::(_ in (CPU1(), CUDALibs()))
end

function MLJBase.fit(model::FFJORDModel, verbosity, X)
    x = collect(MLJBase.matrix(X)')

    n_vars = size(x, 1)
    move = MLJFlux.Mover(model.acceleration)
    n_hidden = model.n_hidden_ratio * n_vars
    tspan = convert.(model.dtype, model.tspan)
    nn = Chain(
        Dense(n_vars, n_hidden, model.actv),
        Dense(n_hidden, n_hidden, model.actv),
        Dense(n_hidden, n_vars, model.actv),
    )
    if model.dtype == Float64
        nn = f64(nn)
    elseif model.dtype == Float32
        nn = f32(nn)
    else
        nn = Flux.paramtype(model.dtype, nn)
    end
    nn = move(nn)
    ffjord_mdl = FFJORD(nn, tspan, model.sol_met; model.basedist, model.sensealg)
    x = move(x)
    lss_f = model.loss(ffjord_mdl, x, move; model.regularize, model.monte_carlo, model.rλ₁, model.rλ₂, model.batch_size)
    res = optimizeit(model, lss_f, ffjord_mdl.p)

    learned_ffjord_mdl = FFJORD(nn, tspan, model.sol_met; model.basedist, model.sensealg, p=res.u)

    fitresult = (learned_ffjord_mdl, res, n_vars)
    cache = nothing
    report = nothing
    fitresult, cache, report
end

function MLJBase.transform(model::FFJORDModel, fitresult, Xnew)
    xnew = collect(MLJBase.matrix(Xnew)')
    learned_ffjord_mdl, res, n_vars = fitresult

    move = MLJFlux.Mover(model.acceleration)
    xnew = move(xnew)
    logpx, λ₁, λ₂ = learned_ffjord_mdl(xnew, res.u, move(randn(eltype(xnew), size(xnew))); model.regularize, model.monte_carlo)

    ynew = exp.(logpx)
    ynew = collect(ynew')
    ynew = DataFrame(ynew, :auto)
end

function MLJBase.fitted_params(model::FFJORDModel, fitresult)
    learned_ffjord_mdl, res, n_vars = fitresult

    (
        learned_params=res.u,
        learned_ffjord_mdl=learned_ffjord_mdl,
        res=res,
        n_vars=n_vars,
    )
end
