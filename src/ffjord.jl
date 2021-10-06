export FFJORDModel, ffjord_loss_4obj, ffjord_loss_1obj

default_tspan = (0.0, 1.0)

function ffjord_logpx(θ::Vector, mdl::DiffEqFlux.CNFLayer, data::Matrix{Float64}, move::MLJFlux.Mover;
        regularize::Bool=false, monte_carlo::Bool=false)
    logpx, λ₁, λ₂ = mdl(data, θ, move(randn(eltype(data), size(data))); regularize, monte_carlo)
    logpx
end

function ffjord_loss_4obj(mdl::DiffEqFlux.CNFLayer, data::Matrix{Float64}, move::MLJFlux.Mover;
        regularize::Bool=false, monte_carlo::Bool=false)
    function p_loss(θ::Vector)
        loss_4obj(ffjord_logpx(θ, mdl, data, move; regularize, monte_carlo))
    end
    p_loss
end

function ffjord_loss_1obj(mdl::DiffEqFlux.CNFLayer, data::Matrix{Float64}, move::MLJFlux.Mover;
        regularize::Bool=false, monte_carlo::Bool=false)
    function p_loss(θ::Vector)
        loss_1obj(ffjord_logpx(θ, mdl, data, move; regularize, monte_carlo))
    end
p_loss
end

const default_ffjord_loss = ffjord_loss_4obj
# const default_sensealg = ZygoteAdjoint()
const default_sensealg = InterpolatingAdjoint(
    autodiff=true,
    chunk_size=0,
    autojacvec=ZygoteVJP(),
)

MLJBase.@mlj_model mutable struct FFJORDModel <: PDFEstimator
    n_hidden_ratio::Int64 = 2::(_ > 0)
    tspan::Tuple{Float64, Float64} = default_tspan
    actv::Function = tanh
    basedist::Union{Distribution, Nothing} = nothing::(isnothing(_) || eltype(_) == model.dtype)

    sol_met::OrdinaryDiffEqAlgorithm = Tsit5()

    adtype::SciMLBase.AbstractADType = GalacticOptim.AutoZygote()
    sensealg::SciMLBase.AbstractSensitivityAlgorithm = default_sensealg

    optms::Vector{OptM} = default_optms

    regularize::Bool = false
    monte_carlo::Bool = false

    loss::Function = default_ffjord_loss

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
    lss_f = model.loss(ffjord_mdl, x, move; model.regularize, model.monte_carlo)
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
