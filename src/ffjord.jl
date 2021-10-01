export FFJORDModel, ffjord_loss_4obj, ffjord_loss_1obj

default_ffjord_optms = [
    OptM(
        method=Optim.KrylovTrustRegion(
            initial_radius=1.0,
            max_radius=128.0,
            eta=1/8,
            rho_lower=1/4,
            rho_upper=3/4,
            cg_tol=ATOL,
        ),
    ),
    OptM(
        method=AMSGrad(),
        maxiters=Int64(typemax(Int8)),
    ),
]

default_tspan = (0.0, 1.0)

function ffjord_logpx(θ::Vector, mdl::DiffEqFlux.CNFLayer, data::Matrix{Float64};
        regularize::Bool=false, monte_carlo::Bool=false)
    logpx, λ₁, λ₂ = mdl(data, θ; regularize, monte_carlo)
    logpx
end

function ffjord_loss_4obj(mdl::DiffEqFlux.CNFLayer, data::Matrix{Float64};
        regularize::Bool=false, monte_carlo::Bool=false)
    function p_loss(θ::Vector)
        loss_4obj(ffjord_logpx(θ, mdl, data; regularize, monte_carlo))
    end
    p_loss
end

function ffjord_loss_1obj(mdl::DiffEqFlux.CNFLayer, data::Matrix{Float64};
        regularize::Bool=false, monte_carlo::Bool=false)
    function p_loss(θ::Vector)
        loss_1obj(ffjord_logpx(θ, mdl, data; regularize, monte_carlo))
    end
p_loss
end

const default_ffjord_loss = ffjord_loss_4obj

MLJBase.@mlj_model mutable struct FFJORDModel <: PDFEstimator
    n_vars::Int64 = 1::(_ > 0)
    n_hidden::Int64 = 1::(_ > 0)
    tspan::Tuple{Float64, Float64} = default_tspan
    actv::Function = tanh
    basedist::Union{Distribution, Nothing} = nothing

    sol_met::OrdinaryDiffEqAlgorithm = Tsit5()

    adtype::SciMLBase.AbstractADType = GalacticOptim.AutoZygote()

    optms::Vector{OptM} = default_ffjord_optms

    regularize::Bool = false
    monte_carlo::Bool = false

    loss::Function = default_ffjord_loss
end

function MLJBase.fit(model::FFJORDModel, verbosity, X)
    x = collect(MLJBase.matrix(X)')

    nn = Chain(
        Dense(model.n_vars, model.n_hidden, model.actv),
        Dense(model.n_hidden, model.n_hidden, model.actv),
        Dense(model.n_hidden, model.n_vars, model.actv),
    ) |> f64
    ffjord_mdl = FFJORD(nn, model.tspan, model.sol_met; model.basedist)
    lss_f = model.loss(ffjord_mdl, x; model.regularize, model.monte_carlo)
    res = optimizeit(model, lss_f, ffjord_mdl.p)

    fitresult = res
    cache = nothing
    report = nothing
    fitresult, cache, report
end

function MLJBase.transform(model::FFJORDModel, fitresult, Xnew)
    xnew = collect(MLJBase.matrix(Xnew)')
    θ = fitresult.u

    nn = Chain(
        Dense(model.n_vars, model.n_hidden, model.actv),
        Dense(model.n_hidden, model.n_hidden, model.actv),
        Dense(model.n_hidden, model.n_vars, model.actv),
    ) |> f64
    ffjord_mdl = FFJORD(nn, model.tspan, model.sol_met; model.basedist, p=θ)
    logpx, λ₁, λ₂ = ffjord_mdl(xnew; model.regularize, model.monte_carlo)

    ynew = exp.(logpx)
    ynew = collect(ynew')
    ynew = DataFrame(ynew, :auto)
end

function MLJBase.fitted_params(model::FFJORDModel, fitresult)
    θ = fitresult.u

    (learned_params=θ,)
end
