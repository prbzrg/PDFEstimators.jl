export MvnModel, mvn_loss_4obj, mvn_loss_1obj

default_mvn_optms = [
    OptM(
        method=Newton(
            alphaguess=LineSearches.InitialHagerZhang(),
            linesearch=LineSearches.HagerZhang(),
        ),
    ),
]

function mvn_logpx(θ::Vector, data::Matrix{Float64})
    n_vars = size(θ, 1) ÷ 2
    μ = θ[1:n_vars]
    Σ = Diagonal(θ[n_vars+1:end] .^ 2)
    dist = MvNormal(μ, Σ)
    logpx = logpdf(dist, data)
    logpx
end

function mvn_loss_4obj(data::Matrix{Float64})
    function p_loss(θ::Vector)
        loss_4obj(mvn_logpx(θ, data))
    end
    p_loss
end

function mvn_loss_1obj(data::Matrix{Float64})
    function p_loss(θ::Vector)
        loss_1obj(mvn_logpx(θ, data))
    end
    p_loss
end

const default_mvn_loss = mvn_loss_4obj

MLJBase.@mlj_model mutable struct MvnModel <: PDFEstimator
    adtype::SciMLBase.AbstractADType = GalacticOptim.AutoZygote()

    optms::Vector{OptM} = default_mvn_optms

    loss::Function = default_mvn_loss
end

function MLJBase.fit(model::MvnModel, verbosity, X)
    x = collect(MLJBase.matrix(X)')

    n_vars = size(x, 1)
    p = rand(n_vars * 2)
    lss_f = model.loss(x)
    res = optimizeit(model, lss_f, p)

    fitresult = res
    cache = nothing
    report = nothing
    fitresult, cache, report
end

function MLJBase.transform(model::MvnModel, fitresult, Xnew)
    xnew = collect(MLJBase.matrix(Xnew)')
    θ = fitresult.u

    n_vars = size(θ, 1) ÷ 2
    μ = θ[1:n_vars]
    Σ = Diagonal(θ[n_vars+1:end] .^ 2)
    dist = MvNormal(μ, Σ)

    ynew = pdf(dist, xnew)
    ynew = reshape(ynew, size(ynew, 1), 1)
    ynew = DataFrame(ynew, :auto)
end

function MLJBase.fitted_params(model::MvnModel, fitresult)
    θ = fitresult.u

    n_vars = size(θ, 1) ÷ 2
    μ = θ[1:n_vars]
    Σ = Diagonal(θ[n_vars+1:end] .^ 2)
    dist = MvNormal(μ, Σ)

    (learned_params=θ, learned_dist=dist)
end
