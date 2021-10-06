export MvnModel, mvn_loss_4obj, mvn_loss_1obj

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

    optms::Vector{OptM} = default_optms

    loss::Function = default_mvn_loss
end

function MLJBase.fit(model::MvnModel, verbosity, X)
    x = collect(MLJBase.matrix(X)')

    n_vars = size(x, 1)
    p = rand(n_vars * 2)
    lss_f = model.loss(x)
    res = optimizeit(model, lss_f, p)

    μ = res.u[1:n_vars]
    Σ = Diagonal(res.u[n_vars+1:end] .^ 2)
    learned_dist = MvNormal(μ, Σ)

    fitresult = (learned_dist, res, n_vars)
    cache = nothing
    report = nothing
    fitresult, cache, report
end

function MLJBase.transform(model::MvnModel, fitresult, Xnew)
    xnew = collect(MLJBase.matrix(Xnew)')
    learned_dist, res, n_vars = fitresult

    ynew = pdf(learned_dist, xnew)
    ynew = reshape(ynew, size(ynew, 1), 1)
    ynew = DataFrame(ynew, :auto)
end

function MLJBase.fitted_params(model::MvnModel, fitresult)
    learned_dist, res, n_vars = fitresult

    (
        learned_params=res.u,
        learned_dist=learned_dist,
        res=res,
        n_vars=n_vars,
    )
end
