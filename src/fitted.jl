export FittedModel

MLJBase.@mlj_model mutable struct FittedModel <: PDFEstimator
    dist_type::UnionAll = MvNormal
end

function MLJBase.fit(model::FittedModel, verbosity, X)
    x = collect(MLJBase.matrix(X)')
    n_vars = size(x, 1)

    learned_dist = fit_mle(model.dist_type, x)

    fitresult = (learned_dist, n_vars)
    cache = nothing
    report = nothing
    fitresult, cache, report
end

function MLJBase.transform(model::FittedModel, fitresult, Xnew)
    xnew = collect(MLJBase.matrix(Xnew)')
    learned_dist, n_vars = fitresult

    ynew = pdf(learned_dist, xnew)
    ynew = reshape(ynew, size(ynew, 1), 1)
    ynew = DataFrame(ynew, :auto)
end

function MLJBase.fitted_params(model::FittedModel, fitresult)
    learned_dist, n_vars = fitresult

    (
        learned_dist=learned_dist,
        n_vars=n_vars,
    )
end
