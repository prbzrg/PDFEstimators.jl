export KDEModel

MLJBase.@mlj_model mutable struct KDEModel <: PDFEstimator end

function MLJBase.fit(model::KDEModel, verbosity, X)
    n_vars = size(X, 2)
    kde_model = n_vars == 1 ? kde(eachcol(X)...) : kde(MLJBase.matrix(X))

    fitresult = (kde_model, n_vars)
    cache = nothing
    report = nothing
    fitresult, cache, report
end

function MLJBase.transform(model::KDEModel, fitresult, Xnew)
    kde_model, n_vars = fitresult

    ynew = n_vars == 1 ? pdf(kde_model, eachcol(Xnew)...) : Diagonal(pdf(kde_model, eachcol(Xnew)...)).diag
    ynew = reshape(ynew, size(ynew, 1), 1)
    ynew = DataFrame(ynew, :auto)
end

function MLJBase.fitted_params(model::KDEModel, fitresult)
    kde_model, n_vars = fitresult

    (
        kde_model=kde_model,
        n_vars=n_vars,
    )
end
