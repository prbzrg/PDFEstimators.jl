export ActualModel

MLJBase.@mlj_model mutable struct ActualModel <: PDFEstimator
    dist::Distribution = Normal()
end

function MLJBase.fit(model::ActualModel, verbosity, X)
    n_vars = size(X, 2)

    fitresult = n_vars
    cache = nothing
    report = nothing
    fitresult, cache, report
end

function MLJBase.transform(model::ActualModel, fitresult, Xnew)
    xnew = collect(MLJBase.matrix(Xnew)')
    n_vars = fitresult

    ynew = n_vars == 1 ? pdf.(model.dist, vec(xnew)) : pdf(model.dist, xnew)
    ynew = reshape(ynew, size(ynew, 1), 1)
    ynew = DataFrame(ynew, :auto)
end

function MLJBase.fitted_params(model::ActualModel, fitresult)
    n_vars = fitresult

    (
        n_vars=n_vars,
    )
end