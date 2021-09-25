export KDEModel

MLJBase.@mlj_model mutable struct KDEModel <: PDFEstimator end

function MLJBase.fit(model::KDEModel, verbosity, X)
    kde_model = kde(size(X, 2) == 1 ? X[!, 1] : MLJBase.matrix(X))

    fitresult = kde_model
    cache = nothing
    report = nothing
    fitresult, cache, report
end

function MLJBase.transform(model::KDEModel, fitresult, Xnew)
    kde_model = fitresult

    ynew = size(Xnew, 2) == 1 ? pdf(kde_model, Xnew[!, 1]) : Diagonal(pdf(kde_model, Xnew[!, 1], Xnew[!, 2])).diag
    ynew = reshape(ynew, size(ynew, 1), 1)
    ynew = DataFrame(ynew, :auto)
end
