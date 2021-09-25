export ActualModel

MLJBase.@mlj_model mutable struct ActualModel <: PDFEstimator
    dist::Distribution = Normal()
end

function MLJBase.fit(model::ActualModel, verbosity, X)
    fitresult = nothing
    cache = nothing
    report = nothing
    fitresult, cache, report
end

function MLJBase.transform(model::ActualModel, fitresult, Xnew)
    ynew = size(Xnew, 2) == 1 ? pdf(model.dist, Xnew[!, 1]) : pdf(model.dist, collect(MLJBase.matrix(Xnew)'))
    ynew = reshape(ynew, size(ynew, 1), 1)
    ynew = DataFrame(ynew, :auto)
end
