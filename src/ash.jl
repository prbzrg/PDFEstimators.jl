export ASHModel

MLJBase.@mlj_model mutable struct ASHModel <: PDFEstimator end

function MLJBase.fit(model::ASHModel, verbosity, X)
    n_vars = size(X, 2)
    ash_model = ash(eachcol(X)...)

    fitresult = (ash_model, n_vars)
    cache = nothing
    report = nothing
    fitresult, cache, report
end

function MLJBase.transform(model::ASHModel, fitresult, Xnew)
    ash_model, n_vars = fitresult

    ynew = broadcast(eachrow(MLJBase.matrix(Xnew))) do xr
        AverageShiftedHistograms.pdf(ash_model, xr...)
    end
    ynew = reshape(ynew, size(ynew, 1), 1)
    ynew = DataFrame(ynew, :auto)
end

function MLJBase.fitted_params(model::ASHModel, fitresult)
    ash_model, n_vars = fitresult

    (
        ash_model=ash_model,
        n_vars=n_vars,
    )
end
