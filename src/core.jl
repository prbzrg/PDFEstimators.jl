abstract type PDFEstimator <: MLJBase.Unsupervised end

function loss_4obj(logpx)
    lin_sca([
        -mean(logpx),
        -std(logpx),
        maximum(logpx),
        -minimum(logpx),
    ])
end

function loss_1obj(logpx)
    -mean(logpx)
end
