export mach_dis

function lin_sca(xs::Vector; weights_base::Float64=2.0)
    sum(enumerate(reverse(xs))) do (i, x)
        (weights_base ^ (i - 1)) * x
    end / ((weights_base ^ size(xs, 1)) - 1)
end

function mach_dis(data, pred_mach, base_mach)
    base_pdf = MLJBase.transform(base_mach, data)[!, 1]
    pred_pdf = MLJBase.transform(pred_mach, data)[!, 1]
    (
        tv_dis=totalvariation(pred_pdf, base_pdf) / size(data, 1),
        cos_dis=cosine_dist(pred_pdf, base_pdf),
        mad_dis=meanad(pred_pdf, base_pdf),
    )
end
