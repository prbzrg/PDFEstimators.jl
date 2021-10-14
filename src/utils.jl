export mach_dis

function optimizeit(model::Union{FFJORDModel, MvnModel}, lss_f::Function, p::Vector{Float64})
    mdl_name = typeof(model).name.name
    res = (u=p,)
    res_l = lss_f(res.u)
    @info "Initial $mdl_name loss is $res_l"
    for optm in model.optms
        m_name = typeof(optm.method).name.name
        t0 = time()
        if optm.method isa Optim.AbstractOptimizer
            res = DiffEqFlux.sciml_train(lss_f, res.u, optm.method, model.adtype;
                progress=false,
                maxiters=optm.maxiters,
                cb=cb,
                x_tol=nothing, f_tol=nothing, g_tol=nothing,
                x_abstol=optm.atol, x_reltol=-Inf64,
                f_abstol=optm.atol, f_reltol=-Inf64,
                g_abstol=optm.atol, g_reltol=-Inf64,
                outer_x_tol=nothing, outer_f_tol=nothing, outer_g_tol=nothing,
                outer_x_abstol=optm.atol, outer_x_reltol=-Inf64,
                outer_f_abstol=optm.atol, outer_f_reltol=-Inf64,
                outer_g_abstol=optm.atol, outer_g_reltol=-Inf64,
                f_calls_limit=0, g_calls_limit=0, h_calls_limit=0,
                allow_f_increases=true, allow_outer_f_increases=true,
                successive_f_tol=typemax(Int64),
                # iterations=optm.maxiters,
                outer_iterations=optm.maxiters,
                store_trace=true,
                trace_simplex=true,
                show_trace=false,
                # extended_trace=true,
                show_every=1,
                # callback=cb,
                time_limit=Inf64,
            )
        else
            res = DiffEqFlux.sciml_train(lss_f, res.u, optm.method, model.adtype;
                progress=false,
                maxiters=optm.maxiters,
                cb=cb,
            )
        end
        t1 = time()
        dt = t1 - t0
        res_l = lss_f(res.u)
        @info "Using $m_name optimization method, $mdl_name loss reached to $res_l, in $dt seconds"
    end
    @info "Final $mdl_name loss is $res_l"
    res
end

function lin_sca(xs::Vector; weights_base::Float64=2.0)
    sum(enumerate(reverse(xs))) do (i, x)
        (weights_base ^ (i - 1)) * x
    end / ((weights_base ^ size(xs, 1)) - 1)
end

function cb(p, l)
    @info "loss reached to $l"
    false
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
