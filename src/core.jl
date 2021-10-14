export OptM

abstract type PDFEstimator <: MLJBase.Unsupervised end

@with_kw struct OptM
    method::Union{Optim.AbstractOptimizer, Flux.Optimise.AbstractOptimiser} = NelderMead()
    maxiters::Int64 = typemax(Int64)
    atol::Float64 = -Inf64
end

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

default_optms = OptM[
    # - Flux
    # OptM(
    #     method=AMSGrad(),
    # ),
    # - Optim
    # -- Zeroth order
    # OptM(
    #     method=ParticleSwarm(),
    # ),
    # OptM(
    #     method=SimulatedAnnealing(),
    # ),
    # OptM(
    #     method=NelderMead(),
    # ),
    # -- First order
    # OptM(
    #     method=AcceleratedGradientDescent(),
    # ),
    # OptM(
    #     method=MomentumGradientDescent(),
    # ),
    # OptM(
    #     method=OACCEL(),
    # ),
    # OptM(
    #     method=NGMRES(),
    # ),
    OptM(
        method=ConjugateGradient(
            alphaguess=LineSearches.InitialHagerZhang(),
            linesearch=LineSearches.HagerZhang(),
            eta=1/2,
            manifold=Flat(),
        ),
        atol=Float64(eps(Float16)),
    ),
    # OptM(
    #     method=GradientDescent(
    #         alphaguess=LineSearches.InitialHagerZhang(),
    #         linesearch=LineSearches.HagerZhang(),
    #         manifold=Flat(),
    #     ),
    # ),
    # OptM(
    #     method=LBFGS(
    #         alphaguess=LineSearches.InitialHagerZhang(),
    #         linesearch=LineSearches.HagerZhang(),
    #         manifold=Flat(),
    #     ),
    # ),
    # OptM(
    #     method=BFGS(
    #         alphaguess=LineSearches.InitialHagerZhang(),
    #         linesearch=LineSearches.HagerZhang(),
    #         manifold=Flat(),
    #     ),
    # ),
    # -- Second order
    # OptM(
    #     method=Optim.KrylovTrustRegion(
    #         initial_radius=1.0,
    #         max_radius=128.0,
    #         eta=1/8,
    #         rho_lower=1/4,
    #         rho_upper=3/4,
    #         cg_tol=0.0,
    #     ),
    # ),
    # OptM(
    #     method=NewtonTrustRegion(
    #         initial_delta=1.0,
    #         delta_hat=128.0,
    #         eta=1/8,
    #         rho_lower=1/4,
    #         rho_upper=3/4,
    #         use_fg=true,
    #     ),
    # ),
    # OptM(
    #     method=Newton(
    #         alphaguess=LineSearches.InitialHagerZhang(),
    #         linesearch=LineSearches.HagerZhang(),
    #     ),
    # ),
]
