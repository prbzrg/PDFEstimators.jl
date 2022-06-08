export OptM

abstract type PDFEstimator <: MLJBase.Unsupervised end

@with_kw struct OptM
    method::Any = NelderMead()
    maxiters::Int64 = Int64(typemax(Int8))
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

long_optms = OptM[
    # - Flux
    OptM(
        method=Optimisers.AMSGrad(),
    ),
    # - Optim
    # -- Zeroth order
    # OptM(
    #     method=ParticleSwarm(),
    # ),
    OptM(
        method=SimulatedAnnealing(),
    ),
    OptM(
        method=NelderMead(),
    ),
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
    OptM(
        method=BFGS(
            alphaguess=LineSearches.InitialHagerZhang(),
            linesearch=LineSearches.HagerZhang(),
            manifold=Flat(),
        ),
    ),
    # -- Second order
    OptM(
        method=Optim.KrylovTrustRegion(
            initial_radius=1.0,
            max_radius=128.0,
            eta=1/8,
            rho_lower=1/4,
            rho_upper=3/4,
            cg_tol=Float64(eps(Float32)),
        ),
    ),
    OptM(
        method=NewtonTrustRegion(
            initial_delta=1.0,
            delta_hat=128.0,
            eta=1/8,
            rho_lower=1/4,
            rho_upper=3/4,
            use_fg=true,
        ),
    ),
    OptM(
        method=Newton(
            alphaguess=LineSearches.InitialHagerZhang(),
            linesearch=LineSearches.HagerZhang(),
        ),
    ),
]
short_optms = OptM[
    OptM(
        method=ConjugateGradient(
            alphaguess=LineSearches.InitialHagerZhang(),
            linesearch=LineSearches.HagerZhang(),
            eta=1/2,
            manifold=Flat(),
        ),
    ),
]
