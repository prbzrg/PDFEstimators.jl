export OptM

abstract type PDFEstimator <: MLJBase.Unsupervised end

const MAXITERS = typemax(Int64)
const ATOL = eps(Float64)
const RTOL = -Inf64

@with_kw struct OptM
    method::Union{Optim.AbstractOptimizer, Flux.Optimise.AbstractOptimiser} = SimulatedAnnealing()
    maxiters::Union{Int64, Nothing} = MAXITERS
    atol = ATOL
    rtol = RTOL
    allow_f_increases = true
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

default_optms = [
    # Zeroth order
    # OptM(
    #     method=ParticleSwarm(),
    # ),
    # OptM(
    #     method=SimulatedAnnealing(),
    # ),
    # OptM(
    #     method=NelderMead(),
    # ),
    # Flux
    # OptM(
    #     method=AMSGrad(),
    #     maxiters=Int64(typemax(Int8)),
    # ),
    # First order
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
    # OptM(
    #     method=ConjugateGradient(
    #         alphaguess=LineSearches.InitialHagerZhang(),
    #         linesearch=LineSearches.HagerZhang(),
    #         eta=1/2,
    #         manifold=Flat(),
    #     ),
    # ),
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
    # Second order
    OptM(
        method=Optim.KrylovTrustRegion(
            initial_radius=1.0,
            max_radius=128.0,
            eta=1/8,
            rho_lower=1/4,
            rho_upper=3/4,
            cg_tol=ATOL,
        ),
    ),
    # OptM(
    #     method=NewtonTrustRegion(),
    # ),
    # OptM(
    #     method=Newton(
    #         alphaguess=LineSearches.InitialHagerZhang(),
    #         linesearch=LineSearches.HagerZhang(),
    #     ),
    # ),
]
