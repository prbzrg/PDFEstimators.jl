module PDFEstimators

    using
        AverageShiftedHistograms,
        CUDA,
        ComputationalResources,
        DataFrames,
        DiffEqFlux,
        DifferentialEquations,
        Distances,
        Distributions,
        Flux,
        ForwardDiff,
        GalacticOptim,
        KernelDensity,
        LineSearches,
        MLJBase,
        MLJFlux,
        MLJModelInterface,
        ModelingToolkit,
        Optim,
        Parameters,
        ReverseDiff,
        SciMLBase,
        ScientificTypes,
        Zygote,
        LinearAlgebra,
        Statistics

    include("core.jl")
    include("actual.jl")
    include("kde.jl")
    include("ash.jl")
    include("mvn.jl")
    include("ffjord.jl")
    include("utils.jl")

    MLJBase.metadata_pkg.(
        [ActualModel, KDEModel, FFJORDModel, MvnModel],
        package_name="PDFEstimators",
        package_uuid="4d826980-ec0a-4f65-a989-f1052e211ebd",
        package_url="unknown",
        is_pure_julia=true,
        package_license="MIT",
        is_wrapper=false,
    )

    MLJBase.metadata_model.(
        [ActualModel, KDEModel, FFJORDModel, MvnModel],
        input_scitype=Table{AbstractVector{ScientificTypes.Continuous}},
        target_scitype=Table{AbstractVector{ScientificTypes.Continuous}},
        output_scitype=Table{AbstractVector{ScientificTypes.Continuous}},
        supports_weights=false,
        docstring=["ActualModel", "KDEModel", "FFJORDModel", "MvnModel"],
        load_path=["PDFEstimators.ActualModel", "PDFEstimators.KDEModel", "PDFEstimators.FFJORDModel", "PDFEstimators.MvnModel"],
    )
end
