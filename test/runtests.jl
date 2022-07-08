using
    PDFEstimators,
    DataFrames,
    Distributions,
    MLJBase,
    Test

@testset "Overall" begin
    include("core.jl")
    include("actual.jl")
    include("fitted.jl")
    include("kde.jl")
    include("ash.jl")
    include("ffjord.jl")
    include("utils.jl")
end
