using
    PDFEstimators,
    DataFrames,
    Distributions,
    MLJBase,
    Test

@testset "Overall" begin
    include("core.jl")
    include("actual.jl")
    include("kde.jl")
    include("ash.jl")
    include("mvn.jl")
    include("ffjord.jl")
    include("utils.jl")
end
