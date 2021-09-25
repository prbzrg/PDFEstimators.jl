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
    include("ffjord.jl")
    include("mvn.jl")
    include("utils.jl")
end
