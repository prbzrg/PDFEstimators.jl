@testset "KDE Model" begin
    @testset "1-var" begin
        @test @timev !isnothing(one_var_test((n_vars, dist) -> KDEModel()))
    end
    @testset "2-var" begin
        @test @timev !isnothing(multi_var_test((n_vars, dist) -> KDEModel()))
    end
    @testset "3-var" begin
        @test_broken @timev !isnothing(multi_var_test((n_vars, dist) -> KDEModel(), n_vars=3))
    end
end
