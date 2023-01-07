@testset "ASH Model" begin
    @testset "1-var" begin
        @test @timev !isnothing(one_var_test((n_vars, dist) -> ASHModel()))
    end
    @testset "2-var" begin
        @test @timev !isnothing(multi_var_test((n_vars, dist) -> ASHModel()))
    end
    @testset "3-var" begin
        @test_broken @timev !isnothing(multi_var_test((n_vars, dist) -> ASHModel(), n_vars=3))
    end
end
