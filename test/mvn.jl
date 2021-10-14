@testset "Mvn Model" begin
    @testset "1-var" begin
        @test @timev !isnothing(one_var_test((n_vars, dist) -> MvnModel()))
    end
    @testset "2-var" begin
        @test @timev !isnothing(multi_var_test((n_vars, dist) -> MvnModel()))
    end
    @testset "3-var" begin
        @test @timev !isnothing(multi_var_test((n_vars, dist) -> MvnModel(), n_vars=3))
    end
end
