@testset "FFJORD Model" begin
    @testset "1-var" begin
        @test !isnothing(one_var_test((n_vars, dist) -> FFJORDModel()))
    end
    @testset "2-var" begin
        @test !isnothing(multi_var_test((n_vars, dist) -> FFJORDModel()))
    end
    @testset "3-var" begin
        @test !isnothing(multi_var_test((n_vars, dist) -> FFJORDModel(), n_vars=3))
    end
end
