@testset "FFJORD Model" begin
    @testset "1-var" begin
        @test !isnothing(one_var_test((n_vars, dist) -> FFJORDModel(n_vars=n_vars, n_hidden=n_vars*2)))
    end
    @testset "2-var" begin
        @test !isnothing(multi_var_test((n_vars, dist) -> FFJORDModel(n_vars=n_vars, n_hidden=n_vars*2)))
    end
    @testset "3-var" begin
        @test !isnothing(multi_var_test((n_vars, dist) -> FFJORDModel(n_vars=n_vars, n_hidden=n_vars*2), n_vars=3))
    end
end
