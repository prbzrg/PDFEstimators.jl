function one_var_test(mdl_gen::Function)
    n_vars = 1
    data_dist = Beta(2, 4)
    train_data = let n=Int64(typemax(Int8))
        data = rand(data_dist, 1, n)
        data = collect(data')
        data = DataFrame(data, :auto)
    end
    train_pred_pdf = let data=train_data, mdl=mdl_gen(1, data_dist)
        mach = machine(mdl, data)
        fit!(mach)
        pred_pdf = MLJBase.transform(mach, data)
    end
    train_pred_pdf
end

function multi_var_test(mdl_gen::Function; n_vars::Int64=2)
    data_dist = Dirichlet(n_vars, 4)
    train_data = let n=Int64(typemax(Int8))
        data = rand(data_dist, n)
        data = collect(data')
        data = DataFrame(data, :auto)
    end
    train_pred_pdf = let data=train_data, mdl=mdl_gen(n_vars, data_dist)
        mach = machine(mdl, data)
        fit!(mach)
        pred_pdf = MLJBase.transform(mach, data)
    end
    train_pred_pdf
end
