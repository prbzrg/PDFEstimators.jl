using PDFEstimators
using Documenter

DocMeta.setdocmeta!(PDFEstimators, :DocTestSetup, :(using PDFEstimators); recursive=true)

makedocs(;
    modules=[PDFEstimators],
    authors="Hossein Pourbozorg <prbzrg@gmail.com> and contributors",
    repo="https://github.com/prbzrg/PDFEstimators.jl/blob/{commit}{path}#{line}",
    sitename="PDFEstimators.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://prbzrg.github.io/PDFEstimators.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/prbzrg/PDFEstimators.jl",
    devbranch="main",
)
