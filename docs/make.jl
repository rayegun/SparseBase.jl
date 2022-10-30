using SparseBase
using Documenter

DocMeta.setdocmeta!(SparseBase, :DocTestSetup, :(using SparseBase); recursive=true)

makedocs(;
    modules=[SparseBase],
    authors="Will Kimmerer <kimmerer@mit.edu> and contributors",
    repo="https://github.com/Wimmerer/SparseBase.jl/blob/{commit}{path}#{line}",
    sitename="SparseBase.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Wimmerer.github.io/SparseBase.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Wimmerer/SparseBase.jl",
    devbranch="main",
)
