using AbstractSparse
using Documenter

DocMeta.setdocmeta!(AbstractSparse, :DocTestSetup, :(using AbstractSparse); recursive=true)

makedocs(;
    modules=[AbstractSparse],
    authors="Will Kimmerer <kimmerer@mit.edu> and contributors",
    repo="https://github.com/Wimmerer/AbstractSparse.jl/blob/{commit}{path}#{line}",
    sitename="AbstractSparse.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Wimmerer.github.io/AbstractSparse.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Wimmerer/AbstractSparse.jl",
    devbranch="main",
)
