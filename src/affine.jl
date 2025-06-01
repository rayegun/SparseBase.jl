using TermInterface

struct AffineIndexOrSymbol{X}
    position::Int32
end
struct AffineBinaryOp{F, X, Y}
    x::X
    y::Y
end
const AffineAdd{X, Y} = AffineBinaryOp{+, X, Y}
const AffineSub{X, Y} = AffineBinaryOp{-, X, Y}
const AffineMul{X, Y} = AffineBinaryOp{*, X, Y}
const AffineCeilDiv{X, Y} = AffineBinaryOp{cld, X, Y}
const AffineFloorDiv{X, Y} = AffineBinaryOp{fld, X, Y}
const AffineMod{X, Y} = AffineBinaryOp{%, X, Y}

TermInterface.istree(::Type{<:AffineBinaryOp}) = true
TermInterface.operation(::Type{<:AffineBinaryOp{F}}) where F = F
TermInterface.arguments(::Type{<:AffineBinaryOp{F, X, Y}}) where {F, X, Y} = (X, Y)
TermInterface.exprhead(::Type{<:AffineBinaryOp}) = :call

struct AffineMap{N, A, T}
    functions::T
end

function (AffineMap{N, A, F})()
