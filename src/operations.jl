# mapping
##########
"""
    mapstored[!](f, [E], [C], A)
    mapstored(f, [E], A ∪ B)
    mapstored(f, [E], (A ∩ B) ∪ C)

Map a function over the stored values of A(, B, C, ...).

Unions and intersections of A, B, C... operate elementwise
over the unions and intersections of the patterns of those matrices.

# Issues: 

  1. This could potentially lead to wrapper hell. I hope it doesn't, I think we would have
`UnionPattern`, `IntersectionPattern` (maybe `ComplementPattern`?). I don't think these
wrappers are quite as bad as the ones in LinAlg, but I could be wrong.

Current implementations (SSGrB) only implement `mapstored[!](f, [C], A)`,
`mapstored[!](f, [C], A ∪ B)` and `mapstored[!](f, [C], A ∩ B)`
"""
function mapstored! end
function mapstored end

struct UnionPattern{A}
    args::A
end

struct IntersectPattern{A}
    args::A
end

# struct NotPattern{A}
#     args::A
# end

∪(x...) = UnionPattern(x)
∩(x...) = IntersectPattern(x)

# TODO: subsume into Willow's algebra pkg.
function defaultpattern end

# reductions
############
function reducestored! end
function reducestored end

# filter[!]
##############
abstract type AbstractFunctionArguments end
struct IndexFunction{F} <: AbstractFunctionArguments
    f::F
end
struct ValueFunction{F} <: AbstractFunctionArguments
    f::F
end
struct IndexAndValueFunction{F} <: AbstractFunctionArguments
    f::F
end
struct ValueAndIndexFunction{F} <: AbstractFunctionArguments
    f::F
end
defaultargwrap(f) = ValueFunction(f)
# TODO: Fill out defaults for Base, LinAlg, stats, etc.
Base.parent(F::AbstractFunctionArguments) = F.f

function dropfill end
function dropfill! end