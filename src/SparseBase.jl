module SparseBase

include("novalue.jl")

import StorageOrders
import SparseArrays
# Trait(ish):

# storage order stuff:
const comptime_storageorder = StorageOrders.comptime_storageorder
const runtime_storageorder = StorageOrders.runtime_storageorder
const storageorder = StorageOrders.storageorder
const RowMajor = StorageOrders.RowMajor
const ColMajor = StorageOrders.ColMajor
const RuntimeOrder = StorageOrders.RuntimeOrder
const NoOrder = StorageOrders.NoOrder

# functionality should be clear, there are implicit values.
# operates on types, not on values.
function issparse end

```
    isisovalued(A)::Bool

True if a type contains a single value across all explicit indices.
May be compile time, for instance UniformScaling.
```
function isisovalued end

```
    haszerobasedinternals(::Type{A})::Bool

True if a type has zero based internal vectors.

This *does not* indicate that `A` is indexed in a zero-based manner,
just that the internal representation is zero based. This is useful if C libs
implement much of the functionality of A.
```
function haszerobasedinternals end

```
    hasfixedsparsity(::Type{A})::Bool

True if the sparsity pattern of the type `A` may be changed.
A `Diagonal` type, for instance, may not have its sparsity pattern changed.
```
function hasfixedsparsity end

```
    isopaque(::Type{A})::Bool

True if internals may be accessed directly. C owned types often set this to true.
```
function isopaque end
# additionally a GraphBLAS.jl implementation wants this to be true.

# FUNCTION TRAITS:
##################

# do we want these here? I think generally speaking we want to determine:
#=
1. associativity
2. distributivity
3. idempotency
4. terminal/annihilator
5. identity
6. a few more

for various functions. All the sparse functions take advantage of some subset of this info.

A big problem is I also think we want these to depend on the element type in some cases.
So we might want it to be `isassociative(f, T...)`.
=#

# High level functions:
#######################################
# metadata
const nnz = SparseArrays.nnz

# @Willow I very strongly want `novalue` to be an acceptable fill value,
# or something else similar.
# Quite useful for graphs. That would mean that `A` might have two eltypes:
# SparseMatrix{T, F}, where eltype(::SparseMatrix{T, F}) = Union{T, F}
# Thoughts?
```
    getfill(A)

The value taken by all non-stored/implicit indices of A.
```
function getfill end

```
    setfill!(A)::A
    setfill(A)::B

Set the value taken by implicit indices of A to a new value.
`setfill` produces a shallow copy of A with the new fill value.
```
function setfill end
function setfill! end

# eltype for most matrices, could be Union{eltype, Missing/NoValue} for graphs
filltype(A::AbstractArray) = eltype(A)

# For everything below this:
# How to let users select implementation? If I have a HyperSparseMatrix defined in HyperSparseMatrices.jl
# how do I say: I want Finch to do this. Should we have a last argument executor? 
# Could also let us support CUDA/ROCm in the future? 
# This will also come up when we want to map canonical Finch kernels down to MKL/CUDA/SSGrB/etc.
# Importantly, can we come up with a default? Can we somehow make Finch override the default if it's available?
# Or maybe we set "compilers" to be the default. This is an incredibly hard design problem.
# Since we might want it to depend on runtime properties of arrays as well.
# mapping
##########
```
    mapstored[!](f, [C], A)
    mapstored(f, A ∪ B)
    mapstored(f, (A ∩ B) ∪ C)

Map a function over the stored values of A(, B, C, ...).

Unions and intersections of A, B, C... operate elementwise
over the unions and intersections of the patterns of those matrices.

TBD: 

1. `eadd` in GraphBLAS doesn't work here, we need some notion of "passthrough".
`missing` doesn't work since missing + 2.0 = missing. GraphBLAS wants some ability
to do `NONSTORED + 2.0 = 2.0` for the `eadd` operation.

2. This could potentially lead to wrapper hell. I hope it doesn't, I think we would have
`UnionPattern`, `IntersectionPattern` (maybe `ComplementPattern`?). I don't think these
wrappers are quite as bad as the ones in LinAlg, but I could be wrong.

Current implementations (SSGrB) only implement `mapstored[!](f, [C], A)
```
function mapstored! end
function mapstored end

# How do I accept `f(i..., x)` for mapping purposes? API question. `TakeIndices` function wrapper?
# Or even better a trait on the function? 

```
    fkeep!(f, [C], A)

Keep values of `A` where `f` is true. `f(i..., x)::Bool`
```
function fkeep! end
function fkeep end

# To coordinates:
#################
# required for extreme fallback construciton between two types.

```
    storedindices(A)

An iterable over the stored indices of `A`. May be a direct view into internals,
but is invalid to modify. May be a lazy iterator.
```
function storedindices end

```
    storedvalues(A)

An iterable over the stored values of `A`. May be a direct view into internals, 
but this is not a requirement, and so shouldn't be used to modify A.
```
function storedvalues end

# LEVEL FORMATS:
################

# What should go in here? We have lots of options, and orthogonal design choices:
#=
We want to say "what are the level formats for some array". This requires names,
without implementations. For that purpose we want something like:
=#
abstract type LevelFormat end
struct DenseLevel <: LevelFormat end
struct BitMapLevel <: LevelFormat end
struct SparseListLevel <: LevelFormat end # is this compressed level or singleton level? Don't remember off top of head.
struct UnknownLevel <: LevelFormat end # for C resident codes? We could maybe ensure safe(ish) fallbacks here?
struct SingletonLevel <: LevelFormat end
# what do you think of a FunctionLevel @Willow?
# it's in a sense opaque, but could implement some otherwise difficult levels
# struct FunctionLevel <: LevelFormat end
#etc

# but what are the names here?

# ITERATION FUNCTIONALITY:
##########################

# I think we should do a few things. The first is define "backup" iteration functionality.
# This can be used to at least iterate things in a sparse manner. 
# With this capability in hand we could then have `@finch` macro that takes this for loop structure
# and compiles it to a much faster version.
# Design TBD.

# SOLVER FUNCTIONALITY:
#######################
# not yet sure on this one, starting to split solvers out right now.
# v0.2 I'll know more about what we need here to make this easier to impl


end
