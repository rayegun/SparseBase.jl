# TODO: MINIMIZE API.
# TODO: Document minimum API.
# TODO: Separate store specific, format specific and general fns.
"""
    isuniformvalued(A)::Bool

True if a type contains a single value across all stored indices.
"""
isuniformvalued(::Any) = false

"""
    hasfixedsparsity(::Type{A})::Bool

True if the sparsity pattern of the type `A` may be changed.
A `Diagonal` type, for instance, may not have its sparsity pattern changed.
"""
hasfixedsparsity(::Any) = false

# functionality should be clear, there are implicit values.
issparse(A) = issparse(typeof(A))
issparse(::Type{A}) where {A} = false
issparse(::Type{<:SparseStoreOrFormat}) = true

"""
    UniformValue{T}

Simple wrapper struct that denotes a uniform value for array input
(all stored values are equal to a single value such as `true`).
"""
struct UniformValue{T}
    val::T
end
Base.eltype(::UniformValue{T}) where {T} = T

"""
    deuniform!(A)::A

If an array supports an iso representation convert it to
the expected non-iso representation internally.
"""
deuniform!(A) = A

# TODO: iscoalesced

"""
    iscoalesced(::Type{A})::Bool = true

Whether each stored index is unique.
"""
iscoalesced(A) = true

"""
    isopaque(::Type{A})::Bool

True if internals may not be accessed directly. C owned types set this to true.
"""
isopaque(::Any) = false

"""
    nzombies(A)::Integer

The number of pending deletions of values in A.
"""
nzombies(A) = 0

"""
    npending(A)::Integer

The number of pending insertions into A.
"""
npending(A) = 0

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

# Metadata
#######################################
"""
    nstored(A)::Integer

Number of stored elements in `A`.
In the dense case this is `length(A)`
"""
nstored(A) = length(A) # default to the dense case.

"""
    getfill(A)

The value taken by all non-stored/implicit indices of A.
"""
function getfill end

"""
    setfill!(A, fill)::A
    setfill(A, fill)::B

Set the value taken by implicit indices of A to a new value.
`setfill` produces a shallow copy of A with the new fill value.
"""
function setfill end
function setfill! end

defaultfill(::Type{T}) where {T} = zero(T)
defaultfill(::Type{Missing}) = missing
defaultfill(::Type{Union{T,Missing}}) where {T} = missing
defaultfill(::Type{NoValues.NoValue}) = NoValues.novalue()
defaultfill(::Type{Union{T,NoValues.NoValue}}) where {T} = NoValues.novalue()
defaultfill(::T) where {T} = defaultfill(T)

# For everything below this:
# How to let users select implementation? If I have a HyperSparseMatrix defined in HyperSparseMatrices.jl
# how do I say: I want Finch to do this.
# Could also let us support CUDA/ROCm in the future? 
# This will also come up when we want to map canonical Finch kernels down to MKL/CUDA/SSGrB/etc.
# Can we come up with a default? Can we somehow make Finch override the default if it's available?
# Or maybe we set "compilers" to be the default.
# We might want it to depend on runtime properties of arrays as well which means barriers.

# How do I accept `f(i..., x)` for mapping purposes? API question. `TakeIndices` function wrapper?
# Or even better a trait on the function? 

# To coordinates:
#################
# required for extreme fallback construction between two types.

"""
    storedindices(A)

Iterables over the stored indices of `A`. May be a direct view into internals,
but is invalid to modify. May be lazy iterables.
"""
function storedindices end

"""
    storedvalues(A)

An iterable over the stored values of `A`. May be a direct view into internals, 
but this is not a requirement, and so shouldn't be used to modify A. 
Must be returned in the same order as `storedindices`.
"""
function storedvalues end

"""
    findstored(A)

An iterable over the stored indices and values of `A`. 
May be a direct view into internals, 
but this is not a requirement, and so shouldn't be used to modify A. 
"""
function findstored end

# Conversions:
##############

"""
    convertinnerstore(T::Type{<:AbstractSparseStore}, A)

Convert A into a similar format, with the internal sparse store converted to
    T.
"""
function convertinnerstore end

# LEVEL FORMATS:
################
# left as an exercise for the maintainer.
