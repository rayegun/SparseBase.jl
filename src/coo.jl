module CoordinateArrays

using ..SparseBase
using ..SparseBase: AbstractSparseArray, getbase, getoffset, getfill, StorageOrder, 
    RowMajor, ColMajor, comptime_storageorder, runtime_storageorder, NoOrder

export CoordinateArray, unjumble!, unjumble

abstract type AbstractCoordinateArray{Tv, Tfill, Bi, Ti, N} <: AbstractSparseArray{Tv, Tfill, Bi, Ti, N} end

SparseBase.issparse(::AbstractCoordinateArray) = true
SparseBase.comptime_storageorder(::AbstractCoordinateArray) = RuntimeOrder()

mutable struct CoordinateArray{Tv, Tfill, Bi, Ti<:Integer, I<:AbstractVector{Ti}, V<:AbstractArray{Tv}, NS, N} <:
    AbstractCoordinateArray{Tv, Tfill, Bi, Ti, N}
    bounds::NTuple{NS, Ti}
    indices::NTuple{NS, I}
    v::V # V could be an arbitrary array, but that can't be pushed into... Sort of necessary.
    # Might be relaxable in the future if we really do want coordinate -> dense levels.
    # In the meantime Tv can be vectors or SMatrix etc just fine.
    fill::Tfill
    issorted::Bool
    iscoalesced::Bool
    sortorder::StorageOrder
    # WE NEED ZOMBIES!!! Should that be recursive? Probably not.
end

function CoordinateArray{Bi}(
    indices::NTuple{NS, I}, v::V, bounds::NTuple{NS, Ti}; fill::Tfill = zero(eltype(v)), order = ColMajor()
) where {Bi, Ti, Tv, Tfill, NS, I<:AbstractVector{Ti}, V<:AbstractArray{Tv}}
    CoordinateArray{Tv, Tfill, Bi, Ti, I, V, NS, NS + ndims(v) - 1}(bounds, indices, v, fill, false, false, order)
end
CoordinateArray(indices, v, bounds; fill=zero(eltype(v)), order = ColMajor()) = CoordinateArray{1}(bounds, indices, v; fill, order)
CoordinateArray{T, Bi}(bounds::NTuple{N, Ti}; fill = zero(T), order = ColMajor()) where {T, Bi, Ti<:Integer, N} = 
    CoordinateArray{Bi}(ntuple(x->Int64[], N), T[], bounds; fill, order)
CoordinateArray{T}(bounds::NTuple{N, Ti}; fill = zero(T), order = ColMajor()) where {T, Ti<:Integer, N} = 
    CoordinateArray(ntuple(x->Int64[], N), T[], bounds; fill, order)

SparseBase.runtime_storageorder(A::CoordinateArray) = A.sortorder

SparseBase.getfill(A::CoordinateArray) = A.fill
SparseBase.setfill(A::CoordinateArray, f) = 
    CoordinateArray{getbase(A)}(A.bounds, A.indices, A.v, f)
function SparseBase.setfill!(A::CoordinateArray, f)
    A.fill = f
    return A
end

SparseBase.nstored(A::CoordinateArray) = length(A.v)
_nsparsedims(A::CoordinateArray) = length(A.bounds)
_ndensedims(A::CoordinateArray) = ndims(A.v) - 1 # trailing dense dimension is indexed by the leading sparse dim.

Base.size(A::CoordinateArray) = (A.bounds..., size(A.v)[1:end-1]...)

# Specialization slightly faster fwiw.
_combiner!(combine, z::AbstractVector, zi, y::AbstractVector, yi) = 
    length(z) == 1 ? z[1] = combine(z[1], y[1]) : z[zi] = combine(z[zi], y[yi])
function _combiner!(combine, z::AbstractArray, zi, y::AbstractArray, yi) 
    zdim = selectdim(z, ndims(z), zi)
    zdim .= combine.(zdim, selectdim(y, ndims(y), yi))
end
# Specialization slightly faster fwiw.
_updater!(z::AbstractVector, zi, y::AbstractVector, yi) = z[zi] = y[yi]
_updater!(z::AbstractArray, zi, y::AbstractArray, yi) = selectdim(z, ndims(z), zi) .= selectdim(y, ndims(y), yi)

# TODO: NEED A SEGMENT SUM BASED COALESCE ONLY, for when it's already sorted, but not coalesced.
function unjumble!(A::CoordinateArray, combine = +; coalesce = true, order = storageorder(A))
    o = getoffset(A)
    A.issorted && (!coalesce || A.iscoalesced) && (return A)
    if order === ColMajor()
        linear = LinearIndices(A.bounds)
    elseif order === RowMajor()
        linear = LinearIndices(A.bounds)'
    else
        throw(ArgumentError("order: $order ∉ {ColMajor(), RowMajor()}"))
    end
    if o == 0
        linearindices = getindex.(Ref(linear), CartesianIndex.(zip(A.indices...)))
    else # don't really trust CartesianIndex(0,0,0) + CartesianIndex(1,2,3) to be constpropped away.
        o = CartesianIndex(ntuple(x->o, _nsparsedims(A)))
        linearindices = getindex.(Ref(linear), (CartesianIndex.(zip(A.indices...))) .+ Ref(o))
    end
    nunique = length(Set(linearindices)) # Slow but avoids the reallocation of the value array at the end.
    permutation = sortperm(linearindices; alg = Base.Sort.DEFAULT_STABLE)
    if (!coalesce || A.iscoalesced) && !A.issorted
        A.indices = getindex.(A.indices, Ref(permutation))
        A.v = copy(selectdim(A.v, ndims(A.v), permutation)) # sort the last dimension, which is the "hidden" dimension.
        # I think it will always be the last dim, even if we want the BYROW COO.
    else
        v = similar(A.v, size(A.v)[1:end-1]..., nunique)
        indices = similar.(A.indices, nunique)
        previousindex = 0
        j = 0
        for i ∈ 1:size(A.v)[end]
            position = permutation[i]
            currentindex = linearindices[permutation[i]]
            if currentindex == previousindex
                _combiner!(combine, v, j, A.v, position)
            else
                j += 1
                for d ∈ eachindex(indices)
                    indices[d][j] = A.indices[d][position]
                end
                _updater!(v, j, A.v, position)
            end
            previousindex = currentindex
        end
        A.iscoalesced = true
        A.indices = indices
        A.v = v
    end
    A.issorted = true
    A.sortorder = order
    return A
end
# TODO: This is unecessarily expensive memory wise, we can unjumble *into* the copy.
# perhaps needs internal expert method with preallocated buffers
unjumble(A::CoordinateArray; coalesce = true, order = ColMajor()) = unjumble!(deepcopy(A); coalesce, order)


# all three of these start out close, but they return a different thing inside.
function _indexhelper(A, i, (sortedsearch, unsortedsearch) = (searchsorted, findall))
    o = getoffset(A)
    range = 1:length(A.indices[1])
    if A.issorted
        for d ∈ (A.sortorder === ColMajor() ? reverse(eachindex(A.indices)) : 
                    A.sortorder === RowMajor() ? eachindex(A.indices) : 
                    throw(ArgumentError("A.sortorder: $(A.sortorder) ∉ {ColMajor(), RowMajor()")))
            range = sortedsearch(view(A.indices[d], range), i[d] - o) .+ (min(range.start, range.stop) - 1)
            if length(range) == 0
                return (false, nothing)
            end
        end
    else
        for d ∈ eachindex(A.indices)
            v = view(A.indices[d], range)
            indices = unsortedsearch(x->x== i[d] - o, v)
            range = parentindices(v)[begin][indices]
            if length(range) == 0
                return (false, nothing)
            end
        end
    end
    return (true, range)
end

function Base.getindex(A::CoordinateArray, i::Vararg{<:Integer}; combine = +)
    nstored(A) == 0 && return getfill(A) # early stop, important for pending tuples.
    if combine === last
        foundidx, range = _indexhelper(A, i, (searchsortedlast, findlast))
    elseif combine === first
        foundidx, range = _indexhelper(A, i, (searchsortedfirst, findfirst))
    else
        foundidx, range = _indexhelper(A, i, (searchsorted, findall))
    end
    return foundidx ? reduce(combine, A.v[i[length(A.indices)+1:end]..., range]) :
        getfill(A)
end

function Base.isstored(A::CoordinateArray, i::Vararg{<:Integer})
    foundidx, _ = _indexhelper(A, i, (searchsortedfirst, findfirst))
    return foundidx
end

function Base.setindex!(A::CoordinateArray, x, i::Vararg{<:Integer})
    @boundscheck checkbounds(A, i...)
    foundidx, range = _indexhelper(A, i)
    if !foundidx # no existing, we can push!, no problem!
        push!(A, x, i)
    elseif length(range == 1)
        A.v[range] .= x 
    else # worst case, we need to delete the old indices.
        for d ∈ _nsparsedims(A)
            deleteat!(A.indices[d], range)
        end
        deleteat!(A.v, range)
        push!(A, x, i)
    end
    return x
end

function Base.push!(A::CoordinateArray, x, i::Vararg{<:Integer}; _stillcoalesced = false)
    if getindex.(A.indices, lastindex(A.indices[1])) < i
        A.issorted = false
    end
    push!(A.v, x)
    push!.(A.indices, i)
    A.iscoalesced = _stillcoalesced
    return x
end

end