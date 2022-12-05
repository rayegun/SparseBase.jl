abstract type AbstractCoordinateArray{Bi, Tv, Tfill, Ti, N} <: AbstractSparseArray{Bi, Tv, Tfill, Ti, N} end

issparse(::AbstractCoordinateArray) = true
Base.eltype(::AbstractCoordinateArray{<:Any, Tv, Tfill}) where {Tv, Tfill} = Union{Tv, Tfill}

mutable struct CoordinateArray{Bi, Tv, Tfill, Ti, I<:AbstractVector{Ti}, V<:AbstractArray{Tv}, NS, N} <:
    AbstractCoordinateArray{Bi, Tv, Tfill, Ti, N}
    bounds::NTuple{NS, Ti}
    indices::NTuple{NS, I}
    v::V # V could be an arbitrary array, but that can't be pushed into... Sort of necessary.
    # Might be relaxable in the future if we really do want coordinate -> dense levels.
    # In the meantime Tv can be vectors or SMatrix etc just fine.
    fill::Tfill
    issorted::Bool
    iscoalesced::Bool
end

function CoordinateArray{Bi}(
    bounds::NTuple{NS, Ti}, indices::NTuple{NS, I}, v::V, fill::Tfill = zero(eltype(v))
) where {Bi, Ti, Tv, Tfill, NS, I<:AbstractVector{Ti}, V<:AbstractArray{Tv}}
    CoordinateArray{Bi, Tv, Tfill, Ti, I, V, NS, NS + ndims(v) - 1}(bounds, indices, v, fill, false, false)
end
CoordinateArray(bounds, indices, v, fill=zero(eltype(v))) = CoordinateArray{1}(bounds, indices, v, fill)

setfill(A::CoordinateArray{Bi}, f) where {Bi} = 
    CoordinateArray{Bi}(A.bounds, A.indices, A.v, f)
function setfill!(A::CoordinateArray, f)
    A.fill = f
    return A
end

nstored(A::CoordinateArray) = length(A.v)
nsparsedims(A::CoordinateArray) = length(A.bounds)
ndensedims(A::CoordinateArray) = ndims(A.v) - 1

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
function unjumble!(A::CoordinateArray, combine = +; coalesce = true)
    linearindices = getindex.(Ref(LinearIndices(A.bounds)), CartesianIndex.(zip(A.indices...)))
    nunique = length(Set(linearindices)) # Slow but avoids the reallocation of the value array at the end.
    permutation = sortperm(linearindices; alg = Base.Sort.DEFAULT_STABLE)
    if (!coalesce || A.iscoalesced) && !A.issorted
        A.indices = getindex.(A.indices, Ref(permutation))
        A.v = copy(selectdim(A.v, ndims(A.v), permutation)) # sort the last dimension, which is the "hidden" dimension.
    elseif !A.issorted && !A.iscoalesced
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
    return A
end
# TODO: This is unecessarily expensive memory wise.
unjumble(A::CoordinateArray; coalesce = true) = unjumble!(deepcopy(A); coalesce)


# all three of these start out close, but they return a different thing inside.
function _indexhelper(A, i; combine = +)
    if !A.issorted
        unjumble!(A, combine) # we could avoid coalesce here... For now we'll coalesce.
    end
    range = 1:length(A.indices[1])
    for d ∈ eachindex(A.indices)
        range = searchsorted(view(A.indices[d], range), i[d]) .+ (range.start - 1)
        if length(range) == 0
            return (false, nothing)
        end
    end
    return (true, range)
end

function Base.getindex(A::CoordinateArray, i::Vararg{<:Integer}; combine = +)
    foundidx, range = _indexhelper(A, i; combine)
    return foundidx ? A.v[i[length(A.indices)+1:end]..., range.start] :
        getfill(A)
end

function Base.isstored(A::CoordinateArray, i::Vararg{<:Integer})
    foundidx, _ = _indexhelper(A, i)
    return foundidx
end
# cannot uncoalesce. Will use push! for that I think.
function Base.setindex!(A::CoordinateArray, x, i::Vararg{<:Integer})
    if getindex.(A.indices, lastindex(A.indices[1])) < i
        A.issorted = false
    end
    push!(A.v, x)
    push!.(A.indices, i)
    A.iscoalesced = false
    return x
end