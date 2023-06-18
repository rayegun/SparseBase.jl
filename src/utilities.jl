decrement(I) = I .- 1
increment(I) = I .+ 1

decrement!(I) = I .-= 1
decrement!(I::Integer) = decrement(I)

increment!(I) = I .+= 1
increment!(I::Integer) = increment(I)

function _jlmalloc(size, ::Type{T}) where {T}
    return ccall(:jl_malloc, Ptr{T}, (UInt,), size)
end
function _jlfree(p::Union{DenseVecOrMat{T},Ptr{T},Ref{T}}) where {T}
    return ccall(:jl_free, Cvoid, (Ptr{T},), p isa DenseVecOrMat ? pointer(p) : p)
end

function _sizedjlmalloc(n, ::Type{T}) where {T}
    return _jlmalloc(n * sizeof(T), T)
end

swapindices(::RowMajor, row, col) = row, col
swapindices(::ColMajor, row, col) = col, row
swapindices(::RowMajor, indices...) = indices
swapindices(::ColMajor, indices...) = reverse(indices)

# single threaded builder: COO -> CS/DCS
########################################
# TODO: This should be temporary. Or the very worst executor.
# Spartan should never use this.
# TODO: rather than ignore dups, combine them...
# TODO: If kept, it might be fastest to avoid sorting A when Tv2 !== Tv etc
# instead sort into Tv2 buffers to avoid copying later.
# TODO: use the double transpose alg, or the builder alg from Tim, ask him which is best for single thread.
# TODO: Since this is to be used for `convert` we might reuse buffers. Just needs to create buffer for pointers.
# pretty bad vs double transpose / builder but fine for testing.
function _build(
    ::Type{<:SinglyCompressedStore{<:Any,order}},
    A::CoordinateStore{Tv,Ti,<:Any,<:Any,2},
    coalesceop=+,
) where {order,Tv,Ti}
    A.sortorder = order
    indices, values, isuniform = _sortcoalesce(A, coalesceop)
    toptr, idx = swapindices(order, indices...)
    pointers = zeros(Ti, (order === ColMajor() ? size(A, 2) : size(A, 1)) + 1)
    for i in toptr
        pointers[i + 1] += 1
    end
    return SinglyCompressedStore(
        order,
        cumsum!(pointers, pointers) .+= 1,
        idx,
        isuniform ? similar(A.v, 0) : values,
        size(A);
        isuniform,
        uniformv=isuniform ? values : defaultfill(Tv),
    )
end

_combiner!(op, newV, i, oldV, j) = (newV[i] = op(newV[i], oldV[j]))
function _combiner!(op, newV::AbstractArray{Tv}, i, oldV::Tv, j) where {Tv}
    return (newV[i] = op(newV[i], oldV))
end

function _updater!(newV::V, i, oldV::V, j) where {V<:AbstractArray}
    (newV[i] = oldV[j]; return newV)
end
function _updater!(newV::AbstractArray{Tv}, i, oldV::Tv, j) where {Tv}
    (newV[i] = oldV; return newV)
end
_updater!(newV, i, oldV, j) = (newV) # uniform = uniform

Base.sortperm(x::Array{CIndex{T}}) where {T} = sortperm(reinterpret(T, x))

function _sortcoalesce(
    A::CoordinateStore{Tv,Ti,<:Any,<:Any,N}, coalesceop=second
) where {Tv,Ti,N}
    order = storageorder(A)
    (order === ColMajor() || order === RowMajor()) ||
        (throw(ArgumentError("Unknown sorting order.")))
    A.issorted &&
        A.iscoalesced &&
        (return A.indices, isuniformvalued(A) ? A.uniformv : A.v, isuniformvalued(A))
    linear = LinearIndices(size(A))
    linear = if order === RowMajor()
        PermutedDimsArray(LinearIndices(reverse(size(A))), ndims(linear):-1:1)
    else
        LinearIndices(size(A))
    end
    linearindices = zeros(Ti, nstored(A))
    indices = A.indices
    for i in eachindex(linearindices)
        linearindices[i] = linear[getindex.(indices, i)...]
    end
    perm = sortperm(linearindices)
    if A.iscoalesced
        return _sort(A, linearindices, perm)
    else
        return _sortcoalesce(A, linearindices, perm, coalesceop)
    end
end

function _sort(A, linearindices, perm)
    indices = A.indices
    newindices = copy.(indices)
    oldvalues = isuniformvalued(A) ? A.uniformv : A.v
    newvalues = isuniformvalued(A) ? A.uniformv : copy(oldvalues)
    newidx = 0
    previous_linearindex = -1
    for i in eachindex(perm)
        current_linearindex = linearindices[perm[i]]
        position = perm[i]
        newidx += 1
        setindex!.(newindices, getindex.(indices, position), newidx)
        newvalues = _updater!(newvalues, newidx, oldvalues, position)
        previous_linearindex = current_linearindex
    end
    return resize!.(newindices, newidx),
    isuniformvalued(A) ? newvalues : resize!(newvalues, newidx),
    isuniformvalued(A)
end
function _sortcoalesce(A, linearindices, perm, coalesceop)
    indices = A.indices
    newindices = copy.(indices)
    oldvalues = isuniformvalued(A) ? A.uniformv : A.v
    newvalues = isuniformvalued(A) ? similar(A.v, nstored(A)) : copy(oldvalues)
    newidx = 0
    previous_linearindex = -1
    for i in eachindex(perm)
        current_linearindex = linearindices[perm[i]]
        position = perm[i]
        if current_linearindex == previous_linearindex
            _combiner!(coalesceop, newvalues, newidx, oldvalues, position)
        else
            newidx += 1
            setindex!.(newindices, getindex.(indices, position), newidx)
            _updater!(newvalues, newidx, oldvalues, position)
        end
        previous_linearindex = current_linearindex
    end
    return resize!.(newindices, newidx), resize!(newvalues, newidx), false
end

# not very fast, but we can use for conversions until Spartan ready.
# TODO: support preallocation (mostly for eltype change)

# TODO: basic double transpose alg
