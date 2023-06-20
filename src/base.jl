# Base defns for stores and formats:

# Base.issorted
###############
# most are sorted by default, except COO I believe.
# Compressed formats can support but we don't allow.
Base.issorted(S::AbstractSparseStore) = true

# self Base.convert methods:
#######################

function Base.convert(
    ::Type{<:CoordinateStore{Tv,Ti}}, A::CoordinateStore{Tv1,Ti1}
) where {Tv,Ti,Tv1,Ti1}
    indices = Ti === Ti1 ? A.indices : copy_oftype.(A.indices, Ti)
    v = Tv === Tv1 ? A.v : copy_oftype(A.v, Tv)
    return CoordinateStore(
        indices,
        v,
        A.bounds;
        A.sortorder,
        A.isuniform,
        uniformv=Tv(A.uniformv),
        A.issorted,
        A.iscoalesced,
    )
end
function Base.convert(
    ::Type{<:CoordinateStore{Tv}}, A::CoordinateStore{Tv1,Ti1}
) where {Tv,Tv1,Ti1}
    return convert(CoordinateStore{Tv,Ti1}, A)
end
Base.convert(::Type{<:CoordinateStore}, A::CoordinateStore) = A

function Base.convert(
    ::Type{<:SinglyCompressedStore{Tv,O,Ti}}, A::SinglyCompressedStore{Tv1,O,Ti1}
) where {Tv,Ti,Tv1,Ti1,O}
    ptr = Ti === Ti1 ? A.ptr : copy_oftype(A.ptr, Ti)
    idx = Ti === Ti1 ? A.idx : copy_oftype(A.idx, Ti)
    v = Tv === Tv1 ? A.v : copy_oftype(A.v, Tv)
    return SinglyCompressedStore(
        storageorder(A), ptr, idx, v, size(A); A.isuniform, A.uniformv
    )
end
function Base.convert(
    ::Type{<:SinglyCompressedStore{Tv}}, A::SinglyCompressedStore{Tv1,<:Any,Ti1}
) where {Tv,Tv1,Ti1}
    return convert(SinglyCompressedStore{Tv,storageorder(A),Ti1}, A)
end

function Base.convert(
    ::Type{<:DoublyCompressedStore{Tv,O,Ti}}, A::DoublyCompressedStore{Tv1,O,Ti1}
) where {Tv,Ti,Tv1,Ti1,O}
    ptr = Ti === Ti1 ? A.ptr : copy_oftype(A.ptr, Ti)
    idx = Ti === Ti1 ? A.idx : copy_oftype(A.idx, Ti)
    h = Ti === Ti1 ? A.h : copy_oftype(A.h, Ti)
    v = Tv === Tv1 ? A.v : copy_oftype(A.v, Tv)
    return DoublyCompressedStore(
        storageorder(A), ptr, h, idx, v, A.size; A.isuniform, A.uniformv
    )
end
function Base.convert(
    ::Type{<:DoublyCompressedStore{Tv}}, A::DoublyCompressedStore{Tv1,<:Any,Ti1}
) where {Tv,Tv1,Ti1}
    return convert(DoublyCompressedStore{Tv,storageorder(A),Ti1}, A)
end

function Base.convert(::Type{<:ByteMapStore{Tv,O}}, A::ByteMapStore{Tv1,O}) where {Tv,Tv1,O}
    v = Tv === Tv1 ? A.v : copy_oftype(A.v, Tv)
    return ByteMapStore(storageorder(A), v; A.isuniform, A.uniformv)
end
Base.convert(::Type{<:ByteMapStore}, A::ByteMapStore) = A

# AbstractArray conversions:
############################

function Base.convert(
    ::Type{<:CoordinateStore{Tv,Ti}}, A::AbstractArray{Tv1}; checkindices=false
) where {Tv,Ti,Tv1}
    v = similar(A, Tv, length(A))
    copyto!(v, A)
    cartindices = CartesianIndices(A)
    indices = reverse(ntuple(i -> Ti.(getindex.(cartindices, i)), ndims(A)))
    return CoordinateStore(
        indices,
        v,
        size(A);
        sortorder=storageorder(A),
        isuniform=false,
        uniformv=defaultfill(Tv),
        issorted=true,
        checkindices,
        iscoalesced=true,
    )
end
function Base.convert(::Type{<:CoordinateStore{Tv}}, A::AbstractArray{Tv1}) where {Tv,Tv1}
    return convert(CoordinateStore{Tv,Int}, A)
end
function Base.convert(::Type{<:CoordinateStore}, A::AbstractArray{Tv1}) where {Tv1}
    return convert(CoordinateStore{Tv1}, A)
end

# COO -> others conversion:
function Base.convert(T::Type{<:SinglyCompressedStore}, A::CoordinateStore)
    return _build(T, A)
end

# Row < - > Col:
# rest should get iterators working.
