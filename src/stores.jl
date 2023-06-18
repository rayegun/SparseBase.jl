"""
    CoordinateStore

"""
mutable struct CoordinateStore{Tv,Ti,V,I,N} <:
               AbstractSparseStore{Tv,RuntimeOrder(),Ti,V,I,N}
    indices::NTuple{N,I}
    v::V

    bounds::NTuple{N,Int}

    isuniform::Bool
    uniformv::Tv

    issorted::Bool
    sortorder::StorageOrder
    iscoalesced::Bool
    function CoordinateStore{Tv,Ti,V,I,N}(
        indices::NTuple{N,I},
        v::V,
        bounds::NTuple{N,<:Integer},
        isuniform::Bool,
        uniformv::Tv,
        issorted::Bool,
        sortorder::StorageOrder,
        iscoalesced::Bool,
    ) where {Tv,Ti,V,I,N}
        all(i -> (length(i) == length(indices[1])), indices) ||
            throw(DimensionMismatch("All vectors in indices must have equal length."))
        # TODO: checkindices on input
        return new{Tv,Ti,V,I,N}(
            indices, v, bounds, isuniform, uniformv, issorted, sortorder, iscoalesced
        )
    end
end

function CoordinateStore(
    indices::NTuple{N,I},
    v::V,
    size=maximum.(indices);
    sortorder=ColMajor(),
    isuniform=false,
    uniformv=defaultfill(eltype(V)),
    issorted=false,
    iscoalesced=false,
) where {N,I,V}
    return CoordinateStore{eltype(V),eltype(I),V,I,N}(
        indices, v, size, isuniform, uniformv, issorted, sortorder, iscoalesced
    )
end

function CoordinateStore{Tv,Ti,N}(
    size=ntuple(i -> typemax(Ti), N);
    sortorder=ColMajor(),
    isuniform=false,
    uniformv=defaultfill(Tv),
    issorted=false,
    iscoalesced=false,
) where {Tv,Ti,N}
    return CoordinateStore(
        ntuple(i -> Vector{Ti}(), N),
        Vector{Tv}(),
        size;
        sortorder,
        isuniform,
        uniformv,
        issorted,
        iscoalesced,
    )
end

comptime_storageorder(::CoordinateStore) = RuntimeOrder()
runtime_storageorder(S::CoordinateStore) = S.sortorder
nstored(S::CoordinateStore) = length(S.v)
Base.axes(S::CoordinateStore) = (:).(1, S.bounds)
Base.axes(S::CoordinateStore, d) = axes(S)[d]
Base.issorted(S::CoordinateStore) = S.issorted

function sortcoalesce(C::CoordinateStore, coalesceop=second)
    indices, newvalues, isuniform = _sortcoalesce(C, coalesceop)
    uniformv = isuniform ? newvalues : defaultfill(eltype(C))
    return CoordinateStore(
        indices,
        isuniform ? similar(C.v, 0) : newvalues,
        size(C);
        issorted=true,
        iscoalesced=true,
        uniformv,
        isuniform,
        sortorder=storageorder(C),
    )
end

# function Base.similar(coo::CoordinateStore{Tv, Ti, V, I, N}) where {Tv, Ti, V, I, N}
#  #TODO
# end

# Compressed store defns:
##########################
abstract type AbstractCompressedStore{Tvalues,Order,Tindex,V,I,N} <:
              AbstractSparseStore{Tvalues,Order,Tindex,V,I,N} end

mutable struct SinglyCompressedStore{Tvalues,Order,Tindex,V,I,N} <:
               AbstractCompressedStore{Tvalues,Order,Tindex,V,I,N}
    # Comments here reflect column major ordering. 
    # To understand as CSR simply swap references to columns with rows and vice versa.

    ptr::I # The pointers into idx/nzval.
    idx::I # the stored row indices.
    v::V # the values of the stored indices.

    vlen::Int # m in CSC, n in CSR. The maximum length of the sparse vectors.
    vdim::Int # n in CSC, m in CSR. The number of sparse vectors being stored.
    # This should be the length of ptr.

    isuniform::Bool
    uniformv::Tvalues
end

function SinglyCompressedStore(
    order::StorageOrder,
    ptr::I,
    idx::I,
    v::V,
    size::Dims{2};
    isuniform=false,
    uniformv=defaultfill(eltype(V)),
) where {I,V}
    size = order === ColMajor() ? size : (size[2], size[1])
    return SinglyCompressedStore{eltype(V),order,eltype(I),V,I,2}(
        ptr, idx, v, size..., isuniform, uniformv
    )
end

CSCStore{Tvalues,Tindex,V,I,N} = SinglyCompressedStore{Tvalues,ColMajor(),Tindex,V,I,N}
CSRStore{Tvalues,Tindex,V,I,N} = SinglyCompressedStore{Tvalues,RowMajor(),Tindex,V,I,N}

function CSCStore(
    ptr::I, idx::I, v::V, size::Dims{2}; isuniform=false, uniformv=defaultfill(eltype(V))
) where {I,V}
    return SinglyCompressedStore(ColMajor(), ptr, idx, v, size; isuniform, uniformv)
end
function CSRStore(
    ptr::I, idx::I, v::V, size::Dims{2}; isuniform=false, uniformv=defaultfill(eltype(V))
) where {I,V}
    return SinglyCompressedStore(RowMajor(), ptr, idx, v, size; isuniform, uniformv)
end

CSCStore{Tv,Ti}() where {Tv,Ti} = CSCStore(Ti[], Ti[], Tv[], (0, 0))
CSRStore{Tv,Ti}() where {Tv,Ti} = CSRStore(Ti[], Ti[], Tv[], (0, 0))

mutable struct DoublyCompressedStore{Tvalues,Order,Tindex,V,I,N} <:
               AbstractCompressedStore{Tvalues,Order,Tindex,V,I,N}
    # Comments here reflect column major ordering. 
    # To understand as DCSR simply swap references to columns with rows and vice versa.

    ptr::I # The pointers into i/nzval. The row indices found in the k'th stored column are
    # found in idx[ptr[k], ptr[k+1]-1]

    # If column (row) j has stored entries, then j = h[k] for some k.
    # j is the k'th stored column.
    h::I
    idx::I # the stored row indices.
    v::V # the coefficients of stored indices in the matrix

    vlen::Int # m in CSC, n in CSR. This is the length of the vectors. In DCSC this is thus the number of rows.
    vdim::Int # n in CSC, m in CSR. This is the number of vectors being stored. in DCSC this is the number of columns.

    isuniform::Bool
    uniformv::Tvalues
end

function DoublyCompressedStore(
    order::StorageOrder,
    ptr::I,
    h::I,
    idx::I,
    v::V,
    size::Dims{2};
    isuniform=false,
    uniformv=defaultfill(eltype(V)),
) where {I,V}
    size = order === ColMajor() ? size : (size[2], size[1])
    return DoublyCompressedStore{eltype(V),order,eltype(I),V,I,N}(
        ptr, h, idx, v, size..., isuniform, uniformv
    )
end

Base.size(s::AbstractCompressedStore{<:Any,RowMajor()}) = s.vdim, s.vlen
Base.size(s::AbstractCompressedStore{<:Any,ColMajor()}) = s.vlen, s.vdim
Base.axes(s::AbstractCompressedStore{<:Any,RowMajor()}) = 1:(s.vdim), 1:(s.vlen)
Base.axes(s::AbstractCompressedStore{<:Any,ColMajor()}) = 1:(s.vlen), 1:(s.vdim)
nstored(S::AbstractCompressedStore) = length(S.v)

mutable struct ByteMapStore{Tvalues,Order,V,I,N} <:
               AbstractSparseStore{Tvalues,Order,Bool,V,I,N}
    bytemap::I
    v::V

    isuniform::Bool
    uniformv::Tvalues
    function ByteMapStore{Tvalues,Order,V,I,N}(
        bytemap::I, v::V, isuniform::Bool, uniformv::Tvalues
    ) where {Tvalues,Order,N,V<:AbstractArray{Tvalues,N},I<:AbstractArray{Bool,N}}
        return new{Tvalues,Order,V,I,N}(bytemap, v, isuniform, uniformv)
    end
end

function ByteMapStore(
    order::StorageOrder, bytemap::I, v::V; isuniform=false, uniformv=defaultfill(eltype(V))
) where {I,V}
    return ByteMapStore{eltype(V),order,V,I,ndims(V)}(bytemap, v, isuniform, uniformv)
end
function ByteMapStore(
    bytemap::I, v::V; isuniform=false, uniformv=defaultfill(eltype(V))
) where {I,V}
    return ByteMapStore{eltype(V),ColMajor(),V,I,ndims(V)}(bytemap, v, isuniform, uniformv)
end

Base.size(s::ByteMapStore{<:Any,ColMajor()}) = size(s.bytemap)
Base.size(s::ByteMapStore{<:Any,RowMajor()}) = size(s.bytemap')
nstored(S::ByteMapStore) = sum(S.bytemap)
# Dense store defns:

mutable struct AbstractArrayStore{Tvalues,V,N} <:
               AbstractSparseStore{Tvalues,ColMajor(),Int,V,Nothing,N}
    v::V
    function AbstractArrayStore{Tvalues,V,N}(
        v::V
    ) where {Tvalues,N,V<:AbstractArray{Tvalues,N}}
        return new{Tvalues,V,N}(v)
    end
end
AbstractArrayStore(v::V) where {V} = AbstractArrayStore{eltype(V),V,ndims(V)}(v)

Base.size(s::AbstractArrayStore) = size(s.v)
nstored(S::AbstractArrayStore) = length(S.v)

# Store structapi defns:
########################
isuniformvalued(::AbstractArrayStore) = false # Todo: fillArrays?
isuniformvalued(S::AbstractSparseStore) = S.isuniform

# TODO: DIA
# TODO: ELLPACK
# TODO: Dist formats here or higher level?
# TODO: Hash, and add hash to DCS[R | C]