"""
    CoordinateStore{Tv, V, I, N} <: AbstractSparseStore{Tv, RuntimeOrder(), V, N}

# Parameters
  - `Tv`: The type of the stored values.
  - `V`: The type of the value storage array.
  - `I`: The type of the index storage arrays tuple.
  - `N`: The number of dimensions of the store.
"""
mutable struct CoordinateStore{Tv,V,I,N} <:
               AbstractSparseStore{Tv,RuntimeOrder(),V,N}
    indices::I # Struct of Arrays format, in lex order.
    v::V 

    bounds::NTuple{N,Int}

    isuniform::Bool
    uniformv::Tv

    issorted::Bool # If the indices are sorted (*except* the last dimension)
    sortorder::StorageOrder
    isjumbled::Bool # if the last dimension is sorted.
    iscoalesced::Bool
    function CoordinateStore{Tv,V,I,N}(
        indices::I,
        v::V,
        bounds::NTuple{N,<:Integer},
        isuniform::Bool,
        uniformv::Tv,
        issorted::Bool,
        sortorder::StorageOrder,
        iscoalesced::Bool,
        isjumbled::Bool
    ) where {Tv,V,I,N}
        all(i -> (length(i) == length(indices[1])), indices) ||
            throw(DimensionMismatch("All vectors in indices must have equal length."))
        # TODO: checkindices on input
        return new{Tv,V,I,N}(
            indices, v, bounds, isuniform, uniformv, issorted, sortorder, iscoalesced, isjumbled
        )
    end
end

function CoordinateStore{Tv,N}(
    Ti::Type{<:Integer},
    size=ntuple(i -> typemax(Ti), N);
    sortorder=ColMajor(),
    isuniform=false,
    uniformv=defaultfill(Tv),
    issorted=false,
    iscoalesced=false,
    isjumbled=true,
) where {Tv,N}
    indices = ntuple(i -> Vector{Ti}(), N)
    v = Vector{Tv}()
    return CoordinateStore{Tv,typeof(v),typeof(indices),N}(
        indices,
        v,
        size,
        isuniform,
        uniformv,
        issorted,
        sortorder,
        iscoalesced,
        isjumbled
    )
end

function CoordinateStore{Tv}(
    indices::NTuple{N},
    v,
    size=maximum.(indices);
    sortorder=ColMajor(),
    isuniform=false,
    uniformv=defaultfill(Tv),
    issorted=false,
    iscoalesced=false,
    isjumbled=true,
) where {Tv, N}
    v = convert(AbstractArray{Tv}, v)
    return CoordinateStore{Tv,typeof(v),typeof(indices),N}(
        indices, v,
        size, isuniform, uniformv, issorted, sortorder, iscoalesced, isjumbled
    )
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
    isjumbled=true,
) where {N,I,V}
    return CoordinateStore{eltype(V),V,typeof(indices),N}(
        indices, v, size, isuniform, uniformv, issorted, sortorder, iscoalesced, isjumbled
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

# Compressed store defns:
##########################
abstract type AbstractCompressedStore{Tvalues,Order,V,N} <:
              AbstractSparseStore{Tvalues,Order,V,N} end

mutable struct SinglyCompressedStore{Tvalues,Order,V,P,I,N} <:
               AbstractCompressedStore{Tvalues,Order,V,N}
    # Comments here reflect column major ordering. 
    # To understand as CSR simply swap references to columns with rows and vice versa.

    ptr::P # The pointers into idx/nzval.
    idx::I # the stored row indices.
    v::V # the values of the stored indices.

    vlen::Int # m in CSC, n in CSR. The maximum length of the sparse vectors.
    vdim::Int # n in CSC, m in CSR. The number of sparse vectors being stored.
    # This should be the length of ptr.

    isuniform::Bool
    uniformv::Tvalues
    isjumbled::Bool # whether last dimension is unordered (within the row | col)
end

# Most specific constructor - handles size unpacking
function SinglyCompressedStore{Tv, Order}(
    ptr::P, idx::I, v::V, size::Dims{2};
    isuniform=false,
    uniformv=defaultfill(Tv),
    isjumbled=false
) where {Tv, Order, P, I, V}
    size = Order === ColMajor() ? size : (size[2], size[1])
    return SinglyCompressedStore{Tv,Order,V,P,I,2}(
        ptr, idx, v,
        size..., isuniform, uniformv, isjumbled
    )
end

# Constructor with type conversion
function SinglyCompressedStore{Tv, Order}(
    Tptr::Type{<:Integer}, Tidx::Type{<:Integer},
    ptr, idx, v, size::Dims{2};
    isuniform=false,
    uniformv=defaultfill(Tv),
    isjumbled=false
) where {Tv, Order}
    ptr = convert(AbstractArray{Tptr}, ptr)
    idx = convert(AbstractArray{Tidx}, idx)
    v = convert(AbstractArray{Tv}, v)
    return SinglyCompressedStore{Tv,Order}(
        ptr, idx, v, size; isuniform, uniformv, isjumbled
    )
end

# Constructor with order parameter
function SinglyCompressedStore{Tv}(
    order::StorageOrder, ptr::P, idx::I, v, size::Dims{2};
    isuniform=false,
    uniformv=defaultfill(Tv),
    isjumbled=false
) where {Tv, P, I}
    v = convert(AbstractArray{Tv}, v)
    return SinglyCompressedStore{Tv,typeof(order)}(
        ptr, idx, v, size; isuniform, uniformv, isjumbled
    )
end

# Most general constructor
function SinglyCompressedStore(
    order::StorageOrder, ptr::P, idx::I, v::V, size::Dims{2};
    isuniform=false,
    uniformv=defaultfill(eltype(V)),
    isjumbled=false
) where {P, I, V}
    return SinglyCompressedStore{eltype(V)}(
        order, ptr, idx, v, size; isuniform, uniformv, isjumbled
    )
end

# Empty constructor
function SinglyCompressedStore{Tv, Order}(
    Tptr::Type{<:Integer}, Tidx::Type{<:Integer},
    size::Dims{2} = (typemax(Int), typemax(Int));
    isuniform=false, uniformv=defaultfill(Tv),
    isjumbled=false
) where {Tv, Order}
    return SinglyCompressedStore{Tv,Order}(
        Tptr, Tidx, Tptr[], Tidx[], Tv[], size; isuniform, uniformv, isjumbled
    )
end

const CSCStore{Tvalues,V,P,I,N} = SinglyCompressedStore{Tvalues,ColMajor(),V,P,I,N}
const CSRStore{Tvalues,V,P,I,N} = SinglyCompressedStore{Tvalues,RowMajor(),V,P,I,N}

function CSCStore(
    ptr::I, idx::I, v::V, size::Dims{2}; 
    isuniform=false, uniformv=defaultfill(eltype(V)), isjumbled=false
) where {I,V}
    return SinglyCompressedStore(
        ColMajor(), ptr, idx, v, size; 
        isuniform, uniformv, isjumbled
    )
end
function CSRStore(
    ptr::I, idx::I, v::V, size::Dims{2}; 
    isuniform=false, uniformv=defaultfill(eltype(V)), isjumbled=false
) where {I,V}
    return SinglyCompressedStore(
        RowMajor(), ptr, idx, v, size; 
        isuniform, uniformv, isjumbled
    )
end

mutable struct DoublyCompressedStore{Tvalues,Order,V,P,H,I,N} <:
               AbstractCompressedStore{Tvalues,Order,V,N}
    # Comments here reflect column major ordering. 
    # To understand as DCSR simply swap references to columns with rows and vice versa.

    ptr::P # The pointers into i/nzval. The row indices found in the k'th stored column are
    # found in idx[ptr[k], ptr[k+1]-1]

    # If column (row) j has stored entries, then j = h[k] for some k.
    # j is the k'th stored column.
    h::H
    idx::I # the stored row indices.
    v::V # the coefficients of stored indices in the matrix

    vlen::Int # m in CSC, n in CSR. This is the length of the vectors. In DCSC this is thus the number of rows.
    vdim::Int # n in CSC, m in CSR. This is the number of vectors being stored. in DCSC this is the number of columns.

    isuniform::Bool
    uniformv::Tvalues
    isjumbled::Bool # whether last dimension is unordered (within the row | col)
end

# Most specific constructor - handles size unpacking
function DoublyCompressedStore{Tv, Order}(
    ptr::P, h::H, idx::I, v::V, size::Dims{2};
    isuniform=false,
    uniformv=defaultfill(Tv), isjumbled=false
) where {Tv, Order, P, H, I, V}
    size = Order === ColMajor() ? size : (size[2], size[1])
    return DoublyCompressedStore{Tv,Order,V,P,H,I,2}(
        ptr, h, idx, v,
        size..., isuniform, uniformv, isjumbled
    )
end

# Constructor with type conversion
function DoublyCompressedStore{Tv, Order}(
    Tptr::Type{<:Integer}, Th::Type{<:Integer}, Tidx::Type{<:Integer},
    ptr, h, idx, v, size::Dims{2};
    isuniform=false,
    uniformv=defaultfill(Tv), isjumbled=false
) where {Tv, Order}
    ptr = convert(AbstractArray{Tptr}, ptr)
    h = convert(AbstractArray{Th}, h)
    idx = convert(AbstractArray{Tidx}, idx)
    v = convert(AbstractArray{Tv}, v)
    return DoublyCompressedStore{Tv,Order}(
        ptr, h, idx, v, size; isuniform, uniformv, isjumbled
    )
end

# Constructor with order parameter
function DoublyCompressedStore{Tv}(
    order::StorageOrder, ptr::P, h::H, idx::I, v, size::Dims{2};
    isuniform=false,
    uniformv=defaultfill(Tv), isjumbled=false
) where {Tv, P, H, I}
    v = convert(AbstractArray{Tv}, v)
    return DoublyCompressedStore{Tv,typeof(order)}(
        ptr, h, idx, v, size; isuniform, uniformv, isjumbled
    )
end

# Most general constructor
function DoublyCompressedStore(
    order::StorageOrder, ptr::P, h::H, idx::I, v::V, size::Dims{2};
    isuniform=false,
    uniformv=defaultfill(eltype(V)), isjumbled=false
) where {P, H, I, V}
    return DoublyCompressedStore{eltype(V)}(
        order, ptr, h, idx, v, size; isuniform, uniformv, isjumbled
    )
end

# Empty constructor
function DoublyCompressedStore{Tv, Order}(
    Tptr::Type{<:Integer}, Th::Type{<:Integer}, Tidx::Type{<:Integer},
    size::Dims{2} = (typemax(Int), typemax(Int)); # this is a very optimistic upperbound
    isuniform=false, uniformv=defaultfill(Tv), isjumbled=false
) where {Tv, Order}
    return DoublyCompressedStore{Tv,Order}(
        Tptr, Th, Tidx, Tptr[], Th[], Tidx[], Tv[], size; isuniform, uniformv, isjumbled
    )
end

const DCSCStore{Tvalues,V,P,H,I,N} = DoublyCompressedStore{Tvalues,ColMajor(),V,P,H,I,N}
const DCSRStore{Tvalues,V,P,H,I,N} = DoublyCompressedStore{Tvalues,RowMajor(),V,P,H,I,N}

function DCSCStore(
    ptr::I, h::I, idx::I, v::V, size::Dims{2};
    isuniform=false, uniformv=defaultfill(eltype(V)), isjumbled=false
) where {I,V}
    return DoublyCompressedStore(
        ColMajor(), ptr, h, idx, v, size;
        isuniform, uniformv, isjumbled
    )
end
function DCSRStore(
    ptr::I, h::I, idx::I, v::V, size::Dims{2};
    isuniform=false, uniformv=defaultfill(eltype(V)), isjumbled=false
) where {I,V}
    return DoublyCompressedStore(
        RowMajor(), ptr, h, idx, v, size;
        isuniform, uniformv, isjumbled
    )
end

Base.size(s::AbstractCompressedStore{<:Any,RowMajor()}) = s.vdim, s.vlen
Base.size(s::AbstractCompressedStore{<:Any,ColMajor()}) = s.vlen, s.vdim
Base.axes(s::AbstractCompressedStore{<:Any,RowMajor()}) = 1:(s.vdim), 1:(s.vlen)
Base.axes(s::AbstractCompressedStore{<:Any,ColMajor()}) = 1:(s.vlen), 1:(s.vdim)
nstored(S::AbstractCompressedStore) = length(S.v)

mutable struct ByteMapStore{Tvalues,Order,V,I,N} <:
               AbstractSparseStore{Tvalues,Order,V,N}
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
               AbstractSparseStore{Tvalues,ColMajor(),V,N}
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
# TODO: Dist formats here or higher level? I suspect higher level
# TODO: Hash, and add hash to DCS[R | C]
