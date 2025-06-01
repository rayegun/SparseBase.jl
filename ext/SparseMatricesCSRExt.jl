module SparseMatricesCSRExt
using SparseBase, SparseArrays, SparseMatricesCSR
using SparseMatricesCSR:
    getrowptr, getcolval, sparsecsr
using SparseBase:
    SinglyCompressedStore,
    RowMajor,
    ColMajor,
    isuniformvalued,
    nstored,
    getexecutor,
    CSCStore,
    CSRStore,
    CoordinateStore
using CIndices
using StorageOrders
using SparseArrays: getcolptr, getrowval, getnzval, findnz
import LinearAlgebra: copy_oftype

SparseBase.issparse(::Type{<:SparseMatrixCSR}) = true
StorageOrders.comptime_storageorder(::Type{<:SparseMatrixCSR}) = RowMajor()

#TODO: determine if Cindices <-> Bi == 0 is possible no-copy

# convert SparseMatrixCSR < - > CSRStore
function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CSRStore{Tvalues}},
    A::SparseMatrixCSR{1,Tvalues,Tindex},
) where {Tvalues,Tindex}
    rowptrs, colvals, v = convert(Vector{Tindex}, getrowptr(A)), 
        convert(Vector{Tindex}, getcolval(A)), convert(Vector{Tvalues}, getnzval(A))
    return SinglyCompressedStore{Tvalues,RowMajor(),typeof(v),typeof(rowptrs),typeof(colvals),2}(
        rowptrs, colvals, v, size(A)
    )
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CSRStore{Tvalues}},
    A::SparseMatrixCSR{0,Tvalues,Tindex},
) where {Tvalues,Tindex}
    rowptrs, colvals = copy(reinterpret(CIndex{Tindex}, getrowptr(A))), 
        copy(reinterpret(CIndex{Tindex}, getcolval(A)))
    v = convert(Vector{Tvalues}, getnzval(A))
    return SinglyCompressedStore{Tvalues,RowMajor(),typeof(v),typeof(rowptrs),typeof(colvals),2}(
        rowptrs, colvals, v, size(A)
    )
end

function Base.convert(
    E::SparseBase.Executor,
    ::Type{<:CSRStore{Tvnew}},
    A::SparseMatrixCSR{<:Any, Tvalues,Tindex},
) where {Tvalues,Tindex,Tvnew}
    convert(E, CSRStore{Tvnew}, A)
end
function Base.convert(
    E::SparseBase.Executor,
    ::Type{<:CSRStore},
    A::SparseMatrixCSR{<:Any, Tvalues,Tindex},
) where {Tvalues,Tindex}
    convert(E, CSRStore{Tvalues}, A)
end

function Base.convert(T::Type{<:CSRStore}, s::SparseMatrixCSR)
    return convert(getexecutor(Base.convert, T, s), T, s)
end

function Base.convert(
    ::SparseBase.Executor, ::Type{SparseMatrixCSR{1, Tv,Ti}}, A::CSRStore{Tv,V,P,I,2}
) where {Tv,Ti,V,P,I}
    v = isuniformvalued(A) ? fill!(similar(A.v, Tv, length(A.idx)), A.uniformv) : A.v
    ptr = copy_oftype(A.ptr, Ti)
    idx = copy_oftype(A.idx, Ti)
    return SparseMatrixCSR{1}(
        size(A)..., ptr, idx, v
    )
end

function Base.convert(
    ::SparseBase.Executor, ::Type{SparseMatrixCSR{0, Tv,Ti}}, A::CSRStore{<:Any,<:Any,P,I,2}
) where {Tv,Ti,P,I}
    # Assuming P and I contain CIndex types for 0-based indexing
    rowptrs = copy_oftype(reinterpret(eltype(eltype(P)), A.ptr), Ti)
    colvals = copy_oftype(reinterpret(eltype(eltype(I)), A.idx), Ti)
    v = isuniformvalued(A) ? fill!(similar(A.v, Tv, length(A.idx)), A.uniformv) : A.v
    return SparseMatrixCSR{0}(
        size(A)..., rowptrs, colvals, convert(Vector{Tv}, v)
    )
end

function Base.convert(
    E::SparseBase.Executor, ::Type{<:SparseMatrixCSR}, A::CSRStore{Tv,<:Any,P,I,2}
) where {Tv,P,I}
    # Determine indexing base from element types
    Bi = (eltype(P) <: CIndex || eltype(I) <: CIndex) ? 0 : 1
    Ti = Bi == 0 ? Int : Int  # Default to Int for index type
    return convert(E, SparseMatrixCSR{Bi, Tv, Ti}, A)
end

function Base.convert(T::Type{<:SparseMatrixCSR}, s::CSRStore)
    return convert(getexecutor(Base.convert, T, s), T, s)
end

# COO < - > SparseMatrixCSR:
############################
function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CoordinateStore{Tvalues}},
    A::SparseMatrixCSR{1,Tvalues,Tindex},
) where {Tvalues,Tindex}
    rows, cols, v = findnz(A)
    return CoordinateStore(
        convert.(Vector{Tindex}, (rows, cols)), convert(Vector{Tvalues}, v), size(A); 
        issorted=true, iscoalesced=true, sortorder = RowMajor()
    )
end
function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CoordinateStore{Tvalues}},
    A::SparseMatrixCSR{0, Tvalues, Tindex},
) where {Tvalues,Tindex}
    rows, cols, v = findnz(A)
    rows, cols = copy(reinterpret(CIndex{Tindex}, rows)), copy(reinterpret(CIndex{Tindex}, cols))
    return CoordinateStore(
        (rows, cols), convert(Vector{Tvalues}, v), size(A); 
        issorted=true, iscoalesced=true, sortorder = RowMajor()
    )
end

function Base.convert(
    E::SparseBase.Executor,
    ::Type{<:CoordinateStore},
    A::SparseMatrixCSR{<:Any, Tvalues, Tindex},
) where {Tvalues,Tindex}
    return convert(E, CoordinateStore{Tvalues}, A)
end

function Base.convert(T::Type{<:CoordinateStore}, s::SparseMatrixCSR)
    return convert(getexecutor(Base.convert, T, s), T, s)
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{SparseMatrixCSR{1,Tv,Ti}},
    A::CoordinateStore{Tvalues, <:Any, <:Any, 2},
) where {Tv,Ti, Tvalues}
    v = isuniformvalued(A) ? fill!(similar(A.v, length(A.indices[1])), A.uniformv) : A.v
    rows, cols = convert.(AbstractArray{Ti}, A.indices)
    return sparsecsr(
        rows, cols,
        convert(AbstractArray{Tv}, v), size(A)...)
end

function Base.convert(
    E::SparseBase.Executor,
    ::Type{<:SparseMatrixCSR},
    A::CoordinateStore{Tvalues,<:Any,<:Any,2},
) where {Tvalues}
    # Default to 1-based indexing and Int type
    return convert(E, SparseMatrixCSR{1, Tvalues, Int}, A)
end

function Base.convert(T::Type{<:SparseMatrixCSR}, s::CoordinateStore)
    return convert(getexecutor(Base.convert, T, s), T, s)
end
end
