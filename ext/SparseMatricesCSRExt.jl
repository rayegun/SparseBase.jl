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

# convert SparseMatrixCSR < - > CSCStore
function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CSRStore{Tvalues,Tindex}},
    A::SparseMatrixCSR{1},
) where {Tvalues,Tindex}
    rowptrs, colvals, v = convert(Vector{Tindex}, getrowptr(A)), 
        convert(Vector{Tindex}, getcolval(A)), convert(Vector{Tvalues}, getnzval(A))
    return SinglyCompressedStore{Tvalues,RowMajor(),Tindex,typeof(v),typeof(rowptrs),2}(
        rowptrs, colvals, v, size(A, 2), size(A, 1), false, zero(Tvalues)
    )
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CSRStore{Tvalues,CIndex{Tindex}}},
    A::SparseMatrixCSR{0,<:Any,Tindex},
) where {Tvalues,Tindex}
    rowptrs, colvals = copy(reinterpret(CIndex{Tindex}, getrowptr(A))), 
        copy(reinterpret(CIndex{Tindex}, getcolval(A)))
    v = convert(Vector{Tvalues}, getnzval(A))
    return SinglyCompressedStore{Tvalues,RowMajor(),CIndex{Tindex},typeof(v),typeof(rowptrs),2}(
        rowptrs, colvals, v, size(A, 2), size(A, 1), false, zero(Tvalues)
    )
end

# TODO: handle SparseMatrixCSR{0, Tvalues, Tindex} case for Tinew

function Base.convert(
    E::SparseBase.Executor,
    ::Type{<:CSRStore{Tvnew}},
    A::SparseMatrixCSR{<:Any, Tvalues,Tindex},
) where {Tvalues,Tindex,Tvnew}
    convert(E, CSRStore{Tvnew, Tindex}, A)
end
function Base.convert(
    E::SparseBase.Executor,
    ::Type{<:CSRStore},
    A::SparseMatrixCSR{<:Any, Tvalues,Tindex},
) where {Tvalues,Tindex}
    convert(E, CSRStore{Tvalues, Tindex}, A)
end

function Base.convert(T::Type{<:CSRStore}, s::SparseMatrixCSR)
    return convert(getexecutor(Base.convert, T, s), T, s)
end

function Base.convert(
    ::SparseBase.Executor, ::Type{SparseMatrixCSR{1, Tv,Ti}}, A::CSRStore
) where {Tv,Ti}
    v = isuniformvalued(A) ? fill!(similar(A.v, Tv, length(A.idx)), A.uniformv) : A.v
    return SparseMatrixCSR{1}(
        size(A)..., convert(Vector{Ti},A.ptr), convert(Vector{Ti},A.idx), convert(Vector{Tv},v)
    )
end

function Base.convert(
    ::SparseBase.Executor, ::Type{SparseMatrixCSR{0, Tv,Ti}}, A::CSRStore{<:Any, CIndex{Tiold}}
) where {Tv,Ti, Tiold}
    rowptrs = copy_oftype(reinterpret(Tiold, A.ptrs), Ti)
    colvals = copy_oftype(reinterpret(Tiold, A.idx), Ti)
    v = isuniformvalued(A) ? fill!(similar(A.v, Tv, length(A.idx)), A.uniformv) : A.v
    return SparseMatrixCSR{0}(
        size(A)..., rowptrs, colvals, convert(Tv,v)
    )
end

function Base.convert(
    E::SparseBase.Executor, ::Type{<:SparseMatrixCSR}, A::CSRStore{Tv, Ti}
) where {Tv, Ti}
    Bi = Ti <: CIndex ? 0 : 1
    return convert(E, SparseMatrixCSR{Bi, Tv, Ti}, A)
end

function Base.convert(T::Type{<:SparseMatrixCSR}, s::CSRStore)
    return convert(getexecutor(Base.convert, T, s), T, s)
end

# COO < - > SparseMatrixCSR:
############################
function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CoordinateStore{Tvalues,Tindex}},
    A::SparseMatrixCSR{1},
) where {Tvalues,Tindex}
    rows, cols, v = findnz(A)
    return CoordinateStore(
        convert.(Vector{Tindex}, (rows, cols)), convert(Vector{Tvalues}, v), size(A); 
        issorted=true, iscoalesced=true, sortorder = RowMajor()
    )
end
function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CoordinateStore{Tvalues,CIndex{Tindex}}},
    A::SparseMatrixCSR{0, <:Any, Tindex},
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
    return convert(E, CoordinateStore{Tvalues, Tindex}, A)
end

function Base.convert(T::Type{<:CoordinateStore}, s::SparseMatrixCSR)
    return convert(getexecutor(Base.convert, T, s), T, s)
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{SparseMatrixCSR{1,Tv,Ti}},
    A::CoordinateStore{Tvalues, Tindex, <:Any, <:Any, 2},
) where {Tv,Ti, Tvalues, Tindex}
    v = isuniformvalued(A) ? fill!(similar(A.v, length(A.indices[1])), A.uniformv) : A.v
    return sparsecsr(
        convert.(AbstractArray{Ti}, A.indices)...,
        convert(AbstractArray{Tv}, v), size(A)...)
end

function Base.convert(
    E::SparseBase.Executor,
    ::Type{<:SparseMatrixCSR},
    A::CoordinateStore{Tvalues,Tindex,<:Any,<:Any,2},
) where {Tvalues,Tindex}
    Bi = Tindex <: CIndex ? 0 : 1
    return convert(E, SparseMatrixCSR{Bi, Tvalues,Tindex}, A)
end

function Base.convert(T::Type{<:SparseMatrixCSR}, s::CoordinateStore)
    return convert(getexecutor(Base.convert, T, s), T, s)
end
end
