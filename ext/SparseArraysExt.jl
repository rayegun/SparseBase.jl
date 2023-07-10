module SparseArraysExt
using SparseBase, SparseArrays
using SparseBase:
    SinglyCompressedStore,
    ColMajor,
    isuniformvalued,
    nstored,
    getexecutor,
    CSCStore,
    CSRStore,
    CoordinateStore
using SparseArrays: AbstractSparseMatrixCSC, getcolptr, getrowval, getnzval, findnz
import LinearAlgebra: copy_oftype
# TODO: This clashes.
SparseBase.issparse(::AbstractSparseMatrixCSC) = true
SparseBase.issparse(::Type{<:AbstractSparseMatrixCSC}) = true

hasfixedsparsity(A::AbstractSparseArray) = SparseArrays._is_fixed(A)

SparseBase.nstored(A::AbstractSparseArray) = SparseArrays.nnz(A)

# TODO: move onto interface based ctor and conversion

# convert SparseMatrixCSC < - > CSCStore
function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CSCStore{Tvalues,Tindex}},
    A::AbstractSparseMatrixCSC{Tvalues,Tindex},
) where {Tvalues,Tindex}
    colptrs, rowvals, v = getcolptr(A), getrowval(A), getnzval(A)
    return SinglyCompressedStore{Tvalues,ColMajor(),Tindex,typeof(v),typeof(rowvals),2}(
        colptrs, rowvals, v, size(A)
    )
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CSCStore{Tvnew,Tinew}},
    A::AbstractSparseMatrixCSC{Tvalues,Tindex},
) where {Tvalues,Tindex,Tvnew,Tinew}
    colptrs, rowvals, v = copy_oftype(getcolptr(A), Tinew),
    copy_oftype(getrowval(A), Tinew),
    Tvnew.(getnzval(A))
    return SinglyCompressedStore{Tvalues,ColMajor(),Tinew,typeof(v),typeof(rowvals),2}(
        colptrs, rowvals, v, size(A)
    )
end
function Base.convert(
    E::SparseBase.Executor,
    ::Type{<:CSCStore{Tvnew}},
    A::AbstractSparseMatrixCSC{Tvalues,Tindex},
) where {Tvalues,Tindex,Tvnew}
    convert(E, CSCStore{Tvnew, Tindex}, A)
end
function Base.convert(
    E::SparseBase.Executor,
    ::Type{<:CSCStore},
    A::AbstractSparseMatrixCSC{Tvalues,Tindex},
) where {Tvalues,Tindex}
    convert(E, CSCStore{Tvalues, Tindex}, A)
end

function Base.convert(T::Type{<:CSCStore}, s::AbstractSparseMatrixCSC)
    return convert(getexecutor(Base.convert, T, s), T, s)
end

function Base.convert(
    ::SparseBase.Executor, ::Type{SparseMatrixCSC{Tv,Ti}}, A::CSCStore{Tv,Ti,V,I,2}
) where {Tv,Ti,V,I}
    v = isuniformvalued(A) ? fill!(similar(A.v, length(A.idx)), A.uniformv) : A.v
    return SparseMatrixCSC{Tv,Ti}(size(A)..., A.ptr, A.idx, v)
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{SparseMatrixCSC{Tvnew,Tinew}},
    A::CSCStore{Tv,Ti,<:Any,<:Any,2},
) where {Tv,Ti,Tvnew,Tinew}
    v = if isuniformvalued(A)
        fill!(similar(A.v, Tvnew, length(A.idx)), A.uniformv)
    else
        Tvnew.(A.v)
    end
    return SparseMatrixCSC{Tv,Ti}(
        size(A)..., copy_oftype(A.ptr, Tinew), copy_oftype(A.idx, Tinew), v
    )
end

function Base.convert(
    E::SparseBase.Executor,
    ::Type{SparseMatrixCSC},
    A::CSCStore{Tvalues,Tindex,<:Any,<:Any,2},
) where {Tvalues,Tindex}
    return convert(E, SparseMatrixCSC{Tvalues,Tindex}, A)
end

function Base.convert(T::Type{<:SparseMatrixCSC}, s::CSCStore{Tvalues}) where {Tvalues}
    return convert(getexecutor(Base.convert, T, s), T, s)
end

# SparseMatrixCSC < - > CSRStore

function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CSRStore{Tvalues,Tindex}},
    A::AbstractSparseMatrixCSC{Tvalues,Tindex},
) where {Tvalues,Tindex}
    A = copy(A')
    colptrs, rowvals, v = getcolptr(A), getrowval(A), getnzval(A)
    return SinglyCompressedStore{Tvalues,RowMajor(),Tindex,typeof(v),typeof(rowvals),2}(
        colptrs, rowvals, v, size(A)
    )
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CSRStore{Tvnew,Tinew}},
    A::AbstractSparseMatrixCSC{Tvalues,Tindex},
) where {Tvalues,Tindex,Tvnew,Tinew}
    A = copy(A')
    colptrs, rowvals, v = copy_oftype(getcolptr(A), Tinew),
    copy_oftype(getrowval(A), Tinew),
    Tvnew.(getnzval(A))
    return SinglyCompressedStore{Tvalues,RowMajor(),Tinew,typeof(v),typeof(rowvals),2}(
        colptrs, rowvals, v, size(A)
    )
end

function Base.convert(
    E::SparseBase.Executor, ::Type{<:CSRStore}, A::AbstractSparseMatrixCSC{Tvalues,Tindex}
) where {Tvalues,Tindex}
    return convert(E, SinglyCompressedStore{Tvalues,RowMajor(),Tindex}, A)
end

function Base.convert(T::Type{<:CSRStore}, s::AbstractSparseMatrixCSC)
    return convert(getexecutor(Base.convert, T, s), T, s)
end

function Base.convert(
    ::SparseBase.Executor, ::Type{SparseMatrixCSC{Tv,Ti}}, A::CSRStore{Tv,Ti,V,I,2}
) where {Tv,Ti,V,I}
    v = isuniformvalued(A) ? fill!(similar(A.v, length(A.idx)), A.uniformv) : A.v
    return copy(SparseMatrixCSC{Tv,Ti}(size(A)..., A.ptr, A.idx, v)')
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{SparseMatrixCSC{Tvnew,Tinew}},
    A::CSRStore{Tv,Ti,<:Any,<:Any,2},
) where {Tv,Ti,Tvnew,Tinew}
    v = if isuniformvalued(A)
        fill!(similar(A.v, Tvnew, length(A.idx)), A.uniformv)
    else
        Tvnew.(A.v)
    end
    return copy(
        SparseMatrixCSC{Tv,Ti}(
            size(A)..., copy_oftype(A.ptr, Tinew), copy_oftype(A.idx, Tinew), v
        )',
    )
end

function Base.convert(
    E::SparseBase.Executor,
    ::Type{SparseMatrixCSC},
    A::CSRStore{Tvalues,Tindex,<:Any,<:Any,2},
) where {Tvalues,Tindex}
    return convert(E, SparseMatrixCSC{Tvalues,Tindex}, A)
end

function Base.convert(T::Type{<:SparseMatrixCSC}, s::CSRStore)
    return convert(getexecutor(Base.convert, T, s), T, s)
end

# COO < - > SparseMatrixCSC:
############################
function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CoordinateStore{Tvalues,Tindex}},
    A::AbstractSparseMatrixCSC{Tvalues,Tindex},
) where {Tvalues,Tindex}
    rows, cols, v = findnz(A)
    return CoordinateStore((rows, cols), v, size(A); issorted=true, iscoalesced=true)
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CoordinateStore{Tvnew,Tinew}},
    A::AbstractSparseMatrixCSC{Tvalues,Tindex},
) where {Tvalues,Tindex,Tvnew,Tinew}
    rows, cols, v = findnz(A)
    return CoordinateStore(
        (convert(AbstractArray{Tinew}, rows), convert(AbstractArray{Tinew}, cols)),
        convert(AbstractArray{Tvnew}, v),
        size(A);
        issorted=true,
        iscoalesced=true,
    )
end

function Base.convert(
    E::SparseBase.Executor,
    ::Type{<:CoordinateStore},
    A::AbstractSparseMatrixCSC{Tvalues,Tindex},
) where {Tvalues,Tindex}
    return convert(E, CoordinateStore{Tvalues,Tindex}, A)
end

function Base.convert(T::Type{<:CoordinateStore}, s::AbstractSparseMatrixCSC)
    return convert(getexecutor(Base.convert, T, s), T, s)
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{SparseMatrixCSC{Tv,Ti}},
    A::CoordinateStore{Tv,Ti,<:Any,<:Any,2},
) where {Tv,Ti}
    v = isuniformvalued(A) ? fill!(similar(A.v, length(A.indices[1])), A.uniformv) : A.v
    return SparseArrays.sparse(A.indices..., v, size(A)...)
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{SparseMatrixCSC{Tvnew,Tinew}},
    A::CoordinateStore{Tv,Ti,<:Any,<:Any,2},
) where {Tv,Ti,Tvnew,Tinew}
    v = if isuniformvalued(A)
        fill!(similar(A.v, Tvnew, length(A.indices[1])), A.uniformv)
    else
        convert(AbstractArray{Tvnew}, A.v)
    end
    rows, cols = convert.(AbstractArray{Tinew}, A.indices)
    return SparseArrays.sparse(rows, cols, v, size(A)...)
end

function Base.convert(
    E::SparseBase.Executor,
    ::Type{SparseMatrixCSC},
    A::CoordinateStore{Tvalues,Tindex,<:Any,<:Any,2},
) where {Tvalues,Tindex}
    return convert(E, SparseMatrixCSC{Tvalues,Tindex}, A)
end

function Base.convert(T::Type{<:SparseMatrixCSC}, s::CoordinateStore)
    return convert(getexecutor(Base.convert, T, s), T, s)
end
end
