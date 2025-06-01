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
    ::Type{<:CSCStore{Tvalues}},
    A::AbstractSparseMatrixCSC{Tvalues,Tindex},
) where {Tvalues,Tindex}
    colptrs, rowvals, v = getcolptr(A), getrowval(A), getnzval(A)
    return SinglyCompressedStore{Tvalues,ColMajor(),typeof(v),typeof(colptrs),typeof(rowvals),2}(
        colptrs, rowvals, v, size(A)
    )
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CSCStore{Tvnew}},
    A::AbstractSparseMatrixCSC{Tvalues,Tindex},
    Tptr::Type{<:Integer}=eltype(getcolptr(A)),
    Tidx::Type{<:Integer}=eltype(getrowval(A))
) where {Tvalues,Tindex,Tvnew}
    colptrs, rowvals, v = copy_oftype(getcolptr(A), Tptr),
    copy_oftype(getrowval(A), Tidx),
    Tvnew.(getnzval(A))
    return SinglyCompressedStore{Tvnew,ColMajor(),typeof(v),typeof(colptrs),typeof(rowvals),2}(
        colptrs, rowvals, v, size(A)
    )
end

function Base.convert(T::Type{<:CSCStore}, s::AbstractSparseMatrixCSC)
    return convert(getexecutor(Base.convert, T, s), T, s)
end

function Base.convert(
    ::SparseBase.Executor, ::Type{SparseMatrixCSC{Tv,Ti}}, A::CSCStore{Tv,V,P,I,2}
) where {Tv,Ti,V,P,I}
    v = isuniformvalued(A) ? fill!(similar(A.v, length(A.idx)), A.uniformv) : A.v
    ptr = copy_oftype(A.ptr, Ti)
    idx = copy_oftype(A.idx, Ti)
    return SparseMatrixCSC{Tv,Ti}(size(A)..., ptr, idx, v)
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{SparseMatrixCSC{Tvnew,Tinew}},
    A::CSCStore{Tv,<:Any,<:Any,<:Any,2},
) where {Tv,Tvnew,Tinew}
    v = if isuniformvalued(A)
        fill!(similar(A.v, Tvnew, length(A.idx)), A.uniformv)
    else
        Tvnew.(A.v)
    end
    return SparseMatrixCSC{Tvnew,Tinew}(
        size(A)..., copy_oftype(A.ptr, Tinew), copy_oftype(A.idx, Tinew), v
    )
end

function Base.convert(
    E::SparseBase.Executor,
    ::Type{SparseMatrixCSC},
    A::CSCStore{Tvalues,<:Any,<:Any,<:Any,2},
) where {Tvalues}
    return convert(E, SparseMatrixCSC{Tvalues,Int}, A)
end

function Base.convert(T::Type{<:SparseMatrixCSC}, s::CSCStore{Tvalues}) where {Tvalues}
    return convert(getexecutor(Base.convert, T, s), T, s)
end

# SparseMatrixCSC < - > CSRStore

function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CSRStore{Tvalues}},
    A::AbstractSparseMatrixCSC{Tvalues,Tindex},
) where {Tvalues,Tindex}
    A = copy(A')
    colptrs, rowvals, v = getcolptr(A), getrowval(A), getnzval(A)
    return SinglyCompressedStore{Tvalues,RowMajor(),typeof(v),typeof(colptrs),typeof(rowvals),2}(
        colptrs, rowvals, v, size(A)
    )
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CSRStore{Tvnew}},
    A::AbstractSparseMatrixCSC{Tvalues,Tindex},
    Tptr::Type{<:Integer}=Tindex,
    Tidx::Type{<:Integer}=Tindex
) where {Tvalues,Tindex,Tvnew}
    A = copy(A')
    colptrs, rowvals, v = copy_oftype(getcolptr(A), Tptr),
    copy_oftype(getrowval(A), Tidx),
    Tvnew.(getnzval(A))
    return SinglyCompressedStore{Tvnew,RowMajor(),typeof(v),typeof(colptrs),typeof(rowvals),2}(
        colptrs, rowvals, v, size(A)
    )
end

function Base.convert(
    E::SparseBase.Executor, ::Type{<:CSRStore}, A::AbstractSparseMatrixCSC{Tvalues,Tindex}
) where {Tvalues,Tindex}
    return convert(E, CSRStore{Tvalues}, A)
end

function Base.convert(T::Type{<:CSRStore}, s::AbstractSparseMatrixCSC)
    return convert(getexecutor(Base.convert, T, s), T, s)
end

function Base.convert(
    ::SparseBase.Executor, ::Type{SparseMatrixCSC{Tv,Ti}}, A::CSRStore{Tv,V,P,I,2}
) where {Tv,Ti,V,P,I}
    v = isuniformvalued(A) ? fill!(similar(A.v, length(A.idx)), A.uniformv) : A.v
    ptr = copy_oftype(A.ptr, Ti)
    idx = copy_oftype(A.idx, Ti)
    return copy(SparseMatrixCSC{Tv,Ti}(size(A)..., ptr, idx, v)')
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{SparseMatrixCSC{Tvnew,Tinew}},
    A::CSRStore{Tv,<:Any,<:Any,<:Any,2},
) where {Tv,Tvnew,Tinew}
    v = if isuniformvalued(A)
        fill!(similar(A.v, Tvnew, length(A.idx)), A.uniformv)
    else
        Tvnew.(A.v)
    end
    return copy(
        SparseMatrixCSC{Tvnew,Tinew}(
            size(A)..., copy_oftype(A.ptr, Tinew), copy_oftype(A.idx, Tinew), v
        )',
    )
end

function Base.convert(
    E::SparseBase.Executor,
    ::Type{SparseMatrixCSC},
    A::CSRStore{Tvalues,<:Any,<:Any,<:Any,2},
) where {Tvalues}
    return convert(E, SparseMatrixCSC{Tvalues,Int}, A)
end

function Base.convert(T::Type{<:SparseMatrixCSC}, s::CSRStore)
    return convert(getexecutor(Base.convert, T, s), T, s)
end

# COO < - > SparseMatrixCSC:
############################
function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CoordinateStore{Tvalues}},
    A::AbstractSparseMatrixCSC{Tvalues,Tindex},
) where {Tvalues,Tindex}
    rows, cols, v = findnz(A)
    return CoordinateStore((rows, cols), v, size(A); issorted=true, iscoalesced=true)
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{<:CoordinateStore{Tvnew}},
    A::AbstractSparseMatrixCSC{Tvalues,Tindex},
    Tidx::Type{<:Integer}=Tindex
) where {Tvalues,Tindex,Tvnew}
    rows, cols, v = findnz(A)
    return CoordinateStore(
        (convert(AbstractArray{Tidx}, rows), convert(AbstractArray{Tidx}, cols)),
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
    return convert(E, CoordinateStore{Tvalues}, A)
end

function Base.convert(T::Type{<:CoordinateStore}, s::AbstractSparseMatrixCSC)
    return convert(getexecutor(Base.convert, T, s), T, s)
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{SparseMatrixCSC{Tv,Ti}},
    A::CoordinateStore{Tv,<:Any,<:Any,2},
) where {Tv,Ti}
    v = isuniformvalued(A) ? fill!(similar(A.v, length(A.indices[1])), A.uniformv) : A.v
    rows, cols = convert.(AbstractArray{Ti}, A.indices)
    return SparseArrays.sparse(rows, cols, v, size(A)...)
end

function Base.convert(
    ::SparseBase.Executor,
    ::Type{SparseMatrixCSC{Tvnew,Tinew}},
    A::CoordinateStore{Tv,<:Any,<:Any,2},
) where {Tv,Tvnew,Tinew}
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
    A::CoordinateStore{Tvalues,<:Any,<:Any,2},
) where {Tvalues}
    return convert(E, SparseMatrixCSC{Tvalues,Int}, A)
end

function Base.convert(T::Type{<:SparseMatrixCSC}, s::CoordinateStore)
    return convert(getexecutor(Base.convert, T, s), T, s)
end
end
