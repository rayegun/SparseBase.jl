abstract type AbstractSparseStore{Tv,Order,Ti,V,I,N} end
function StorageOrders.comptime_storageorder(
    ::AbstractSparseStore{<:Any,Order}
) where {Order}
    return Order
end
abstract type AbstractSparseFormat{Tv,Tfill,Order,Ti,N} <: AbstractArray{Tv,N} end
function StorageOrders.comptime_storageorder(
    ::AbstractSparseFormat{<:Any,<:Any,Order}
) where {Order}
    return Order
end

Base.ndims(::AbstractSparseStore{<:Any,<:Any,<:Any,<:Any,<:Any,N}) where {N} = N
Base.size(A::AbstractSparseStore) = length.(axes(A))
Base.size(A::AbstractSparseStore, d) = d <= ndims(A) ? size(A)[d] : 1
Base.axes(A::AbstractSparseStore, d) = d <= ndims(A) ? axes(A)[d] : Base.OneTo(1)
const SparseStoreOrFormat{Tv,Order,Ti,N} = Union{
    <:AbstractSparseStore{Tv,Order,Ti,<:Any,<:Any,N},
    <:AbstractSparseFormat{Tv,<:Any,Order,Ti,N},
} where {Tv,Order,Ti,N}
Base.size(S::SparseStoreOrFormat, d) = d <= ndims(S) ? size(S)[d] : 1

"""
    filltype(A)

Type of implicit values of A. Most arrays either have no fill, or only support fill
in the same domain as eltype(A).
"""
filltype(T::Type) = eltype(T)
filltype(::Type{Base.Bottom}) = throw(ArgumentError("Union{} does not have elements"))
filltype(A) = filltype(typeof(A))
filltype(::Type{<:AbstractSparseFormat{<:Any, T}}) where {T} = @isdefined(T) ? T : Any

function filltype(::AbstractSparseStore)
    throw(
        ArgumentError(
            "Sparse stores have no fill, they must be wrapped in an AbstractSparseFormat"
        ),
    )
end

storedeltype(T::Type) = eltype(T)
storedeltype(A) = storedeltype(typeof(A))
storedeltype(::Type{<:SparseStoreOrFormat{T}}) where {T} = @isdefined(T) ? T : Any

indexeltype(::Type) = Any
indexeltype(::Type{Base.Bottom}) = throw(ArgumentError("Union{} does not have elements"))
indexeltype(A) = indexeltype(typeof(A))
indexeltype(::Type{<:SparseStoreOrFormat{<:Any, <:Any, T}}) where {T} = @isdefined(T) ? T : Any

Base.eltype(A::AbstractSparseFormat) = Union{filltype(A),storedeltype(A)}
Base.eltype(A::AbstractSparseStore) = storedeltype(A)
