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
