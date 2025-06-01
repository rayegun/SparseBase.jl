module Communication
using ..SparseBase: AbstractSparseStore, CoordinateStore, SinglyCompressedStore
using ..SparseBase
using StorageOrders: storageorder, ColMajor, RowMajor, comptime_storageorder
using StorageOrders
# NOTE: This is likely a temporary home.
# Functions and stubs in this module are intended to enable
# (potentially inefficient) communication of SparseBase stores.
# Actual impl of communication functions will be in an ext with MPI (and UCX, etc?)
# Current impl uses point to point and collective. No RMA, which will be added later.

# Multiple messages will be sent for each object. This is somewhat unavoidable, the alternative is serialization.
# Only use after benchmarking against lowercase serialization based functions (should they exist)

# TODO: Serious pass improving interface here.
# Current commit based on bare minimum for SuperLU_DIST.
# TODO: Add back untested things not used by SuperLU_DIST.

function bcaststore! end

function sendstore end # TODO

function recvstore! end # TODO

function scatterstore! end
function scatterstore_memchunks! end

function gatherstore! end # TODO

# ContinuousPartitioning adapted from barche/MPIArrays.jl.
struct ContinuousPartitioning{N} <: AbstractArray{Int,N}
    ranks::LinearIndices{N,NTuple{N,Base.OneTo{Int}}}
    index_starts::NTuple{N,Vector{Int}}
    index_ends::NTuple{N,Vector{Int}}

    function ContinuousPartitioning(partition_sizes::Vararg{Any,N}) where {N}
        index_starts = Vector{Int}.(undef, length.(partition_sizes))
        index_ends = Vector{Int}.(undef, length.(partition_sizes))
        for (idxstart, idxend, nb_elems_dist) in
            zip(index_starts, index_ends, partition_sizes)
            currentstart = 1
            currentend = 0
            for i in eachindex(idxstart)
                currentend += nb_elems_dist[i]
                idxstart[i] = currentstart
                idxend[i] = currentend
                currentstart += nb_elems_dist[i]
            end
        end
        ranks = LinearIndices(length.(partition_sizes))
        return new{N}(ranks, index_starts, index_ends)
    end
end

localpart(A) = A

localsize(A) = size(localpart(A))
localsize(A, dim) = size(localpart(A), dim)

# returns memory splits for splitting A along its storage order into n chunks.
# for instance a CSCStore would be split into n chunks by column.
# further splitting by lower dimensions is not handled here.
# `extents` must be a vector of ranges or a vector of maximum rows | cols for each chunk.
function getmemchunks(A::AbstractSparseStore, extents)
    return _splitmem(A, extents)
end
getmemchunks(::Nothing, extents) = nothing

function _splitmem(A, part::ContinuousPartitioning)
    (count(i -> i == 1, size(part)) == 1 || sum(size(part)) == ndims(part)) ||
        error("_splitmem only supports splitting down a single dimension.")
    dim = storageorder(A) === RowMajor() ? 1 : ndims(A)
    return _splitmem(localpart(A), part.index_ends[dim])
end

function _splitmem(A::CoordinateStore, splits::Vector)
    order = storageorder(A)
    order === ColMajor() ||
        order === RowMajor() ||
        throw(ArgumentError("order ∉ {ColMajor(), RowMajor()}"))
    !A.issorted && (throw(ArgumentError("Cannot split unsorted COOStore")))

    leadingidx = order === ColMajor() ? A.indices[end] : A.indices[begin]
    memsizes = zeros(Int, length(splits))
    i = 1
    for j in eachindex(leadingidx)
        if leadingidx[j] > splits[i]
            i += 1
            i > length(memsizes) && break
        end
        memsizes[i] += 1
    end
    return ntuple(i -> memsizes, ndims(A) + 1)
end

function _splitmem(A::SinglyCompressedStore, splits::Vector)
    # memory size of idx and v for each split
    ptr = A.ptr
    # idxchunks = Int.(getindex.(Ref(ptr), getproperty.(splits, :stop) .+ 1)) .- 
    # Int.(getindex.(Ref(ptr), getproperty.(splits, :start)))
    idxchunks = diff([one(Int); Int.(getindex.(Ref(ptr), splits .+ 1))])
    # length.splits doesn't quite show the full story, since each chunk
    # will also need the next number up as well.
    # but that may not be overlapped during scatterv.
    return (diff([zero(eltype(splits)); splits]), idxchunks, idxchunks)
end

Base.IndexStyle(::Type{ContinuousPartitioning{N}}) where {N} = IndexCartesian()
Base.size(p::ContinuousPartitioning) = length.(p.index_starts)
@inline function Base.getindex(
    p::ContinuousPartitioning{N}, I::Vararg{Integer,N}
) where {N}
    return UnitRange.(getindex.(p.index_starts, I), getindex.(p.index_ends, I))
end
@inline function Base.getindex(p::ContinuousPartitioning{N}, I::Integer) where {N}
    indsub = Tuple(CartesianIndices(sum.(partition_sizes(p)))[I])
    return UnitRange.(getindex.(p.index_starts, indsub), getindex.(p.index_ends, indsub))
end
function partition_sizes(p::ContinuousPartitioning)
    result = (p.index_ends .- p.index_starts)
    for v in result
        v .+= 1
    end
    return result
end
function distribute_evenly(nb_elems, parts)
    local_len = nb_elems ÷ parts
    remainder = nb_elems % parts
    return [p <= remainder ? local_len + 1 : local_len for p in 1:parts]
end

"""
# Notes:
  - This is currently intended mostly for initial distribution and interfacing with external packages:
    SuperLU_DIST, CombBLAS, etc.
    The idea being to perform distribution of Julia vectors for use with these codes.
  - Eventually a port of DISTAL to be based on Finch/Spartan may be feasible.
"""
mutable struct DistributedSparseStore{Tv,Order,Ti,V,I,S,N,C} <:
               AbstractSparseStore{Tv,Order,Ti,V,I,N}
    globalsize::NTuple{N,Int}
    localstore::S
    partition::ContinuousPartitioning{N}
    comm::C
    # win::MPI.Win
end
localpart(D::DistributedSparseStore) = D.localstore
Base.size(D::DistributedSparseStore) = D.globalsize
"""
    localindices(a::DistributedSparseStore, rank::Integer)

Get the local index range (expressed in terms of global indices) of the given rank
"""
function localindices end

function DistributedSparseStore(
    localstore::AbstractSparseStore{Tv,Order,Ti,V,I,N}, partition, comm
) where {Tv,Order,Ti,V,I,N}
    partition = if partition isa ContinuousPartitioning
        partition
    else
        ContinuousPartitioning(partition...)
    end
    return DistributedSparseStore{Tv,Order,Ti,V,I,typeof(localstore),N,typeof(comm)}(
        maximum.(partition.index_ends), localstore, partition, comm
    )
end

function convertinnerstore(::Type{T}, D::DistributedSparseStore) where {T}
    if localpart(D) isa T
        return D
    else
        localstore = convert(T, localpart(D))
        return DistributedSparseStore(D.globalsize, localstore, D.partition, D.comm)
    end
end

for f in [
    :isuniformvalued,
    :hasfixedsparsity,
    :issparse,
    :iscoalesced,
    :isopaque,
    :storedeltype,
    :indexeltype,
]
    @eval begin
        SparseBase.$f(A::DistributedSparseStore) = SparseBase.$f(A.localstore)
    end
end
function StorageOrders.storageorder(D::DistributedSparseStore)
    return StorageOrders.storageorder(D.localstore)
end
end
