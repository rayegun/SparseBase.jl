module MPIExt
using MPI
using SparseBase
using SparseBase: CoordinateStore, nstored, SinglyCompressedStore, isuniformvalued
using SparseBase.Communication
using SparseBase.Communication: ContinuousPartitioning, localpart, partition_sizes

# TODO: vector types should be handled better.

# function Communication.localindices(D::DistributedSparseStore)
#     return D.partition[MPI.Comm_rank(D.comm) + 1]
# end

function Communication.bcaststore!(store::CoordinateStore, root::Integer, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    store.bounds, ns, store.isuniform, store.issorted, store.iscoalesced, store.sortorder, store.uniformv = MPI.Bcast(
        (
            size(store),
            nstored(store),
            isuniformvalued(store),
            issorted(store),
            store.iscoalesced,
            storageorder(store),
            store.uniformv,
        ),
        root,
        comm,
    )
    if rank != root
        resize!.(store.indices, ns)
        resize!(store.v, ns)
    end
    for index in store.indices
        MPI.Bcast!(index, root, comm)
    end
    MPI.Bcast!(store.v, root, comm)
    return store
end

function Communication.bcaststore!(
    store::SinglyCompressedStore, root::Integer, comm::MPI.Comm
)
    rank = MPI.Comm_rank(comm)
    store.vlen, store.vdim, ns, store.isuniform, store.uniformv = MPI.Bcast(
        (store.vlen, store.vdim, nstored(store), isuniformvalued(store), store.uniformv),
        root,
        comm,
    )
    if rank != root
        resize!(store.ptr, store.vdim + 1)
        resize!(store.idx, ns)
        resize!(store.v, ns)
    end
    MPI.Barrier(comm)
    MPI.Bcast!(store.ptr, root, comm)
    MPI.Bcast!(store.idx, root, comm)
    MPI.Bcast!(store.v, root, comm)
    return store
end

# TODO: sendstore and recvstore!

# TODO: These names are terrible.
# chunksize? memsize?
# chunksizes may (should) be nothing on non-root
# returns (rstore, localindices, globalindices)

function scatter_memchunk!(
    out,
    in,
    rootmemsizes;
    root=0,
    comm=MPI.COMM_WORLD,
    localmemsize=MPI.Scatter(
        MPI.Comm_rank(comm) == root ? rootmemsizes : nothing, Int, comm; root
    ),
)
    resize!(out, localmemsize)
    vbuff = MPI.Comm_rank(comm) == root ? VBuffer(in, rootmemsizes) : nothing
    return MPI.Scatterv!(vbuff, out, root, comm)
end

# TODO: Somehow this became the CompressedStore impl?
# switch back to COO impl :/
# function Communication.scatterstore!(
#     rstore::CoordinateStore,
#     sstore::Union{Nothing,CoordinateStore},
#     chunksizes; # dimchunks is an iterable of the number of rows / cols (CSR / CSC)
#     # in each rank.
#     root::Integer=0,
#     comm::MPI.Comm=MPI.COMM_WORLD,
# )
#     rank = MPI.Comm_rank(comm)
#     isroot = rank == root
#     metasource = isroot ? sstore : rstore
#     dimtosplit = storageorder(rstore) === RowMajor() ? 1 : ndims(store)
#     rstore.isuniform, rstore.issorted, rstore.iscoalesced, rstore.sortorder, rstore.uniformv, rstore.bounds = MPI.Bcast(
#         (
#             isuniformvalued(metasource),
#             issorted(metasource),
#             metasource.iscoalesced,
#             storageorder(metasource),
#             metasource.uniformv,
#             size(metasource),
#         ),
#         root,
#         comm,
#     )
# 
#     part = isroot ? Communication.ContinuousPartitioning(chunksizes, sstore.vlen) : nothing
#     isroot && (@show part)
#     memsizes = Communication.getmemchunks(sstore, part)
# 
#     # At this point rstore still needs vdim, first_dim and vectors.
#     # Global size must also be moved from vdim -> a return value or it will be lost.
#     globalvdim = rstore.vdim
#     first_dim = MPI.Scatter(
#         isroot ? (part.index_starts[dimtosplit]) : nothing, Int, comm; root
#     )
#     ptrmemsizes = isroot ? memsizes[begin] : nothing
#     ptrchunksize = MPI.Scatter(ptrmemsizes, Int, comm; root)
# 
#     # Scatter the ptrs for each chunk *except* the last index which overlaps.
#     scatter_memchunk!(
#         rstore.ptr,
#         isroot ? sstore.ptr : nothing,
#         ptrmemsizes;
#         root,
#         comm,
#         localmemsize=ptrchunksize + 1,
#     )
#     # Scatter the overlapped ptr indices.
#     finalptrs =
#         !isroot ? nothing : append!((sstore.ptr[cumsum(ptrmemsizes) .+ 1]), sstore.ptr[end])
#     rstore.ptr[end] = MPI.Scatter(finalptrs, eltype(rstore.ptr), comm; root)
#     rstore.ptr .-= (rstore.ptr[begin] .- one(eltype(rstore.ptr))) # start ptrs at {0 | 1}
# 
#     idxmemsizes = isroot ? memsizes[end - 1] : nothing
#     idxchunksize = MPI.Scatter(idxmemsizes, Int, comm; root)
#     scatter_memchunk!(
#         rstore.idx,
#         isroot ? sstore.idx : nothing,
#         idxmemsizes;
#         root,
#         comm,
#         localmemsize=idxchunksize,
#     )
#     # May not need to send values if isuniform (pattern).
#     if !SparseBase.isuniformvalued(rstore)
#         scatter_memchunk!(
#             rstore.v,
#             isroot ? sstore.v : nothing,
#             idxmemsizes;
#             root,
#             comm,
#             localmemsize=idxchunksize,
#         )
#     end
#     rstore.vdim = length(rstore.ptr) - 1
#     return rstore, globalvdim, first_dim
# end

function Communication.scatterstore!(
    rstore::SinglyCompressedStore,
    sstore::Union{Nothing,SinglyCompressedStore},
    chunksizes; # chunksizes is an iterable of the number of rows / cols (CSR / CSC)
    # in each rank.
    root::Integer=0,
    comm::MPI.Comm=MPI.COMM_WORLD,
)
    rank = MPI.Comm_rank(comm)
    isroot = rank == root
    metasource = isroot ? sstore : rstore
    dimtosplit = storageorder(rstore) === RowMajor() ? 1 : ndims(store)
    rstore.isuniform, rstore.uniformv, rstore.vlen, rstore.vdim = MPI.Bcast(
        (
            isuniformvalued(metasource),
            metasource.uniformv,
            metasource.vlen,
            metasource.vdim,
        ),
        root,
        comm,
    )

    part = isroot ? Communication.ContinuousPartitioning(chunksizes, sstore.vlen) : nothing
    isroot && (@show part)
    memsizes = Communication.getmemchunks(sstore, part)

    # At this point rstore still needs vdim, first_dim and vectors.
    # Global size must also be moved from vdim -> a return value or it will be lost.
    globalvdim = rstore.vdim
    first_dim = MPI.Scatter(
        isroot ? (part.index_starts[dimtosplit]) : nothing, Int, comm; root
    )
    ptrmemsizes = isroot ? memsizes[begin] : nothing
    ptrchunksize = MPI.Scatter(ptrmemsizes, Int, comm; root)

    # Scatter the ptrs for each chunk *except* the last index which overlaps.
    scatter_memchunk!(
        rstore.ptr,
        isroot ? sstore.ptr : nothing,
        ptrmemsizes;
        root,
        comm,
        localmemsize=ptrchunksize + 1,
    )
    # Scatter the overlapped ptr indices.
    finalptrs =
        !isroot ? nothing : append!((sstore.ptr[cumsum(ptrmemsizes) .+ 1]), sstore.ptr[end])
    rstore.ptr[end] = MPI.Scatter(finalptrs, eltype(rstore.ptr), comm; root)
    rstore.ptr .-= (rstore.ptr[begin] .- one(eltype(rstore.ptr))) # start ptrs at {0 | 1}

    idxmemsizes = isroot ? memsizes[end - 1] : nothing
    idxchunksize = MPI.Scatter(idxmemsizes, Int, comm; root)
    scatter_memchunk!(
        rstore.idx,
        isroot ? sstore.idx : nothing,
        idxmemsizes;
        root,
        comm,
        localmemsize=idxchunksize,
    )
    # May not need to send values if isuniform (pattern).
    if !SparseBase.isuniformvalued(rstore)
        scatter_memchunk!(
            rstore.v,
            isroot ? sstore.v : nothing,
            idxmemsizes;
            root,
            comm,
            localmemsize=idxchunksize,
        )
    end
    rstore.vdim = length(rstore.ptr) - 1
    # not sufficient to construct a ContinuousPartitioning on each rank.
    return rstore, globalvdim, first_dim
end
end
