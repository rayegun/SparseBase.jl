module SparseBase

using StorageOrders: StorageOrders
using CIndices: CIndices
using CIndices: CIndex
import LinearAlgebra: copy_oftype
# storage order stuff:
const comptime_storageorder = StorageOrders.comptime_storageorder
const runtime_storageorder = StorageOrders.runtime_storageorder
const storageorder = StorageOrders.storageorder
const StorageOrder = StorageOrders.StorageOrder
const RowMajor = StorageOrders.RowMajor
const ColMajor = StorageOrders.ColMajor
const RuntimeOrder = StorageOrders.RuntimeOrder
const NoOrder = StorageOrders.NoOrder

include("abstract.jl")
include("novalue.jl")
include("structapi.jl")

include("operators.jl")
include("operations.jl")

include("stores.jl")
include("executors.jl")
include("base.jl")
include("utilities.jl")
include("communication.jl")
# ITERATION FUNCTIONALITY:
##########################

# I think we should do a few things. The first is define "backup" iteration functionality.
# This can be used to at least iterate things in a sparse manner. 
# With this capability in hand we could then have `@finch` macro that takes this for loop structure
# and compiles it to a much faster version.
# Design TBD.

# SOLVER FUNCTIONALITY:
#######################
# not yet sure on this one, starting to split solvers out right now.
# v0.2 I'll know more about what we need here to make this easier to impl

# include("conversion.jl")

export novalue, NoValue

export ColMajor, RowMajor, RuntimeOrder, NoOrder, storageorder
export indexeltype, storedeltype, isuniformvalued, hasfixedsparsity, iscoalesced, isopaque
end
