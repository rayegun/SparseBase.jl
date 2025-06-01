struct RunTimeValue end

struct DenseAxis{Extent} end
struct SparseAxis{Extent} end
struct FixedAxis{N} end
struct VariableAxis end

struct Axis{DS, FV, Itype, PtrType, IndicesType, ParentType}
    # PtrType = Nothing if fixedaxis | Union{Nothing, type of pointer vector} if variable axis.
    indptr::PtrType
    # Indicestype = Nothing if FixedAxis | type of indices vector if variable axis.
    indices::IndicesType
    # Given by Extent in DenseAxis{Extent} if available, maximum range of axis.
    size::Itype 
    # Given by N in FixedAxis{N} if available, number of stored values in axis.
    nstored::Int
    parent::ParentType
end


struct ContiguousAxisComponent{Itype, RangeType, Extent}
    extent::RangeType # Given by Extent in DenseAxis{Extent} if available, maximum range of axis.
end
struct NonContiguousAxisComponent{Itype, IndicesType, RangeType, Extent}
    extent::RangeType
    indices::IndicesType
end

struct FunctionIndexAxisComponenet{Itype, Fe, Fi}
    f_extent::Fe # function from index to extent
    f_index::Fi # function from index to index
end

struct FixedAxisComponent{Itype, N}
    nstored::Int # Given by N in FixedAxis{N} if available, number of stored values in axis.
end
struct VariableAxisComponent{Itype, IndicesPtrType}
    nstored::Itype # number of stored values in axis.
    indicesptr::IndicesPtrType 
end


