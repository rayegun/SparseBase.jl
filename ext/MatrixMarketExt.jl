module MatrixMarketExt
using MatrixMarket
using MatrixMarket:
    parse_eltype,
    parse_symmetric,
    mmread,
    num_splits,
    find_splits,
    parse_val,
    parse_row,
    parse_col
using SparseBase: CoordinateStore, defaultfill, SinglyCompressedStore
function _mmread(filename::String, ::Type{Tv}, ::Type{Ti}) where {Tv,Ti}
    stream = open(filename, "r")
    result = _mmread(stream, Tv, Ti)
    close(stream)
    return result
end
function _mmread(stream::IO, ::Type{Tv}, ::Type{Ti}) where {Tv,Ti}
    rows, cols, entries, rep, field, symm = mmread(stream, true)

    T = parse_eltype(field)
    symfunc = parse_symmetric(symm)

    if rep == "coordinate"
        getvals = (field != "pattern")
        rn = Vector{Ti}(undef, entries)
        cn = Vector{Ti}(undef, entries)
        vals = getvals ? Vector{Tv}(undef, entries) : one(Tv)
        for i in 1:entries
            line = readline(stream)
            splits = find_splits(line, num_splits(T))
            rn[i] = convert(Ti, parse_row(line, splits))
            cn[i] = convert(Ti, parse_col(line, splits, T))
            getvals && (vals[i] = convert(Tv, parse_val(line, splits, T)))
        end

        result = (rn, cn, vals, rows, cols, entries, rep, field, symm)
    else
        vals = [Tv(parse(Float64, readline(stream))) for _ in 1:entries]
        A = reshape(vals, rows, cols)
        result = symfunc(A)
    end

    return result
end

function desymmetrizer(rows, cols, values, symm)
    if symm != "general"
        l = length(rows)
        change_values = true
        if symm == "hermitian"
            f = conj
            if !(values isa AbstractArray)
                values = fill(values, 2l)
            end
        elseif symm == "symmetric"
            f = identity
            if values isa AbstractArray
                resize!(values, 2l)
            else
                change_values = false
            end
        elseif symm == "skew-symmetric"
            f = -
            if !(values isa AbstractArray)
                values = fill(values, 2l)
            end
        else
            throw(MatrixMarket.FileFormatException("Unknown matrix symmetry: $symm."))
        end
        change_values ? _desymmetrizer!(f, rows, cols, values) : _desymmetrizer(rows, cols)
    end
    return rows, cols, values
end

function _desymmetrizer!(f, rows, cols, values::AbstractArray)
    l = length(rows)
    resize!.((rows, cols), 2l) #pessimistic, no values on diagonal
    n = 1
    for i in 1:l
        r, c = rows[i], cols[i]
        if r == c
            continue
        else
            rows[n + l], cols[n + l] = c, r
            (values[n + l] = f(values[i]))
            n += 1
        end
    end
    return resize!.((rows, cols, values), l + n - 1)
end

function _desymmetrizer!(rows, cols)
    l = length(rows)
    resize!.((rows, cols), 2l) #pessimistic, no values on diagonal
    n = 1
    for i in 1:l
        r, c = rows[i], cols[i]
        if r == c
            continue
        else
            rows[n + l], cols[n + l] = c, r
            n += 1
        end
    end
    return resize!.((rows, cols), l + n - 1)
end

function MatrixMarket.mmread(
    ::Type{<:CoordinateStore{Tv,Ti}}, filename; desymmetrize=true
) where {Tv,Ti}
    result = _mmread(filename, Tv, Ti)
    if result isa AbstractArray
        convert(CoordinateStore{Tv,Ti}, result)
    else # TODO: need to take care of symm here somehow. Or handle higher up.
        rows, cols, values, nrows, ncols, nstored, rep, field, symm = result
        rows, cols, values =
            desymmetrize ? desymmetrizer(rows, cols, values, symm) : (rows, cols, values)
        CoordinateStore(
            (rows, cols),
            values isa AbstractArray ? values : Tv[],
            (nrows, ncols);
            isuniform=values isa Tv,
            uniformv=values isa Tv ? values : defaultfill(Tv),
            issorted=false,
            iscoalesced=true,
        )
    end
end

function MatrixMarket.mmread(
    T::Type{<:SinglyCompressedStore{Tv,<:Any,Ti}}, filename; desymmetrize=true
) where {Tv,Ti}
    coo = mmread(CoordinateStore{Tv,Ti}, filename; desymmetrize)
    return convert(T, coo)
end
end
