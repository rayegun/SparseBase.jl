module NoValues
import Base:
    !,
    ~,
    +,
    -,
    *,
    &,
    |,
    xor,
    zero,
    one,
    oneunit,
    isfinite,
    isinf,
    isodd,
    isinteger,
    isreal,
    isnan,
    iszero,
    transpose,
    adjoint,
    float,
    complex,
    conj,
    abs,
    abs2,
    iseven,
    ispow2,
    real,
    imag,
    sign,
    inv,
    /,
    ^,
    mod,
    rem,
    min,
    max

export NoValue, novalue, isnovalue, NoValueException, rmnovaluetype

"""
    NoValue

A type with no fields whose singleton instance [`novalue`](@ref) is used
to represent implicit values in sparse arrays. The value this type takes
is determined by the operation, but typically corresponds to the identity.
"""
struct NoValue end

"""
    novalue

The singleton instance of type [`NoValue`](@ref) representing a non-stored value.
"""
const novalue = NoValue()

"""
    isnovalue(x)

Indicate whether `x` is [`novalue`](@ref).
"""
isnovalue(x) = x === novalue

show(io::IO, x::NoValue) = print(io, "novalue")

"""
    NoValueException(msg)

Exception thrown when a [`novalue`](@ref) value is encountered in a situation
where it is not supported. The error message, in the `msg` field
may provide more specific details.
"""
struct NoValueException <: Exception
    msg::String
end

showerror(io::IO, ex::NoValueException) = print(io, "NoValueException: ", ex.msg)

"""
    rmnovaluetype(T::Type)

If `T` is a union of types containing `NoValue`, return a new type with
`NoValue` removed.

# Examples
```jldoctest
julia> rmnovaluetype(Union{Int64,NoValue})
Int64

julia> rmnovaluetype(Any)
Any
```
"""
rmnovaluetype(::Type{T}) where {T} = typesplit(T, NoValue)

function rmnovaluetype_checked(T::Type)
    R = rmnovaluetype(T)
    R >: T && error("could not compute non-novalue type")
    return R
end

promote_rule(::Type{NoValue}, S::Type) = S
promote_rule(T::Type{Union{Nothing, NoValue}}, S::Type) = Union{S, Nothing, NoValue}
# function promote_rule(T::Type{>:Union{Nothing, NoValue}}, S::Type)
#     R = nonnothingtype(T)
#     R >: T && return Any
#     T = R
#     R = nonmissingtype(T)
#     R >: T && return Any
#     T = R
#     R = promote_type(T, S)
#     return Union{R, Nothing, NoValue}
# end
# function promote_rule(T::Type{>:NoValue}, S::Type)
#     R = nonmissingtype(T)
#     R >: T && return Any
#     T = R
#     R = promote_type(T, S)
#     return Union{R, NoValue}
# end

convert(::Type{T}, x::T) where {T>:NoValue} = x
convert(::Type{T}, x::T) where {T>:Union{NoValue, Nothing}} = x
convert(::Type{T}, x) where {T>:NoValue} = convert(rmnovaluetype_checked(T), x)
convert(::Type{T}, x) where {T>:Union{NoValue, Nothing}} = convert(rmnovaluetype_checked(nonnothingtype_checked(T)), x)

function convert(::Type{T}, ::NoValue) where {T}
    return ArgumentError(
        "Cannot convert NoValue to $T. Likely missing a setfill somewhere!"
    )
end

# What's the identity of a comparison operator???
# I'll have to think on the graph implications of this.
# and when you want to use them...
# # Comparison operators
# ==(::NoValue, ::Any) = missing
# ==(::Any, ::NoValue) = missing
# # To fix ambiguity
# ==(::NoValue, ::WeakRef) = missing
# ==(::WeakRef, ::NoValue) = missing
# isequal(::NoValue, ::NoValue) = true
# isequal(::NoValue, ::Any) = false
# isequal(::Any, ::NoValue) = false
# <(::NoValue, ::NoValue) = missing
# <(::NoValue, ::Any) = missing
# <(::Any, ::NoValue) = missing
# isless(::NoValue, ::NoValue) = false
# isless(::NoValue, ::Any) = false
# isless(::Any, ::NoValue) = true
# isapprox(::NoValue, ::NoValue; kwargs...) = missing
# isapprox(::NoValue, ::Any; kwargs...) = missing
# isapprox(::Any, ::NoValue; kwargs...) = missing

# Unary operators/functions
for f in (
    :(!),
    :(~),
    :(+),
    :(-),
    :(*),
    :(&),
    :(|),
    :(xor),
    :(zero),
    :(one),
    :(oneunit),
    :(isfinite),
    :(isinf),
    :(isodd),
    :(isinteger),
    :(isreal),
    :(isnan),
    :(iszero),
    :(transpose),
    :(adjoint),
    :(float),
    :(complex),
    :(conj),
    :(abs),
    :(abs2),
    :(iseven),
    :(ispow2),
    :(real),
    :(imag),
    :(sign),
    :(inv),
)
    @eval ($f)(::NoValue) = novalue
end
for f in (:(Base.zero), :(Base.one), :(Base.oneunit))
    @eval ($f)(::Type{NoValue}) = novalue
    @eval function $(f)(::Type{Union{T,NoValue}}) where {T}
        T === Any && throw(MethodError($f, (Any,)))  # To prevent StackOverflowError
        return $f(T)
    end
end
for f in (:(Base.float), :(Base.complex))
    @eval $f(::Type{NoValue}) = NoValue
    @eval function $f(::Type{Union{T,NoValue}}) where {T}
        T === Any && throw(MethodError($f, (Any,)))  # To prevent StackOverflowError
        return Union{$f(T),NoValue}
    end
end

# Binary operators/functions
for f in (:(+), :(-), :(*), :(/), :(^), :(mod), :(rem), :(min), :(max))
    @eval begin
        ($f)(::NoValue, ::NoValue) = novalue
    end
end
end
