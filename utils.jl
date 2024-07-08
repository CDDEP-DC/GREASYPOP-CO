#=
Copyright 2023 Alexander Tulchinsky

This file is part of Greasypop.

Greasypop is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Greasypop is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with Greasypop. If not, see <https://www.gnu.org/licenses/>.
=#

using StatsBase
using InlineStrings
using SparseArrays

## may change these
const CBGkey = UInt16
const Hnum = UInt16
const Pnum = UInt32
const Hkey = Tuple{Hnum,CBGkey}
const Pkey = Tuple{Pnum,Hnum,CBGkey}
const GQkey = Tuple{UInt16,CBGkey}
const WRKkey = Tuple{UInt32, UInt8, String31}

struct PersonData
    hh::Hkey
    sample::UInt32
    age::Int16
    working::Bool
    commuter::Bool
    com_cat::Union{Missing,UInt8} ## wp category (industry, etc) for placing commuters in wp's 
    com_inc::Union{Missing,UInt8} ## income category for commuters, to make wp contact networks
    sch_grade::Union{Missing,String3}
    #more_traits::Dict{Symbol,Union{Missing,Bool}}
    ## ^^ that would be nice, but several million dicts will use a lot of memory, so...
    ## (these have to be in the same order as "additional traits" in config.json)
    sch_public::Union{Missing,Bool}
    sch_private::Union{Missing,Bool}
    female::Union{Missing,Bool}
    race_black_alone::Union{Missing,Bool}
    white_non_hispanic::Union{Missing,Bool}
    hispanic::Union{Missing,Bool}
end

struct Household
    sample::UInt32
    people::Vector{Pkey}
end

struct GQres
    type::Symbol
    residents::Vector{Pkey}
end

## make dicts callable, with a closure to handle missing keys
(d::AbstractDict)(x,f::Base.Callable) = get(f,d,x)

## subset of dict by keys
dsubset(d::Dict{K,V},s) where {K<:Any,V<:Any} = Dict{K,V}(k=>d[k] for k in intersect(keys(d),s))

mutable struct Indexer{I}
    i::I
end
## constructor
Indexer{T}() where {T<:Number} = Indexer(zero(T))
## an indexer object functions like a closure
## when called with a dict and key, returns that key's index,
## inserts the key with a new index if it doesn't exist
function (ix::Indexer{I})(d::Dict{K,I}, k::K) where {I<:Number, K<:Any}
    if haskey(d,k)
        return d[k]
    else
        ix.i += 1
        d[k] = ix.i
        return ix.i
    end
end

## convert to integer, missing becomes 0
int(x::T) where T<:Real = round(Int64, x)
int(x::Missing) = Int64(0)

## convert x to type T, missing stays missing
mcon(::Type{T}, x::Missing) where {T<:Any} = missing
mcon(::Type{T}, x::U) where {T,U} = convert(T,x)::T

## convert missing bool to false
mtrue(x::Union{Missing,Bool}) = coalesce(x,false)

## filter dict on values
filterv(f, d::Dict) = filter( ((k,v),) -> f(v) , d)

## filter dict on keys
filterk(f, d::Dict) = filter( ((k,v),) -> f(k) , d)

## merge two dictionaries with vector values
vecmerge = mergewith(vcat)
## as above, but modifies the first dict in-place
vecmerge! = mergewith!(vcat)

## "flattens" a dictionary whose values are vectors/collections
## returns a vector of pairs
dflat(d::Dict) = collect(Iterators.flatmap(x->((x.first => y) for y in x.second), d))

## continuous index ranges with lengths given by vec
function ranges(vec::Vector{I}) where {I<:Integer}
    x = cumsum(vec)
    return [a:b for (a,b) in zip([1;x[1:end-1].+1], x)]
end

## returns first nonempty member of v
function first_nonempty(v) 
    i = findfirst(!isempty, v)
    isnothing(i) ? empty(v) : v[i]
end

## index of first true in v, otherwise missing
first_true(v) = something(findfirst(v),missing)

## replace values less than threshhold with 0
thresh(x,v) = x < v ? zero(x) : x

## random lognormal
rlogn(mu::T, sigma::T) where T<:Real = exp(mu + sigma*randn())

## round a vector to integers while preserving sum
## (using largest-remainder method)
function lrRound(v::AbstractVector{T}) where T<:Real
    vrnd = floor.(Int64, v)
    verr = v .- vrnd
    vrem = round(Int64, sum(v) - sum(vrnd))
    vidxs = sortperm(verr, rev=true)
    for i in 1:vrem
        vrnd[vidxs[i]] += 1    
    end
    return vrnd    
end

## make it work with matrices too
function lrRound(v::AbstractMatrix{T}) where T<:Real
    orig_dims = size(v)
    vrnd = lrRound(vec(v))
    return reshape(vrnd, orig_dims)
end

## but round each row
function rowRound(m::AbstractMatrix{T}) where T<:Real
    res = zeros(Int,size(m))
    for i in axes(m,1)
        res[i,:] = lrRound(m[i,:])
    end
    return res
end

## but round each column
function colRound(m::AbstractMatrix{T}) where T<:Real
    res = zeros(Int,size(m))
    for i in axes(m,2)
        res[:,i] = lrRound(m[:,i])
    end
    return res
end

## sample from a vector of counts; returns an index and depletes the counts
##    "AbstractArray" means this also works on 1D _views_ of a matrix
function drawCounts!(v::AbstractArray{I}) where {I<:Integer}
    i = wsample(eachindex(v),v) ## wsample() from StatsBase
    v[i] -= 1 ## modify v
    return i
end

## sample n from a vec of counts; returns vec of indices and depletes counts
function drawCounts!(v::AbstractArray{Ia}, n::Ib) where {Ia<:Integer,Ib<:Integer}
    ## should probably throw an error if n > sum(v)
    n = min(n,sum(v))
    res = zeros(Int64, n)
    for i in 1:n
        res[i] = drawCounts!(v)
    end
    return res
end
