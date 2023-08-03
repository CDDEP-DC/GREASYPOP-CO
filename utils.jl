#=
Copyright 2023 Alexander Tulchinsky

This file is part of Greasypop.

Greasypop is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Greasypop is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with Greasypop. If not, see <https://www.gnu.org/licenses/>.
=#

using StatsBase
using InlineStrings

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
function ranges(vec::Vector{Int64})
    x = cumsum(vec)
    return [a:b for (a,b) in zip([1;x[1:end-1].+1], x)]
end

## returns first nonempty member of v
function first_nonempty(v) 
    i  = findfirst(!isempty, v)
    isnothing(i) ? empty(v) : v[i]
end

## random lognormal
rlogn(mu::T, sigma::T) where T<:Real = exp(mu + sigma*randn())

## round a vector to integers while preserving sum
## (using largest-remainder method)
function lrRound(v::Vector{T}) where T<:Real
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
function lrRound(v::Matrix{T}) where T<:Real
    orig_dims = size(v)
    vrnd = lrRound(vec(v))
    return reshape(vrnd, orig_dims)
end

## sample from a vector of counts; returns an index and depletes the counts
##    "AbstractArray" means this also works on 1D _views_ of a matrix
function drawCounts!(v::AbstractArray{Int64})
    i = wsample(eachindex(v),v) ## wsample() from StatsBase
    v[i] -= 1 ## modify v
    return i
end

## sample n from a vec of counts; returns vec of indices and depletes counts
function drawCounts!(v::AbstractArray{Int64}, n::Int64)
    ## should probably throw an error if n > sum(v)
    n = min(n,sum(v))
    res = zeros(Int64, n)
    for i in 1:n
        res[i] = drawCounts!(v)
    end
    return res
end



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
    female::Union{Missing,Bool}
    working::Bool
    commuter::Bool
    #job_listed::Union{Missing,Bool}
    com_LODES_low::Bool ## commutes to <40k/yr job
    com_LODES_high::Bool ## commutes to >40k/yr job
    sch_grade::Union{Missing,String3}
end

struct Household
    sample::UInt32
    people::Vector{Pkey}
end

struct GQres
    type::Symbol
    residents::Vector{Pkey}
end
