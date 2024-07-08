#=
Copyright 2023 Alexander Tulchinsky

This file is part of Greasypop.

Greasypop is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Greasypop is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with Greasypop. If not, see <https://www.gnu.org/licenses/>.
=#

using CSV
using DataFrames
using SparseArrays
using Serialization
using JSON

function tryJSON(f::AbstractString)::Dict{String,Any}
    try
        return JSON.parsefile(abspath(f))
    catch e
        return Dict{String,Any}()
    end
end

function dser_path(f::AbstractString)
    #println("reading ", f)
    return deserialize(abspath(f))
end

function ser_path(f::AbstractString,obj::Any)
    #println("writing ", f)
    serialize(abspath(f), obj)
    return nothing
end

function read_df(f::AbstractString; kwargs...)
    return CSV.read(abspath(f), DataFrame; kwargs...)
end

function write_df(f::AbstractString, df; kwargs...)
    CSV.write(abspath(f), df; kwargs...)
end

## creates a sparse dataframe from a sparse matrix
spDataFrame(m::SparseMatrixCSC, labels::Union{Vector,Symbol}=:auto) = DataFrame(collect(findnz(m)), labels)

function write_df(f::AbstractString, m::SparseMatrixCSC, labels::Union{Vector,Symbol}=:auto; kwargs...)
    CSV.write(abspath(f), spDataFrame(m, labels); kwargs...)
end
