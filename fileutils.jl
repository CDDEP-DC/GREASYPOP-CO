#=
Copyright (c) 2023 Alexander Tulchinsky

This file is part of Greasypop. Use of this source code is governed by The Clear BSD License.

SPDX-License-Identifier: BSD-3-Clause-Clear
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
