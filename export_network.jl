#=
Copyright (c) 2023 Alexander Tulchinsky

This file is part of Greasypop. Use of this source code is governed by The Clear BSD License.

SPDX-License-Identifier: BSD-3-Clause-Clear
=#

using SparseArrays
include("fileutils.jl")

## Matrix Market native exchange format for sparse matrices
##  https://math.nist.gov/MatrixMarket/formats.html#MMformat
using MatrixMarket

mkpath("pop_export")

println("household adjacency matrix")

m = dser_path("jlse/adj_mat_hh.jlse")
MatrixMarket.mmwrite(abspath("pop_export/adj_upper_triang_hh.mtx"), m)

d = nothing
GC.gc()

println("non-household adjacency matrix")

m = dser_path("jlse/adj_mat_non_hh.jlse")
MatrixMarket.mmwrite(abspath("pop_export/adj_upper_triang_non_hh.mtx"), m)

d = nothing
GC.gc()

println("workplace only adjacency matrix")

m = dser_path("jlse/adj_mat_wp.jlse")
MatrixMarket.mmwrite(abspath("pop_export/adj_upper_triang_wp.mtx"), m)

d = nothing
GC.gc()

println("school only adjacency matrix")

m = dser_path("jlse/adj_mat_sch.jlse")
MatrixMarket.mmwrite(abspath("pop_export/adj_upper_triang_sch.mtx"), m)

d = nothing
GC.gc()

println("groups quarters only adjacency matrix")

m = dser_path("jlse/adj_mat_gq.jlse")
MatrixMarket.mmwrite(abspath("pop_export/adj_upper_triang_gq.mtx"), m)

d = nothing
GC.gc()

println("index keys")

d = dser_path("jlse/adj_mat_keys.jlse")
df = DataFrame([(index_one=i, index_zero=i-1, p_id=Int(v[1]), hh_id=Int(v[2]), cbg_id=Int(v[3])) for (i,v) in enumerate(d)])
write_df("pop_export/adj_mat_keys.csv",df)

## indices of people commuting fron outside the synth pop area
##  these have no household connections
d = dser_path("jlse/adj_dummy_keys.jlse")
df = DataFrame(sort([
            (index_one=Int64(k), index_zero=Int64(k)-1, p_id=Int(v[1]), hh_id=Int(v[2]), cbg_id=Int(v[3]))
            for (k,v) in d
            ], by=x->x.index_one))
write_df("pop_export/adj_dummy_keys.csv",df)

## indices of people working outside the synth pop area
##  these have no workplace connections
d = dser_path("jlse/adj_out_workers.jlse")
df = DataFrame(sort([
            (index_one=Int64(k), index_zero=Int64(k)-1, p_id=Int(v[1]), hh_id=Int(v[2]), cbg_id=Int(v[3]))
            for (k,v) in d
            ], by=x->x.index_one))
write_df("pop_export/adj_out_workers.csv",df)

println("done")

