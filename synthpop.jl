#=
Copyright (c) 2023 Alexander Tulchinsky

This file is part of Greasypop. Use of this source code is governed by The Clear BSD License.

SPDX-License-Identifier: BSD-3-Clause-Clear
=#

include("households.jl")
include("schools.jl")
include("workplaces.jl")
include("netw.jl")

println("creating people, households, and group quarters")
generate_people()
println("creating schools")
generate_schools()
println("creating workplaces")
generate_commute_matrices()
generate_jobs_and_workers()
println("creating network")
generate_networks()
generate_location_matrices()
println("done")
