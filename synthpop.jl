#=
Copyright 2023 Alexander Tulchinsky

This file is part of Greasypop.

Greasypop is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Greasypop is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with Greasypop. If not, see <https://www.gnu.org/licenses/>.
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
