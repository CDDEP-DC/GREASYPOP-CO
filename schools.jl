#=
Copyright 2023 Alexander Tulchinsky

This file is part of Greasypop.

Greasypop is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Greasypop is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with Greasypop. If not, see <https://www.gnu.org/licenses/>.
=#

#using Statistics

include("utils.jl")
include("fileutils.jl")

function read_sch_cap()
    sch_cols = Dict("NCESSCH"=>String15,"TEACHERS"=>Int64,"STUDENTS"=>Int64)
    schools = read_df("processed/schools.csv"; select=collect(keys(sch_cols)), types=sch_cols)
    return Dict(schools.NCESSCH .=> schools.STUDENTS)
end

function find_closest(n::Int)
    ## grades offered by each school
    schools = read_df("processed/schools.csv"; types=Dict("NCESSCH"=>String15))
    ## distance matrix
    distmat = read_df("processed/cbg_sch_distmat.csv"; types=Dict("GEOID"=>String15))
    ## reshape and group by geography
    distmat = groupby(stack(distmat), :GEOID)
    ## for each grade, store a dict of geo=>schools
    closest = Dict{String3,Dict{String15,Vector{String15}}}()
    ## and the distances
    distances = Dict{String3,Dict{String15,Vector{Float64}}}()

    ## assume kindergartens also offer preschool
    ## (we haven't generated private preschools and don't have enough public preschools)
    schools[:,"G_PK_OFFERED"] .= schools[!,"G_PK_OFFERED"] .| schools[!,"G_KG_OFFERED"]

    for (k,gr) in zip(String3.([["p","k"];string.(1:12)]), [["PK","KG"];string.(1:12)])
        mask = schools[!,"G_"*gr*"_OFFERED"]
        sch_by_geo = Dict{String15,Vector{String15}}()
        dist_by_geo = Dict{String15,Vector{Float64}}()
        ## the keys of distmat are geo groups
        for geo in keys(distmat)
            avail = @view distmat[geo][mask,:]
            topidxs = partialsortperm(avail.value, 1:n)
            sch_by_geo[geo[1]] = String15.(avail.variable[topidxs])
            dist_by_geo[geo[1]] = avail.value[topidxs]
        end
        closest[k] = sch_by_geo
        distances[k] = dist_by_geo
    end

    return closest, distances
end

function read_p_in_school(cbgs::Dict{CBGkey, String15})
    people = dser_path("jlse/people.jlse")
    ## excluding college and grad school
    p_in_school = filterv(p->(!ismissing(p.sch_grade) && !in(p.sch_grade, ["c","g"])), people)

    ## sort by cbg,household so we can make kids in the same household go to the same school
    pkeys = collect(keys(p_in_school))
    idxs = sortperm([x[[3,2]] for x in pkeys])
    pkeys = pkeys[idxs]

    ## person key, grade, cbg code
    return [(k, (p_in_school[k]).sch_grade, cbgs[k[3]]) for k in pkeys]
end


## can't just draw school-wise from closest
## how to explain existence of schools that aren't closest (or 2 closest) for anyone
## model school-choosing behavior??
## -- p depends on distance (p ~ 1/dist) and # remaining spots (p ~ spots_left)
## -- only need to consider n closest
## ...?

## send to closest (or 2 closest) first; people who live together more likely to go to same school
## create schools (= string ids), place people into schools
function generate_schools()

    dConfig = tryJSON("config.json")
    n_schools::Int = get(dConfig, "n_closest_schools", 4)
    ## assign students to closest avaiable school (90%) or 2nd closest (10%)
    prob_closest::Float64 = get(dConfig, "p_closest_school", 0.9)

    closest, _ = find_closest(n_schools)
    cbgs = dser_path("jlse/cbg_idxs.jlse")
    p_in_school = read_p_in_school(cbgs)

    sch_capacity = read_sch_cap()
    ## shrink school capacities to prevent underfilling non-closest schools
    sch_capacity = Dict(k=>round(Int,v*0.8) for (k,v) in sch_capacity)

    ## school code => vector of person keys
    sch_students = Dict{String15,Vector{Pkey}}()
    ## initialize with empty vectors
    for k in keys(sch_capacity)
        sch_students[k] = Vector{Pkey}()
    end

    ## read person key, grade level, and cbg code for each person
    for (pk,gr,geo) in p_in_school
        ## look up closest schools for grade level and cbg
        opts = closest[gr][geo]
        ## find the first that hasn't been filled to capacity
        idx_avail = findfirst(k->sch_capacity[k]>length(sch_students[k]), opts)
        ## if all schools full, try again with increased capacity
        idx_avail = isnothing(idx_avail) ? findfirst(k->1.5*sch_capacity[k]>length(sch_students[k]), opts) : idx_avail
        ## if all schools full, try again with increased capacity
        idx_avail = isnothing(idx_avail) ? findfirst(k->2.5*sch_capacity[k]>length(sch_students[k]), opts) : idx_avail
        ## if still full, just overfill the closest
        idx_avail = isnothing(idx_avail) ? 1 : idx_avail
        ## choose closest or next closest
        idx_choice = rand() < prob_closest ? idx_avail : idx_avail+1
        idx_choice = idx_choice > lastindex(opts) ? 1 : idx_choice
        ## append student to school
        push!(sch_students[opts[idx_choice]], pk)
    end

    ## save to file for next step
    ser_path("jlse/sch_students.jlse", sch_students)
    return nothing
end



