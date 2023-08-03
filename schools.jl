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

    for (k,gr) in zip(String3.(["k";string.(1:12)]), ["KG";string.(1:12)])
        mask = schools[!,"G_"*gr*"_OFFERED"]
        d = Dict{String15,Vector{String15}}()
        ## the keys of distmat are geo groups
        for geo in keys(distmat)
            avail = @view distmat[geo][mask,:]
            d[geo[1]] = String15.(first(sort(avail, :value).variable, n))
        end
        closest[k] = d
    end
    ## assume kindergartens also offer preschool
    ## (we haven't generated private preschools and we have to send the preschoolers somewhere)
    closest[String3("p")] = closest["k"]
    return closest
end


#function read_closest()
#    closest = Dict{String3,Dict{String15,Vector{String15}}}()
#    for (k,gr) in zip(String3.(["p";"k";string.(1:12)]), ["PK";"KG";string.(1:12)])
#        df = read_df("processed/cbg_sch_"*gr*".csv"; types=Dict("GEOID"=>String15,"NCESSCH"=>String15,"0"=>Float64) )
#        df_by_geo = combine(groupby(df, :GEOID), :NCESSCH => Ref => :closest5)
#        closest[k] = Dict(df_by_geo.GEOID .=> collect.(df_by_geo.closest5))
#    end
#    return closest
#end

function read_p_in_school(cbgs::Dict{CBGkey, String15})
    people = dser_path("jlse/people.jlse")
    ## excluding college and grad school
    p_in_school = filterv(p->(!ismissing(p.sch_grade) && !in(p.sch_grade, ["c","g"])), people)
    ## person key, grade, cbg code
    return [(k, p.sch_grade, cbgs[k[3]]) for (k,p) in p_in_school]
end


## create schools (= string ids), place people into schools
function generate_schools()

    dConfig = tryJSON("config.json")
    n_schools::Int = get(dConfig, "n_closest_schools", 5)
    ## assign students to closest avaiable school (90%) or 2nd closest (10%)
    prob_closest::Float64 = get(dConfig, "p_closest_school", 0.9)

    sch_capacity = read_sch_cap()
    closest = find_closest(n_schools)
    cbgs = dser_path("jlse/cbg_idxs.jlse")
    p_in_school = read_p_in_school(cbgs)

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
        ## if all schools full, overfill the closest
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
