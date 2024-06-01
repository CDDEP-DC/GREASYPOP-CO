#=
Copyright 2023 Alexander Tulchinsky

This file is part of Greasypop.

Greasypop is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Greasypop is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with Greasypop. If not, see <https://www.gnu.org/licenses/>.
=#

using Graphs
#using GraphPlot
using SparseArrays
using LinearAlgebra
using Random
using Distributions

include("utils.jl")
include("fileutils.jl")

## neighbors for each vertex in graph g
function neighVec(g)
    return [neighbors(g,x) for x in vertices(g)]
end

function read_gq_ppl()
    ## gq residents
    gq_res = Dict(k => v.residents for (k,v) in dser_path("jlse/gqs.jlse"))
    ## gq employees; remove extra worker tag
    gq_workers = Dict(k => [x[1:3] for x in v] for (k,v) in dser_path("jlse/gq_workers.jlse"))
    return vecmerge(gq_res, gq_workers)
end

function assign_teachers_to_grades(school_key::String15,students_by_grade::Vector{S},sch_workers::Dict{String15,Vector{Pkey}}) where S<:Any
    ## person ids of teachers in this school
    teachers = sch_workers[school_key]
    ## a school could have been created with teachers but no students
    if isempty(students_by_grade) 
        return [(t...,String3("0")) for t in teachers]
    else
        ## proportionmap from statsbase: p students in each grade in this school
        pstudents = proportionmap([x[4]::String3 for x in students_by_grade])
        ## n teachers by grade, assuming it's exactly proportional to students
        n_tmp = Dict(k=>v*length(teachers) for (k::String3,v::Float64) in pstudents)
        ## round to integers and expand into a vector having the same length as "teachers"
        teacher_grades = reduce(vcat, fill(k,v) for (k::String3,v::Int) in Dict(keys(n_tmp) .=> lrRound(collect(values(n_tmp)))))
        ## append grade to each teacher's person id (teacher order is already random)
        return [(t...,g) for (t::Pkey,g::String3) in zip(teachers,teacher_grades)]
    end
end

function read_sch_ppl()
    ## for looking up the school grade of each student:
    ppl = Dict(k=>v.sch_grade for (k,v) in dser_path("jlse/people.jlse"))
    ## append each student's grade to their person id
    sch_students_x_grade = Dict{String15,Vector{Tuple{fieldtypes(Pkey)...,String3}}}(k=>[(personkey..., ppl[personkey]) 
        for personkey in v] for (k,v) in dser_path("jlse/sch_students.jlse"))
    ## strip income category from school workers, leaving only person id
    sch_workers = Dict(k => [x[1:3] for x in v] for (k,v) in dser_path("jlse/sch_workers.jlse"))
    ## assign a grade to each teacher and append it to their person id
    teachers_by_grade = Dict(k => assign_teachers_to_grades(k,v,sch_workers) for (k,v) in sch_students_x_grade)
    ## return students and teachers by school
    return vecmerge(sch_students_x_grade,teachers_by_grade)
end

function read_hh_ppl()
    hh = dser_path("jlse/hh.jlse")
    return Dict(k => v.people for (k,v) in hh)
end

## "dummies" = people commuting from outside the synth area, have no household or location info
function get_dummies()
    ## remove extra worker tag
    return [x[1:3] for x in dser_path("jlse/work_dummies.jlse")]
end

## integer index for each "real" person in synth pop, and for each "dummy" person
function person_indices()
    ## everyone in households and gq's in the synth area
    people = keys(dser_path("jlse/people.jlse"))
    ## workers without households
    dummies = get_dummies()    

    n = length(people)
    return (
        Dict(people .=> 1:n),
        Dict(dummies .=> (1+n):(n+length(dummies)))
        )
end

## connects the keys in keyvec into a stochastic block model (SBM) network
## - note, the degree distribution within blocks in an SBM is similar to 
##      a random (erdos-renyi) network. For a more realistic degree distribution (e.g., with hubs)
##      try a degree-corrected SBM (Karrer & Newman 2010, https://arxiv.org/abs/1008.3926)
##
## if use_groups is true, the 4th element of a key should be the group identity (the first 3 would be a person key)
## returns a list of (source, dest) edges where source and dest are keys in keyvec
function connect_SBM(keyvec::Vector{T}, K::Int, min_N::Int, assoc_coeff::Float64, use_groups::Bool=true) where T<:Any
    keyvec = unique(keyvec) ## duplicates would result in self-connections
    n = length(keyvec)
    if n < 2 ## nothing to connect
        return Tuple{T,T}[]
    elseif n < min_N ## more than one but less than thresh = fully connected
        g = complete_graph(n)
        ## indices in g correspond to positions in keyvec
        return [(keyvec[x.src],keyvec[x.dst]) for x in edges(g)]
    else
        ## more than min_N = use SBM
        if use_groups
            ## group membership is based on key element #4
            group_labels = unique(k[4] for k in keyvec)
            group_indices = [findall(k->k[4]==g, keyvec) for g in group_labels] 
        else 
            ## otherwise put everyone in one group
            group_indices = [eachindex(keyvec)] 
        end

        len_grp_idx = length.(group_indices) ## number of vertices in each group
        n_vec = filter(x->x>0, len_grp_idx) ## drop any 0-size groups
        n_groups = length(n_vec)
        w_planted = Diagonal(fill(K,n_groups)) ## contact matrix if no group mixing
        prop_i = n_vec ./ sum(n_vec) ## proportion in each group
        w_random = repeat(transpose(prop_i) * K, n_groups) ## contact matrix if random group mixing
        c_matrix = assoc_coeff * w_planted + (1 - assoc_coeff) * w_random  ## linear interpolation between planted and random contact matrices

        ## can't have more than n connections to a group
        for r in eachrow(c_matrix)
            r .= min.(r,n_vec)
        end
        ## can't have more than n-1 within-group connections
        d_tmp = view(c_matrix, diagind(c_matrix))
        d_tmp .= min.(d_tmp, n_vec .- 1)

        ## note, graph gen function takes group sizes in n_vec; the vertices/edges in the
        ##  resulting graph are just numbered based on group sizes, in the order given in n_vec
        ##  to recover keys, need to translate g index -> keyvec index -> key
        g = stochastic_block_model(c_matrix,n_vec)
        keyvec_indices = reduce(vcat, group_indices) ## for translating indices

        ## algo sometimes creates 0-degree vertices; not technically wrong, but force them to have 1 connection anyway
        fix_zeros = findall(degree(g).==0)
        for x in fix_zeros
            add_edge!(g, x, rand(vertices(g)))
        end

        ## convert indices in g to keyvec keys for each edge
        result = Tuple{T,T}[]
        for x in edges(g)
            if use_groups
                (group_labels[findfirst(c->in(x.src,c) , ranges(len_grp_idx))] == keyvec[keyvec_indices[x.src]][4]) &&
                    (group_labels[findfirst(c->in(x.dst,c) , ranges(len_grp_idx))] == keyvec[keyvec_indices[x.dst]][4]) ||
                    throw("group assignment error")
            end
            
            push!(result, (keyvec[keyvec_indices[x.src]],keyvec[keyvec_indices[x.dst]]))
        end
        return result
    end ## if n
end

## as above, but using a simple small-world network
function connect_small_world(keyvec::Vector{T}, K::Int, min_N::Int, B::Float64) where T<:Any
    keyvec = unique(keyvec) ## e.g., duplicates happen if someone lives and works at the same gq
    n = length(keyvec) ## size of group
    if n < 2 ## nothing to connect
        return Tuple{T,T}[]
    elseif n < min_N ## more than one but less than thresh = fully connected
        g = complete_graph(n)
    else
        g = watts_strogatz(n, K, B) ## small-world network
    end
    ## indices in g correspond to positions in keyvec
    return [(keyvec[x.src],keyvec[x.dst]) for x in edges(g)]
end

## as above, but completely connected graph
function connect_complete(keyvec::Vector{T}) where T<:Any
    keyvec = unique(keyvec)
    n = length(keyvec) ## size of group
    if n < 2 ## nothing to connect
        return Tuple{T,T}[]
    else
        g = complete_graph(n)
    end
    ## indices in g correspond to positions in keyvec
    return [(keyvec[x.src],keyvec[x.dst]) for x in edges(g)]
end

## sparse adjacency matrix of bits should be an efficient way to store the network
## (also, simple to convert to Graphs.jl graph for analysis)
## (also, has good lookup performance, same as hash table)
function sp_from_groups(connect_fn, keygroups::Vector{Vector{T}}, p_idxs::Dict{Pkey,Int64}) where T<:Tuple
    src_idxs = Int64[]
    dst_idxs = Int64[]

    for keyvec in keygroups 
        ## connect keys into network, then convert to integer indices
        for (s_key, d_key) in connect_fn(keyvec)
            ## key elements 1-3 correspond to a person key
            push!(src_idxs, p_idxs[s_key[1:3]])
            push!(dst_idxs, p_idxs[d_key[1:3]])
        end
    end

    ## create sparse matrix from indices
    return sparse([src_idxs;dst_idxs],[dst_idxs;src_idxs],
                        trues(length(src_idxs)+length(dst_idxs)),
                        length(p_idxs),length(p_idxs)) ## ensure size is p_idxs by p_idxs
end


##
## generate all networks
##
function generate_networks()

    dConfig = tryJSON("config.json")
    work_K::Int = get(dConfig, "workplace_K", 8) ## mean degree for wp networks
    school_K::Int = get(dConfig, "school_K", 12) ## mean degree for school networks
    gq_K::Int = get(dConfig, "gq_K", 12) ## mean degree for group-quarters networks
    other_K::Int = get(dConfig, "netw_K", 8) ## mean degree for other networks
    sm_world_B::Float64 = get(dConfig, "netw_B", 0.25) ## beta param for small world networks
    work_assoc_coeff::Float64 = get(dConfig, "income_associativity_coefficient", 0.9) ## group associativity for workplace networks
    sch_assoc_coeff::Float64 = get(dConfig, "school_associativity_coefficient", 0.9) ## group associativity for school networks
    
    ## from workplaces.jl; workers grouped into companies
    ## worker is (person id, hh id, cbg id, income category)
    company_workers = collect(values(dser_path("jlse/company_workers.jlse")))
    #school students and teachers
    ## student/teacher is (person id, hh id, cbg id, grade) -- first 3 are a person key
    ppl_in_schools = collect(values(read_sch_ppl()))
    ## gq residents and employees 
    ppl_in_gq = collect(values(read_gq_ppl()))
    
    ## households; assume they're fully connected (and maybe have a higher transmission rate within)
    hh_ppl = read_hh_ppl()
    ## save household membership Dict
    ser_path("jlse/hh_ppl.jlse", hh_ppl)
    ## just groups as vectors
    ppl_in_hhs = collect(values(hh_ppl))
    
    ## each person needs an integer index
    p_idxs, dummy_idxs = person_indices()
    ## save dummies to file; these ppl have workplace but no household; sim should infect them randomly at home
    ser_path("jlse/adj_dummy_keys.jlse", Dict(v=>k for (k,v) in dummy_idxs))
    ## merge people and dummies for network
    merge!(p_idxs, dummy_idxs)
    ## will need the index keys to look up people
    ser_path("jlse/adj_mat_keys.jlse", first.(sort(collect(p_idxs),by=p->p[2])))
    
    ## generate network
    ## workplaces: using stochastic block model (SBM) network
    ## so that all results are comparable, use the SBM algo even if there's only one income group in a wp
    adj_wp = sp_from_groups(v->connect_SBM(v, work_K, work_K+2, work_assoc_coeff, true), company_workers, p_idxs)
    ## schools: using SBM, grouped by grade
    adj_sch = sp_from_groups(v->connect_SBM(v, school_K, school_K+2, sch_assoc_coeff, true), ppl_in_schools, p_idxs)
    ## other institutions (currently just gq's) : using small-world network
    adj_gq = sp_from_groups(v->connect_small_world(v, gq_K, gq_K+2, sm_world_B), ppl_in_gq, p_idxs)
    ## households are fully connected
    adj_hh = sp_from_groups(v->connect_complete(v), ppl_in_hhs, p_idxs)
    ## save combined non-hh netw for when those distinctions are not needed
    adj_mat_non_hh = adj_wp .| adj_sch .| adj_gq
    
    ## matrices must be symmetrical, save space by only storing half
    ser_path("jlse/adj_mat_non_hh.jlse",sparse(UpperTriangular(adj_mat_non_hh)))
    ser_path("jlse/adj_mat_wp.jlse",sparse(UpperTriangular(adj_wp)))
    ser_path("jlse/adj_mat_sch.jlse",sparse(UpperTriangular(adj_sch)))
    ser_path("jlse/adj_mat_gq.jlse",sparse(UpperTriangular(adj_gq)))
    ser_path("jlse/adj_mat_hh.jlse",sparse(UpperTriangular(adj_hh)))
    
    ## keep track of people working outside synth area (have no workplace network, sim should infect them randomly at work)
    outside_workers = dser_path("jlse/outside_workers.jlse")
    ser_path("jlse/adj_out_workers.jlse", Dict(p_idxs[only(x)[1:3]] => only(x)[1:3] for x in values(outside_workers)))
    
    return nothing   
end


##
## generate info needed to simulate ephemeral location-based contacts
##

## assume that the chance of meeting a neighbor is the same as meeting someone who works in your neighborhood
##  (otherwise need another parameter to describe the difference)

## group potential encounters by census tract (CBG seems too restrictive)

## home locations; e.g, 
## cbg A: people 1, 2
## cbg B: 3, 4
##
## work locations; e.g., 
## cbg A: 1, 3
## cbg B: 2
##
## at home, person 1 could meet: 2, 3
##          person 2 could meet: 1, 3
##          person 3 could meet: 4, 2
##
## at work, person 1 could meet: 3, 2
##          person 2 could meet: 3, 4
##          person 3 could meet: 1, 2

## make a separate matrix for home and work neighborhood contacts?
##  (# of contact events could be independent, or maybe work is closed)
## matrix is not symmetrical (local resident you meet while at work probably won't meet you when they're at work)
## "source" person will be column (because reading by columns is faster in julia)

## home neighborhood contacts
##  for each location:
##   for each person who lives there (hh or non-inst gq):
##    look up everyone else who lives there (hh or gq) + append everyone who works there
## work neighborhood contacts
##  for each location:
##   for each person who works there:
##    look up everyone else who works there + append everyone who lives there (hh or non-inst gq)

## save memory by not generating the contact matrix
##  just a matrix of locations (columns) x people (rows)
##  then simple O1 look-up to get a person's home or work location, and pick from that column

function generate_location_matrices()

    w = dser_path("jlse/company_workers.jlse") ## employers/employees (with work locations)
    hh = dser_path("jlse/hh.jlse") ## households/residents (with hh locations)
    cbg_idxs = dser_path("jlse/cbg_idxs.jlse") ## location (cbg) keys used in person/hh keys
    cbg_idxs = Dict(k=>String31(v) for (k,v) in cbg_idxs)
    gqs = dser_path("jlse/gqs.jlse") ## group-quarters/residents (with gq locations)
    ## assume only non-inst GQ residents are available for ephemeral local contacts
    ni_types = Set([:milGQ, :ninst1864civ])
    gq_noninst = filterv(x->(x.type in ni_types), gqs)
    ## use the same matrix indices as in the regular contact networks
    k = dser_path("jlse/adj_mat_keys.jlse")
    p_idxs = Dict(k .=> eachindex(k))

    ## group potential encounters by census tract (CBG seems too restrictive)
    hh_tracts = unique(x[1:end-1] for x in values(cbg_idxs))
    work_tracts = unique(x[3][1:end-1] for x in keys(w))
    tracts = sort(unique([hh_tracts; work_tracts]))

    ## convert tract codes to integer indices for constructing a matrix
    loc_idxs = Dict(tracts .=> eachindex(tracts))
    ## save location indices
    ser_path("jlse/loc_mat_keys.jlse",loc_idxs)

    ## group individuals by tract code (= cbg code minus last character)
    ## dataframe provides fast grouping
    w_df_by_loc = groupby(DataFrame((k[3][1:end-1], v) for (k,v) in w), "1")
    ## place person-vectors in a dict with integer tract indices as the keys
    workers_by_loc = Dict(loc_idxs[loc["1"]]=>reduce(vcat, w_df_by_loc[loc][!,"2"]) for loc in keys(w_df_by_loc))
    ## convert person keys to network matrix indices (same indices as regular contact networks)
    w_idxs_by_loc = Dict(k=>[p_idxs[i[1:3]] for i in v] for (k,v) in workers_by_loc)

    h_df_by_loc = groupby(DataFrame((cbg_idxs[k[2]][1:end-1], v.people) for (k,v) in hh), "1")
    hh_ppl_by_loc = Dict(loc_idxs[loc["1"]]=>reduce(vcat, h_df_by_loc[loc][!,"2"]) for loc in keys(h_df_by_loc))
    hh_idxs_by_loc = Dict(k=>[p_idxs[i] for i in v] for (k,v) in hh_ppl_by_loc)

    gq_df_by_loc = groupby(DataFrame((cbg_idxs[k[2]][1:end-1], v.residents) for (k,v) in gq_noninst), "1")
    gq_ppl_by_loc = Dict(loc_idxs[loc["1"]]=>reduce(vcat, gq_df_by_loc[loc][!,"2"]) for loc in keys(gq_df_by_loc))
    gq_idxs_by_loc = Dict(k=>[p_idxs[i] for i in v] for (k,v) in gq_ppl_by_loc)

    ## residential locations include households and non-inst GQs:
    res_idxs_by_loc = vecmerge(hh_idxs_by_loc, gq_idxs_by_loc)

    ## construct matrices for use in simulation
    ## columns are locations, rows are people (because we'll be looking up by location)
    ##  note, currently people have one job max
    w_loc_contact_mat = sparse(
        reduce(vcat, collect(values(w_idxs_by_loc))), 
        reduce(vcat, [fill(k,length(v)) for (k,v) in w_idxs_by_loc]),
        trues(sum(length.(values(w_idxs_by_loc)))),
        length(k),length(tracts)
    )

    res_loc_contact_mat = sparse(
        reduce(vcat, collect(values(res_idxs_by_loc))), 
        reduce(vcat, [fill(k,length(v)) for (k,v) in res_idxs_by_loc]),
        trues(sum(length.(values(res_idxs_by_loc)))),
        length(k),length(tracts)
    )

    ser_path("jlse/work_loc_contact_mat.jlse",w_loc_contact_mat)
    ser_path("jlse/res_loc_contact_mat.jlse",res_loc_contact_mat)

    ## save work and home loc idx for each person idx, for fast lookup
    ##  note, currently people have one job max
    w_loc_by_p_idx = Dict(reduce(vcat, [v .=> k for (k,v) in w_idxs_by_loc]))
    res_loc_by_p_idx = Dict(reduce(vcat, [v .=> k for (k,v) in res_idxs_by_loc]))

    ser_path("jlse/work_loc_lookup.jlse",w_loc_by_p_idx)
    ser_path("jlse/res_loc_lookup.jlse",res_loc_by_p_idx)

    return nothing
end



