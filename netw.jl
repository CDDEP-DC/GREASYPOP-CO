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

include("utils.jl")
include("fileutils.jl")

## neighbors for each vertex in graph g
function neighVec(g)
    return [neighbors(g,x) for x in vertices(g)]
end

function read_gq_ppl()
    ## gq residents
    gqs = dser_path("jlse/gqs.jlse")
    gq_res = Dict(k => v.residents for (k,v) in gqs)
    ## gq employees
    gq_workers = dser_path("jlse/gq_workers.jlse")
    return vecmerge(gq_res, gq_workers)
end

function read_sch_ppl()
    ## school students
    sch_students = dser_path("jlse/sch_students.jlse")
    ## teachers
    sch_workers = dser_path("jlse/sch_workers.jlse")
    return vecmerge(sch_students,sch_workers)
end

function read_hh_ppl()
    hh = dser_path("jlse/hh.jlse")
    return Dict(k => v.people for (k,v) in hh)
end

## returns a list of all groups of people in "institutions" 
##   (excluding households, treating those separately)
function read_groups()
    ## gq residents and employees
    gq_ppl = read_gq_ppl()
    ## school students and teachers
    sch_ppl = read_sch_ppl()
    ## from workplaces.jl:
    company_workers = dser_path("jlse/company_workers.jlse")

    return [collect(values(gq_ppl)); 
            collect(values(sch_ppl));
            collect(values(company_workers))]
end

## "dummies" = people commuting from outside the synth area, have no household or location info
## (could get workplaces.jl to save these as they're generated)
function get_dummies()
    sch_workers = dser_path("jlse/sch_workers.jlse")
    gq_workers = dser_path("jlse/gq_workers.jlse")
    company_workers = dser_path("jlse/company_workers.jlse")
    dummies_sch = reduce(vcat, [filter(x -> x[3]==0, v) for v in values(sch_workers)])
    dummies_gq = reduce(vcat, [filter(x -> x[3]==0, v) for v in values(gq_workers)])
    dummies_work = reduce(vcat, [filter(x -> x[3]==0, v) for v in values(company_workers)])
    return [dummies_sch; dummies_gq; dummies_work]
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

## sparse adjacency matrix of bits should be an efficient way to store the network
## (also, simple to convert to Graphs.jl graph for analysis)
## (also, has good lookup performance, same as hash table)
function generate_sparse()
    ## people grouped by workplace, school, group quarters
    ##  (currently, we don't care what kind of group it is)
    ppl_in_groups = read_groups()
    ## each person needs an integer index
    p_idxs, dummy_idxs = person_indices()
    ## save dummies to file; these ppl have workplace but no household; sim should infect them randomly at home
    ser_path("jlse/adj_dummy_keys.jlse", Dict(v=>k for (k,v) in dummy_idxs))
    ## merge people and dummies for network
    merge!(p_idxs, dummy_idxs)

    ## generate network
    src_idxs = Int64[]
    dst_idxs = Int64[]
    
    ## small-world netw params: k = 8, B = 0.25 seems ok
    dConfig = tryJSON("config.json")
    K = get(dConfig, "netw_K", 8)
    B = get(dConfig, "netw_B", 0.25)
    ## if less than k+2, assume fully connected
    min_N = K + 2

    for keyvec in ppl_in_groups
        keyvec = unique(keyvec) ## e.g., if someone lives and works at the same gq
        n = length(keyvec) ## size of group
        if n > 1 ## otherwise, nothing to do
            if n < min_N ## more than one but less than thresh = fully connected
                g = complete_graph(n)
            else
                g = watts_strogatz(n, K, B) ## small-world network
            end
            ## convert indices in g to global person indices for each edge
            for x in edges(g)
                push!(src_idxs, p_idxs[keyvec[x.src]])
                push!(dst_idxs, p_idxs[keyvec[x.dst]])
            end
        end    
    end

    ## create sparse matrix from indices
    adj_mat = sparse([src_idxs;dst_idxs],[dst_idxs;src_idxs],
                        trues(length(src_idxs)+length(dst_idxs)),
                        length(p_idxs),length(p_idxs)) ## ensure size is p_idxs by p_idxs

    ## matrix must be symmetrical, save space by only storing half
    ser_path("jlse/adj_mat.jlse",sparse(UpperTriangular(adj_mat)))

    ## will need the index keys to look up people
    ser_path("jlse/adj_mat_keys.jlse", first.(sort(collect(p_idxs),by=p->p[2])))

    ## track household membership separately; assume they're fully connected
    ##  (and maybe have a higher transmission rate within)
    hh_ppl = read_hh_ppl()
    ser_path("jlse/hh_ppl.jlse", hh_ppl)

    ## keep track of people working outside synth area (have no workplace network, sim should infect them randomly at work)
    outside_workers = dser_path("jlse/outside_workers.jlse")
    ser_path("jlse/adj_out_workers.jlse", Dict(p_idxs[only(x)] => only(x) for x in values(outside_workers)))
    
   return nothing
end

function rev_mat_keys()
    k = dser_path("jlse/adj_mat_keys.jlse")
    return Dict(k .=> eachindex(k))
end

## create adjacency matrix for people in households
## (using same indices created in generate_sparse() above)
function gen_hh_sparse()
    ppl_in_hhs = values(dser_path("jlse/hh_ppl.jlse"))
    p_idxs = rev_mat_keys()
    src_idxs = Int64[]
    dst_idxs = Int64[]

    for keyvec in ppl_in_hhs
        keyvec = unique(keyvec) ## no one should be listed twice in a hh
        n = length(keyvec) ## size of group
        if n > 1 ## otherwise, nothing to do
            ##hhs are fully connected
            g = complete_graph(n)
            for x in edges(g)
                push!(src_idxs, p_idxs[keyvec[x.src]])
                push!(dst_idxs, p_idxs[keyvec[x.dst]])
            end
        end
    end

    ## create sparse matrix from indices
    adj_mat = sparse([src_idxs;dst_idxs],[dst_idxs;src_idxs],
                        trues(length(src_idxs)+length(dst_idxs)),
                        length(p_idxs),length(p_idxs)) ## force size to be same as overall adj matrix

    ## matrix must be symmetrical, save space by only storing half
    ser_path("jlse/hh_adj_mat.jlse",sparse(UpperTriangular(adj_mat)))

    return nothing
end




##
## fns below not used, keep for testing purposes
##

## generate network as adjacency list (vector of neighbors for each node)
function generate_network()
    ## people grouped by workplace, school, group quarters
    ##  (currently, we don't care what kind of group it is)
    ## note, workers who live outside synth area have no cbg (or household)
    ##       -- these become infected randomly at home
    ppl_in_groups = read_groups()

    ## small-world netw params: k = 8, B = 0.25 seems ok
    K = 8
    B = 0.25
    ## if less than k+2, assume fully connected
    min_N = K + 2

    ## network is every person's id associated with a vector of their neighbors' ids
    netw = Dict{Pkey, Vector{Pkey}}()

    for keyvec in ppl_in_groups
        keyvec = unique(keyvec) ## e.g., if someone lives and works at the same gq
        n = length(keyvec) ## size of group
        if n > 1 ## otherwise, nothing to do
            if n < min_N ## more than one but less than thresh = fully connected
                g = complete_graph(n)
            else
                g = watts_strogatz(n, K, B) ## small-world network
            end
            ## neighbors in g are vector indices
            ## associate each person's id with a vector of their neighbors' ids
            d = Dict(keyvec .=> [keyvec[idxs] for idxs in neighVec(g)])
            ## combine with network so far
            vecmerge!(netw, d)
        end    
    end

    ser_path("jlse/netw.jlse",netw)

    ## track household membership separately; assume they're fully connected
    ##  (and maybe have a higher transmission rate within)
    #hh_ppl = read_hh_ppl()
    #ser_path("jlse/hh_ppl.jlse", hh_ppl)

    ## keep track of people working outside synth area
    ##   (these become infected randomly at work)
    #outside_workers = dser_path("jlse/outside_workers.jlse")

   return nothing
end

## generate adjacency list from household membership
function netw_from_hhs(hppl::Dict{Hkey,Vector{Pkey}})
    hnet = Dict{Pkey,Vector{Pkey}}()
    for v::Vector{Pkey} in values(hppl)
        if length(v) > 1
            for i in eachindex(v)
                hnet[v[i]] = v[Not(i)]
            end
        end
    end
    return hnet
end

function merge_hh_net()
    n = dser_path("jlse/netw.jlse")
    hppl = dser_path("jlse/hh_ppl.jlse")
    return vecmerge(n, netw_from_hhs(hppl))
end

function sparse_from_adjdict(d::Dict{Pkey,Vector{Pkey}})
    ## use only the keys in the dict
    ## figure out who's not in the graph later
    p_idxs = Dict(keys(d) .=> 1:length(keys(d)))
    src_idxs = Int64[]
    dst_idxs = Int64[]
    for (k::Pkey,v::Vector{Pkey}) in d
        for p::Pkey in v
            push!(src_idxs, p_idxs[k])
            push!(dst_idxs, p_idxs[p])
        end
    end
    ## create sparse matrix from indices
    adj_mat = sparse([src_idxs;dst_idxs],[dst_idxs;src_idxs],
        trues(length(src_idxs)+length(dst_idxs)),
        length(p_idxs),length(p_idxs)) ## ensure size is p_idxs by p_idxs
    ## matrix indices to person keys
    idx_keys = first.(sort(collect(p_idxs),by=p->p[2]))

    return adj_mat, idx_keys
end




