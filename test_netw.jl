#=
Copyright 2023 Alexander Tulchinsky

This file is part of Greasypop.

Greasypop is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Greasypop is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with Greasypop. If not, see <https://www.gnu.org/licenses/>.
=#

#using Distributed
#using Random
using Graphs
#using GraphPlot
using SparseArrays
using LinearAlgebra
#using SharedArrays
#using Distributions
#using Plots
using StatsBase

using MatrixDepot
#using SNAPDatasets

include("utils.jl")
include("fileutils.jl")

## from Bonchev 2004
function Bonchev_vd_info0(t::AbstractGraph{T}) where T
    v = filter(x->x>0,degree(t))
    A = sum(v)
    return (sum( v .* log2.(v) )) / ( A * log2(A) )
end

function Bonchev_vd_info(t::AbstractGraph{T}) where T
    v = filter(x->x>0,degree(t))
    A = sum(v)
    return sum(  v .* log2.(v) ./ (A * log2(A)) ) 
end

## from Saberi 2021
makehub(t::AbstractGraph{T}) where T = sum(degree(t) .^ 2) / sum(degree(t))





#g = watts_strogatz(100,8,0.25)
#gplot(g)
## simulate infection at p = ?
#diffusion_rate(g, 0.5, 10)
## mean path length
#f = floyd_warshall_shortest_paths(g)
#sum(f.dists) / length(f.dists)

## neighbors
#g = watts_strogatz(20,6,0.3)
#[neighbors(g,x) for x in vertices(g)]


#include("netw.jl")

#generate_network()

#full_mat, full_keys = sparse_from_adjdict(merge_hh_net());
#ser_path("jlse/full_graph_mat.jlse", sparse(UpperTriangular(full_mat)))
#ser_path("jlse/full_graph_keys.jlse", full_keys)


## note this method adds zero-degree vertices if adj_mat has 0-sum rows
#full_graph = SimpleGraph(sparse(Symmetric(dser_path("jlse/full_graph_mat.jlse"))))
#ser_path("jlse/full_graph.jlse",full_graph)

#full_graph = dser_path("jlse/full_graph.jlse")
#node_keys = dser_path("jlse/full_graph_keys.jlse")



f(v) = mean(v[2:end] ./ v[1:end-1])


net_comp = Dict()

m = matrixdepot("SNAP/loc-Gowalla")
t = SimpleGraph(m)
net_comp["loc_gowalla"] = [mean(local_clustering_coefficient(t))
global_clustering_coefficient(t)
mean(pagerank(t))
rich_club(t,1)
assortativity(t)
Bonchev_vd_info(t)
makehub(t)
mean(degree(t))
mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)]

m = matrixdepot("SNAP/loc-Brightkite")
t = SimpleGraph(m)
net_comp["loc_brightkite"] = [mean(local_clustering_coefficient(t))
global_clustering_coefficient(t)
mean(pagerank(t))
rich_club(t,1)
assortativity(t)
Bonchev_vd_info(t)
makehub(t)
mean(degree(t))
mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)]

m = matrixdepot("Newman/as-22july06")
t = SimpleGraph(m)
net_comp["routers"] = [mean(local_clustering_coefficient(t))
global_clustering_coefficient(t)
mean(pagerank(t))
rich_club(t,1)
assortativity(t)
Bonchev_vd_info(t)
makehub(t)
mean(degree(t))
mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)]

m = matrixdepot("Barabasi/NotreDame_yeast")
t = SimpleGraph(m)
net_comp["yeast"] = [mean(local_clustering_coefficient(t))
global_clustering_coefficient(t)
mean(pagerank(t))
rich_club(t,1)
assortativity(t)
Bonchev_vd_info(t)
makehub(t)
mean(degree(t))
mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)]

m = matrixdepot("DIMACS10/coPapersDBLP")
t = SimpleGraph(m)
net_comp["citation_dblp"] = [mean(local_clustering_coefficient(t))
global_clustering_coefficient(t)
mean(pagerank(t))
rich_club(t,1)
assortativity(t)
Bonchev_vd_info(t)
makehub(t)
mean(degree(t))
mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)]

m = matrixdepot("SNAP/com-DBLP")
t = SimpleGraph(m)
net_comp["com_dblp"] = [mean(local_clustering_coefficient(t))
global_clustering_coefficient(t)
mean(pagerank(t))
rich_club(t,1)
assortativity(t)
Bonchev_vd_info(t)
makehub(t)
mean(degree(t))
mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)]

ser_path("jlse/net_comp.jlse",net_comp)


t = SimpleGraph(sparse(matrixdepot("SNAP/com-Orkut")))
net_comp["com_orkut"] = [mean(local_clustering_coefficient(t))
global_clustering_coefficient(t)
mean(pagerank(t))
rich_club(t,1)
assortativity(t)
Bonchev_vd_info(t)
makehub(t)
mean(degree(t))
mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)]

ser_path("jlse/net_comp.jlse",net_comp)
t = nothing
GC.gc()

t = SimpleGraph(sparse(matrixdepot("SNAP/com-Amazon")))
net_comp["com_amazon"] = [mean(local_clustering_coefficient(t))
global_clustering_coefficient(t)
mean(pagerank(t))
rich_club(t,1)
assortativity(t)
Bonchev_vd_info(t)
makehub(t)
mean(degree(t))
mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)]

ser_path("jlse/net_comp.jlse",net_comp)
t = nothing
GC.gc()

t = SimpleGraph(sparse(matrixdepot("SNAP/com-Youtube")))
net_comp["com_youtube"] = [mean(local_clustering_coefficient(t))
global_clustering_coefficient(t)
mean(pagerank(t))
rich_club(t,1)
assortativity(t)
Bonchev_vd_info(t)
makehub(t)
mean(degree(t))
mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)]

ser_path("jlse/net_comp.jlse",net_comp)
t = nothing
GC.gc()

t = SimpleGraph(sparse(matrixdepot("SNAP/com-LiveJournal")))
net_comp["com_liveJournal"] = [mean(local_clustering_coefficient(t))
global_clustering_coefficient(t)
mean(pagerank(t))
rich_club(t,1)
assortativity(t)
Bonchev_vd_info(t)
makehub(t)
mean(degree(t))
mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)]

ser_path("jlse/net_comp.jlse",net_comp)
t = nothing
GC.gc()


#m = matrixdepot("SNAP/p2p-Gnutella08")
#t = SimpleDiGraph(m)
#m = matrixdepot("SNAP/email-Eu-core")
#t = SimpleDiGraph(m)

#t = loadsnap(:ca_astroph)
#t = loadsnap(:ego_twitter_u)
#t = loadsnap(:email_enron)
#t = loadsnap(:facebook_combined)
#t = loadsnap(:soc_slashdot0902_u)

net_comp = dser_path("jlse/net_comp.jlse")

t = dser_path("jlse/full_graph.jlse")
cc_loc = mean(local_clustering_coefficient(t, vertices(t)))
cc_glob = global_clustering_coefficient(t)
mu_prank = mean(pagerank(t))
rclub = rich_club(t,1)
assy = assortativity(t)
I_v = Bonchev_vd_info(t)
mhub = makehub(t)
mdeg = mean(degree(t))
r = mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)

net_comp["LA_run2"] = [mean(local_clustering_coefficient(t))
global_clustering_coefficient(t)
mean(pagerank(t))
rich_club(t,1)
assortativity(t)
Bonchev_vd_info(t)
makehub(t)
mean(degree(t))
mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)]

ser_path("jlse/net_comp.jlse", net_comp)

#t = watts_strogatz(17609971,8,0.25)

#t = barabasi_albert(17609971, 4)









mask = BitVector([1,1,0,0,1,0])
y = reduce(hcat, net_comp[k][mask] for k in
    ["LA_full","small_world","scale_free","p2p","email","ca_astroph",
    "ego_twitter_u","email_enron","facebook_combined","soc_slashdot0902_u"])
shapes = ([[:star4, :circle, :circle]; fill(:+,7)])

plot([1:3], y, shape=reshape(shapes,1,10),
    label="", xticks=:none, 
    linestyle=:dash, linealpha=0.5)





FTdist(v1::Matrix{T},v2::Matrix{T}) where T<:Real = sum((sqrt.(v1 .+ one(T)) .- sqrt.(v2 .+ one(T))) .^ 2)
summarize(idxs::Vector{Int64}, pop::Matrix{Int64}) = sum(view(pop,idxs,:), dims=1)
function summarize!(dest::Matrix{Int64}, idxs::Vector{Int64}, pop::Matrix{Int64})
    sum!(dest, view(pop,idxs,:))
    return nothing
end

## rmse standardized by st dev
SRMSE(O::Vector{T},E::Vector{T}) where T<:Real =  sqrt(mean((O .- E) .^ 2)) / std(E)

function read_test_targs()
	acs_targets = read_df("processed/test_targets.csv"; types=Dict("Geo"=>String15))
    targ_idxs = axes(acs_targets,1)
	targs = [Matrix{Int64}(acs_targets[[targ_id],2:end]) for targ_id in targ_idxs]
	geos = [acs_targets[targ_id,1] for targ_id in targ_idxs]
	return (targs, geos, names(acs_targets)[2:end])
end

function read_test_samples()
	puma_samples_all = read_df("processed/test_samples.csv"; types=Dict("SERIALNO"=>String15))
	hh_ids = puma_samples_all[:,1] ## for looking up the household serial numbers later
	shared_samples = Matrix{Int64}(puma_samples_all[:,2:end])
	return (shared_samples, hh_ids)	
end

function calc_test_scores()
    targs_all, geos_all, targ_colnames = read_test_targs()
    samples, hh_ids = read_test_samples()
    hh_idx = Dict(hh_ids .=> eachindex(hh_ids))
    ## each target's county
    county = [g[1:5] for g in geos_all]
    counties = unique(county)
    summary = similar(first(targs_all))

    for c in counties
        cbg_hhs = dser_path("jlse/CO/hh"*c*".jlse");        
        targs = targs_all[county .== c]
        geos = geos_all[county .== c]
        targ_idx = Dict(geos .=> eachindex(geos))
        test_scores = Dict{String15,Float64}()
        for (cbg,households) in cbg_hhs
            summarize!(summary, [hh_idx[k] for k in households], samples)
            test_scores[cbg] = FTdist(summary, targs[targ_idx[cbg]])
        end
        ser_path("jlse/CO/test_"*c*"_scores.jlse",test_scores)
    end
end



## targets are census block group (cbg) summary statistics
function read_targets(targ_idxs=[])
	acs_targets = read_df("processed/acs_targets.csv"; types=Dict("Geo"=>String15))
	if isempty(targ_idxs)
		targ_idxs = axes(acs_targets,1)
	end
	## pull target stats from the dataframe and convert to matrix or array for faster math 
	targs = [Matrix{Int64}(acs_targets[[targ_id],2:end]) for targ_id in targ_idxs]
	geos = [acs_targets[targ_id,1] for targ_id in targ_idxs]
	return (targs, geos, names(acs_targets)[2:end])
end

## returns a sharedmatrix of samples -- can pass it to local parallel processes without copying data
function read_samples()
	puma_samples_all = read_df("processed/census_samples.csv"; types=Dict("SERIALNO"=>String15))
	hh_ids = puma_samples_all[:,1] ## for looking up the household serial numbers later
	shared_samples = Matrix{Int64}(puma_samples_all[:,2:end])
	return (shared_samples, hh_ids)	
end


function all_summaries(test::Bool)
    if test
        targs_all, geos_all, targ_colnames = read_test_targs()
        samples, hh_ids = read_test_samples()
    else
        targs_all, geos_all, targ_colnames = read_targets()
        samples, hh_ids = read_samples()
    end

    hh_idx = Dict(hh_ids .=> eachindex(hh_ids))
    county = [g[1:5] for g in geos_all]
    counties = unique(county)
    summary = similar(first(targs_all))
    summaries = Dict{String15, typeof(summary)}()

    for c in counties
        cbg_hhs = dser_path("jlse/CO/hh"*c*".jlse")
        for (cbg,households) in cbg_hhs
            summarize!(summary, [hh_idx[k] for k in households], samples)
            summaries[cbg] = copy(summary)
        end
    end
    
    t_mat = zeros(length(targs_all), length(first(targs_all)))
    for (i,r) in enumerate(targs_all)
        t_mat[i,:] = r
    end
    
    s_mat = zeros(length(summaries), length(first(values(summaries))))
    for (i,g) in enumerate(geos_all)
        s_mat[i,:] = summaries[g]
    end
    
    return s_mat, t_mat, geos_all, targ_colnames
end


s_mat, t_mat, geos_all, targ_colnames = all_summaries(false)
err_by_targ = [(n, SRMSE(s_mat[:,i],t_mat[:,i])) for (i,n) in enumerate(targ_colnames)]

s_mat, t_mat, geos_all, targ_colnames = all_summaries(true)
err_by_testvar = [(n, SRMSE(s_mat[:,i],t_mat[:,i])) for (i,n) in enumerate(targ_colnames)]

y = [[x[2] for x in err_by_targ]; [x[2] for x in err_by_testvar]]
scatter(y, label="", xticks=:none)

#quantile.(Chisq(66), [0.05, 0.95])
#cdf(Chisq(66),48.0)

#quantile.(Chisq(37), [0.05, 0.95])
#cdf(Chisq(37),124.0)



function random_answer(glob_samp_ref::Matrix{Int64}, mask::BitVector, targ::Matrix{Int64}, n::Int64, params::Dict{Symbol, R}) where R<:Real
    samples = glob_samp_ref[mask, :]
    idxs = axes(samples,1) 
    ## if no valid samples, return a bad score (this shouldn't happen)
    if isempty(idxs)
        return (Vector{Int64}(), 0, Inf, 0.0)
    end
    c0 = rand(idxs,n)
    summary = summarize(c0, samples)
    E0 = FTdist(summary, targ)
    res = findall(mask)[c0]
    return (res, 0, E0, 0.0)
end

function random_answerer(shared_samples::Matrix{Int64}, params::Dict{Symbol, R}) where R<:Real
    function f(mask::BitVector, targ::Matrix{Int64}, n::Int64)
        return random_answer(shared_samples, mask, targ, n, params)
    end
    return f
end

## number of households in each cbg
function read_hh_counts()
	hhcdf = read_df("processed/hh_counts.csv"; types=Dict("Geo"=>String15))
	## convert to dicts for easier lookup
	##  note the .=> syntax; this broadcasts associations between two arrays, similar to dict(zip()) in python
	return Dict(hhcdf[:,1] .=> hhcdf[:,2])
end

## geographic data for target cbg's -- will use these to determine which samples to use
function read_targ_geo()
	cbg_geo_cols = Dict("Geo"=>String15,"st_puma"=>String7,"cbsa"=>String7,"county"=>String7,"R"=>Float64,"U"=>Float64)
	cbg_geo = read_df("processed/cbg_geo.csv"; select=collect(keys(cbg_geo_cols)), types=cbg_geo_cols)
	cbg_puma = Dict(cbg_geo[:,"Geo"] .=> cbg_geo[:,"st_puma"])
	cbg_county = Dict(cbg_geo[:,"Geo"] .=> cbg_geo[:,"county"])
	cbg_cbsa = Dict(cbg_geo[:,"Geo"] .=> cbg_geo[:,"cbsa"])
	cbg_urban = Dict(cbg_geo[:,"Geo"] .=> cbg_geo[:,"U"])
	return (cbg_puma, cbg_county, cbg_cbsa, cbg_urban)
end

## looks up which samples in df to use, based on column name k and associated geo data of targets
## don't usually need to specify function return types but in this case, target_geo_codes might be empty and 
##   this way it still returns a vector{BitVector}
function sample_lookup(df::DataFrame, k::Symbol, target_geo_codes)::Vector{BitVector}
	if k == :U
		return [urbanization_lookup(df,x) for x in target_geo_codes]
	else
		## note .== syntax for element-wise comparison
		return [coalesce.(df[!,k] .== x, false) for x in target_geo_codes]
	end
end

## creates annealing function and executes it on available processors
## returns a vector of whatever anneal() returns
function dont_optimize(samples::Matrix{Int64}, samp_masks::Vector{BitVector}, targs::Vector{Matrix{Int64}}, n_hhs::Vector{Int64}, params)
	a_fn = random_answerer(samples, params)
	return map(a_fn, samp_masks, targs, n_hhs)
end

## performs several optimization runs on the target cbg's in each county
## writes each county's results to a separate file
function process_counties(test::Bool)

    if test
        samples, hh_ids = read_test_samples()
        targs_all, geos_all, targ_colnames = read_test_targs()
    else
    	samples, hh_ids = read_samples()
        targs_all, geos_all, targ_colnames = read_targets()
    end

	cbg_puma, cbg_county, cbg_cbsa, cbg_urban = read_targ_geo()
	samp_geo_cols = Dict("SERIALNO"=>String15,"st_puma"=>String7,"cbsa"=>String7,"county"=>String7,"R"=>Float64,"U"=>Float64)
	samp_geo = read_df("processed/samp_geo.csv"; select=collect(keys(samp_geo_cols)), types=samp_geo_cols)
	hh_counts = read_hh_counts()
	n_hhs_all = [hh_counts[g] for g in geos_all]
	county = [g[1:5] for g in geos_all]

	params = Dict(:maxgens => 0, :critval => 0.0, :cooldown => 0.0, :report => 0)

    counties = unique(county)
	for c in counties
		targs = targs_all[county .== c]
		geos = geos_all[county .== c]
		n_hhs = n_hhs_all[county .== c]

		## look up samples for each target based on target's puma code
		## puma is the most local sample level, has the fewest samples available
		samp_masks = sample_lookup(samp_geo, :st_puma, [cbg_puma[g] for g in geos])
		x = dont_optimize(samples, samp_masks, targs, n_hhs, params)

		## store results as dict keyed by cbg code; look up actual hh id's of sample indices
		scores = Dict(geos .=> [a[3] for a in x])
		households = Dict(geos .=> [hh_ids[a[1]] for a in x])

        if test
            ser_path("jlse/CO/rand_test_"*c*".jlse",households)
            ser_path("jlse/CO/rand_test_"*c*"_scores.jlse",scores)
        else
            ser_path("jlse/CO/rand_"*c*".jlse",households)
            ser_path("jlse/CO/rand_"*c*"_scores.jlse",scores)
        end
	end

	return nothing
end

#process_counties(true)
#process_counties(false)

geos = read_targ_geo()
counties = unique(values(geos[2]))

x = [dser_path("jlse/CO/test_"*c*"_scores.jlse") for c in counties]
test_scores = merge(x...)
x = [dser_path("jlse/CO/hh"*c*"_scores.jlse") for c in counties]
scores = merge(x...)
x = [dser_path("jlse/CO/rand_"*c*"_scores.jlse") for c in counties]
rand_scores = merge(x...)
x = [dser_path("jlse/CO/rand_test_"*c*"_scores.jlse") for c in counties]
rand_test_scores = merge(x...)

cbgs = collect(keys(scores))

s1 = [scores[k] for k in cbgs]
s2 = [rand_scores[k] for k in cbgs]

plot([s1 s2], label="", xticks=:none)
plot([log.(s1) log.(s2)], label="", xticks=:none)

s1 = [test_scores[k] for k in cbgs]
s2 = [rand_test_scores[k] for k in cbgs]


