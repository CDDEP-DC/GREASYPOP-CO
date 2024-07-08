#=
Copyright 2023 Alexander Tulchinsky

This file is part of Greasypop.

Greasypop is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Greasypop is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with Greasypop. If not, see <https://www.gnu.org/licenses/>.
=#

#using Distributed
using Random
using Graphs
#using GraphPlot
using SparseArrays
using LinearAlgebra
using SharedArrays
using Distributions
using Plots
using StatsPlots
using StatsBase


#using MatrixDepot
#using SNAPDatasets

include("utils.jl")
include("fileutils.jl")


##
## network stats
##

## from Bonchev 2004, Ir(vd)
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



#f(v) = mean(v[2:end] ./ v[1:end-1])


#net_comp = Dict()
net_comp = dser_path("net_comp.jlse")

#M = sparse(Symmetric(dser_path("jlse/hh_adj_mat.jlse"))) .| sparse(Symmetric(dser_path("jlse/adj_mat.jlse")))
#M = dser_path("jlse/hh_adj_mat.jlse") .| dser_path("jlse/adj_mat.jlse")

x = (Edge(p) for p in zip(findnz(
    dser_path("jlse/adj_mat_hh.jlse") .| dser_path("jlse/adj_mat_non_hh.jlse")
    )[1:2]...))

t = SimpleGraphFromIterator(x)
x = nothing
GC.gc()

N = nv(t)  # 9424031
n0 = sum(degree(t).==0)  # 401057
mu = mean(degree(t)) ## 8.483098686750925
nname = "MD"

#N - n0
#mean(filter(i->i>0,degree(t)))
#deg_zero_i = findall(degree(t).==0)
#rem_vertices!(t, deg_zero_i)
#nv(t)
#mean(degree(t))
#nname = "MD0"


## "static scale free" algo 
## Goh K-I, Kahng B, Kim D: Universal behaviour of load distribution in scale-free networks. Phys Rev Lett 87(27):278701, 2001.

t = nothing
GC.gc()
t = barabasi_albert(N, 4); nname = "BA"
t = erdos_renyi(N, mu/N); nname = "ER"
t = watts_strogatz(N, 8, 0.25); nname = "WS"
t = static_scale_free(N, round(Int,0.5*mu*N), 3); nname = "SSF"
mean(degree(t))



net_comp[nname] = [mean(local_clustering_coefficient(t))
global_clustering_coefficient(t)
mean(pagerank(t))
rich_club(t,1)
assortativity(t)
Bonchev_vd_info(t)
makehub(t)
mean(degree(t))
#mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)
]

net_comp

ser_path("net_comp.jlse", net_comp)

nc_names = ["local clust","global clust","pagerank","rich club","assortativity","bonchev","makehub","mean degree"]
use_cols = [8,1,2,5,7,6]
[nc_names[x] for x in use_cols]
[k=>[round(v[x];sigdigits=3) for x in use_cols] for (k,v) in net_comp]






#adj_n0 = false
#if adj_n0
#    for x in shuffle(vertices(t))[1:n0]
#        for d in neighbors(t,x)
#            rem_edge!(t,x,d)
#        end
#    end
#end
#mean(degree(t))



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
#mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)
]

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
#mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)
]

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
#mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)
]

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
#mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)
]

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
#mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)
]

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
#mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)
]

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
#mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)
]

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
#mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)
]

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
#mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)
]

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
#mean(f(diffusion_rate(t,0.1,20)) for x in 1:20)
]

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

#net_comp = dser_path("jlse/net_comp.jlse")

mask = BitVector([1,1,0,0,1,0])
y = reduce(hcat, net_comp[k][mask] for k in
    ["LA_full","small_world","scale_free","p2p","email","ca_astroph",
    "ego_twitter_u","email_enron","facebook_combined","soc_slashdot0902_u"])
shapes = ([[:star4, :circle, :circle]; fill(:+,7)])

plot([1:3], y, shape=reshape(shapes,1,10),
    label="", xticks=:none, 
    linestyle=:dash, linealpha=0.5)



##
## checking fit of synth pop
##

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


scatter([x[2] for x in err_by_targ], label="", xticks=:none)
[(x[1],x[2]) for x in err_by_targ if x[2]>0.3]

[(x[1],x[2]) for x in err_by_testvar if x[2]<0.4]

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

calc_test_scores()
process_counties(true)
process_counties(false)

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

s05,s95 = log.(0.25 .* quantile.(Chisq(85), [0.05, 0.95]))
t05,t95 = log.(0.25 .* quantile.(Chisq(31), [0.05, 0.95]))

s1 = [scores[k] for k in cbgs]
s2 = [rand_scores[k] for k in cbgs]

t1 = [test_scores[k] for k in cbgs]
t2 = [rand_test_scores[k] for k in cbgs]

circ = Shape(Plots.partialcircle(0, 2π))
plot([log.(s1) log.(s2)], label="", xticks=:none, seriestype=:scatter, grid=false,
    ylims=(1.0,9.1),
    xlabel="census block group",ylabel="ln(¼ FT²)",
    markerstrokewidth=[0 0.5], markersize=[1.5 2.5], markercolor=[:black :white],
    markershape=[:none circ],markerstrokecolor=:black, markeralpha=[1.0 0.5],
    markerstrokealpha=[1.0 0.75],
    dpi=300)
hline!([s95], linecolor=:orangered, linestyle=:dash, label="")
hline!([t95], linecolor=:orangered, linestyle=:dash, label="")
savefig("scores-vs-rand.png")
savefig("test-vs-rand.png")



xlabels = ["optimized selection,\ntargeted variables" "random selection,\ntargeted variables"]
vals = [log.(s1) log.(s2)]
crit_val = s95

xlabels = ["optimized selection,\nuntargeted variables" "random selection,\nuntargeted variables"]
vals = [log.(t1) log.(t2)]
crit_val = t95


dotplot(xlabels, vals, label="",grid=false,ylabel="mismatch score",tick_direction=:out, 
    ylims=(1.0,9.1),
    markerstrokewidth=0, markersize=1.5, markercolor=:black, markershape=:none,
    markeralpha=0.5,
    dpi=300)

violin!(xlabels, vals, label="",
    fillalpha=0.0,linealpha=0.5,linewidth=0.5,linecolor=:black)

hline!([crit_val], linecolor=:orangered, linestyle=:dash, label="")

savefig("scores-vs-rand.png")
savefig("test-vs-rand.png")




## testing income segregated workplaces

cbg_idxs = dser_path("jlse/cbg_idxs.jlse")
hh = dser_path("jlse/hh.jlse")
gqs = dser_path("jlse/gqs.jlse")
people = dser_path("jlse/people.jlse")
df_gq_summary = dser_path("jlse/df_gq_summary.jlse")

length(values(people))
sum(p.working for p in values(people))
sum(p.commuter for p in values(people))
sum(p.com_LODES_low for p in values(people))
sum(p.com_LODES_high for p in values(people))
sum(coalesce(p.female,rand([true,false])) for p in values(people))
mean(p.age for p in values(people))

sum(df_gq_summary.noninst1864)
sum(df_gq_summary.working)
sum(df_gq_summary.not_working)
sum(df_gq_summary.commuter)
sum(df_gq_summary.wfh)
sum(df_gq_summary.com_LODES_low)
sum(df_gq_summary.com_LODES_high)

gq_ppl = [people[k] for k in reduce(vcat, [x.residents for x in values(gqs)])]
sum(df_gq_summary.instu18 + df_gq_summary.inst1864 + df_gq_summary.noninst1864 + df_gq_summary.inst65o)

sum(p.working for p in values(gq_ppl))
sum(p.commuter for p in values(gq_ppl))
sum(p.com_LODES_low for p in values(gq_ppl))
sum(p.com_LODES_high for p in values(gq_ppl))
mean(p.age for p in values(gq_ppl))

df = read_df("processed/work_od_matrix.csv"; types=Dict("h_cbg"=>String15))
df_noinc = read_df("processed/work_od_matrix_no_inc.csv"; types=Dict("h_cbg"=>String15))
df
df_noinc
df[!,:outside]
df_noinc[!, :outside]

hh_samps = read_df("processed/hh_samples.csv"; types=Dict("SERIALNO"=>String15))
hh_samps[:,
        [:SERIALNO,:NP,:HINCP,
        #:employed, :unemployed, :armed_forces, 
        :in_lf,
        :nilf, :work_from_home, :commuter, 
        #:worked_past_yr,
        :has_job,:com_LODES_low,:com_LODES_high]]

p_samps = read_df("processed/p_samples.csv"; types=Dict("SERIALNO"=>String15));
p_samps[: , [:SERIALNO,:WAGP,:PINCP,:PERNP,:COW,:ESR,:work_from_home,:commuter,:has_job,:com_LODES_low,:com_LODES_high]]






## testing stochastic block model

using Graphs
using GraphPlot
using LinearAlgebra

n_k = 2
mean_deg = 8
n1 = 18
n2 = 9
n_vec = [n1,n2]

w_planted = Diagonal(fill(mean_deg,n_k))
prop_i = n_vec ./ sum(n_vec)
w_random = repeat(transpose(prop_i) * mean_deg, n_k)

assoc_coeff = 0.9
c_matrix = assoc_coeff * w_planted + (1 - assoc_coeff) * w_random

## can't have more than n connections to a group
for r in eachrow(c_matrix)
    r .= min.(r,n_vec)
end
c_matrix

## can't have more than n-1 within-group connections
d_tmp = view(c_matrix, diagind(c_matrix))
d_tmp .= min.(d_tmp, n_vec .- 1)
c_matrix

## note, in a stochastic block model (SBM) the degree distribution within blocks is similar to 
##  a random (erdos-renyi) network. For a more realistic degree distribution (e.g., with hubs)
##  try a degree-corrected SBM (Karrer & Newman 2010, https://arxiv.org/abs/1008.3926)
g = stochastic_block_model(c_matrix,n_vec)
gplot(g)

## Graphs.jl sometimes creates 0-degree vertices but I don't think it should
fix_zeros = findall(degree(g).==0)
for x in fix_zeros
    add_edge!(g, x, rand(vertices(g)))
end

mean(degree(g))
mean(degree(g)[1:n1])
mean(degree(g)[(n1+1):end])

c_matrix

g2 = stochastic_block_model(reshape([8.0], 1, 1),[30])
gplot(g2)
mean(degree(g2))




## making sure all schools are getting students allocated 

sch_students = dser_path("jlse/sch_students.jlse")

schools = read_df("processed/schools.csv"; types=Dict("NCESSCH"=>String15))
distmat = read_df("processed/cbg_sch_distmat.csv"; types=Dict("GEOID"=>String15))
schools_idx = Dict(schools.NCESSCH .=> eachindex(schools.NCESSCH))
cap_by_school = Dict(k=>schools[schools_idx[k], "STUDENTS"] for (k,_) in sch_students)
n_students_by_school = Dict(k=>length(v) for (k,v) in sch_students)

sum(values(cap_by_school))
sum(values(n_students_by_school))
sum(values(n_students_by_school)) / sum(values(cap_by_school))

spots_left = Dict(k=>(cap_by_school[k]-n_students_by_school[k]) for (k,_) in sch_students)
p_filled = Dict(k=>(n_students_by_school[k] / cap_by_school[k]) for (k,_) in sch_students)
p_new = Dict(k=>(n_students_by_school[k] / cap_by_school[k]) for (k,_) in sch_students)

partialsort(collect(values(p_filled)),1:20)
partialsort(collect(values(p_new)),1:20)

sort(collect(values(p_filled)))[end-20:end]

using Plots
plot!(sort(collect(values(p_filled))))
plot!(sort(collect(values(p_new))))

sortperm(collect(values(p_filled)))
cap_by_school[collect(keys(p_filled))[601]]


## checking census data integrity

c10 = read_df("geo/2010_Census_Tract_to_2010_PUMA.txt";types=String)
acs_hh = read_df("bak/hh_all.csv",types=Dict("Geo"=>String))
od_matrix = read_df("processed/work_od_matrix_no_inc.csv"; types=Dict("h_cbg"=>String))
dConfig = tryJSON("geos.json")
geos = String.(dConfig["geos"])

c10[!,"Tract"] = c10.STATEFP .* c10.COUNTYFP .* c10.TRACTCE
c10[!,"County"] = c10.STATEFP .* c10.COUNTYFP
c10 = c10[[(r.STATEFP in geos || r.County in geos) for r in eachrow(c10)], ["Tract","PUMA5CE"]]

cbgs_in_acs = acs_hh.Geo
tracts_in_acs = unique( [x[1:end-1] for x in acs_hh.Geo] )
tracts_in_c10 = c10.Tract
all(sort(tracts_in_acs) .== sort(tracts_in_c10))

cbg_work_dests = names(od_matrix)[2:end-1]
cbg_origins = od_matrix.h_cbg
all([in(x,cbgs_in_acs) for x in cbg_work_dests])
all([in(x,cbgs_in_acs) for x in cbg_origins])

mask = [!in(x,cbg_origins) for x in cbgs_in_acs]
not_in_origins = cbgs_in_acs[mask]
acs_hh[[r.Geo in not_in_origins for r in eachrow(acs_hh)],:]


## test location-based contact matrices

loc_mat_keys = dser_path("jlse/loc_mat_keys.jlse")
work_loc_contact_mat = dser_path("jlse/work_loc_contact_mat.jlse")
res_loc_contact_mat = dser_path("jlse/res_loc_contact_mat.jlse")
work_loc_lookup = dser_path("jlse/work_loc_lookup.jlse")
res_loc_lookup = dser_path("jlse/res_loc_lookup.jlse")

w = dser_path("jlse/company_workers.jlse") ## employers/employees (with work locations)
hh = dser_path("jlse/hh.jlse") ## households/residents (with hh locations)
cbg_idxs = dser_path("jlse/cbg_idxs.jlse") ## location (cbg) keys used in person/hh keys
cbg_idxs = Dict(k=>String31(v) for (k,v) in cbg_idxs)
gqs = dser_path("jlse/gqs.jlse") ## group-quarters/residents (with gq locations)
## assume only non-inst GQ residents are available for ephemeral local contacts
gq_noninst = filterv(x->x.type==:noninst1864, gqs)
## use the same matrix indices as in the regular contact networks
k = dser_path("jlse/adj_mat_keys.jlse")
p_idxs = Dict(k .=> eachindex(k))

## group potential encounters by census tract (CBG seems too restrictive)
hh_tracts = unique(x[1:end-1] for x in values(cbg_idxs))
work_tracts = unique(x[3][1:end-1] for x in keys(w))
tracts = sort(unique([hh_tracts; work_tracts]))

loc_mat_keys_rev = Dict(v=>k for (k,v) in loc_mat_keys)

i = 100
t = findnz(work_loc_contact_mat[:,i])[1]
all(work_loc_lookup[x] == i for x in t)
loc_mat_keys_rev[i]
filterk(x->x[3][1:end-1]==loc_mat_keys_rev[i], w)
reduce(vcat, values(filterk(x->x[3][1:end-1]==loc_mat_keys_rev[i], w)))
t2 = [p_idxs[k[1:3]] for k in reduce(vcat, values(filterk(x->x[3][1:end-1]==loc_mat_keys_rev[i], w)))]
all(sort(t) .== sort(t2))

i = 123
t = findnz(res_loc_contact_mat[:,i])[1]
all(res_loc_lookup[x] == i for x in t)
loc_mat_keys_rev[i]
filterk(x->cbg_idxs[x[2]][1:end-1]==loc_mat_keys_rev[i], hh)
tk1 = reduce(vcat, [h.people for h in values(filterk(x->cbg_idxs[x[2]][1:end-1]==loc_mat_keys_rev[i], hh))])
filterk(x->cbg_idxs[x[2]][1:end-1]==loc_mat_keys_rev[i], gq_noninst)
tk2 = reduce(vcat, [x.residents for x in values(filterk(x->cbg_idxs[x[2]][1:end-1]==loc_mat_keys_rev[i], gq_noninst))])
t2 = [p_idxs[k[1:3]] for k in [tk1;tk2]]
all(sort(t) .== sort(t2))

