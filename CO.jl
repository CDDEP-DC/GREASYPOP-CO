#=
Copyright 2023 Alexander Tulchinsky

This file is part of Greasypop.

Greasypop is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Greasypop is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with Greasypop. If not, see <https://www.gnu.org/licenses/>.
=#

using Distributed
include("fileutils.jl")

## the definitions in this block are sent to all processors
@everywhere begin

    ## for sharing data between local processors without copying
    using SharedArrays

    ## mutation function replaces one element in combs with an element from opts
    ## avoid making an array copy, it only takes 1 assignment to revert
    function mutate!(comb, opts)
        cidx = rand(eachindex(comb))
        orig = comb[cidx]
        comb[cidx] = rand(opts) ## mutate passed vector
        return (cidx,orig) ## return info needed to revert
    end

    function revert!(comb,(cidx,orig))
        comb[cidx] = orig
        return nothing
    end

    ## pick the rows given by idxs from the pop, return summary of columns
    ##  use view() to avoid making a copy
    summarize(idxs::Vector{Int64}, pop::Matrix{Int64}) = sum(view(pop,idxs,:), dims=1)
    ## use sum!() to save on array allocations
    function summarize!(dest::Matrix{Int64}, idxs::Vector{Int64}, pop::Matrix{Int64})
        sum!(dest, view(pop,idxs,:))
        return nothing
    end

    ## distance functions
    ## note if coming from Numpy: Julia functions do not automatically broadcast over collections
    ## --- must specify that intent using dot-syntax: .+ for operators, f.() for functions
    ## (also, Julia docs said to avoid collections of abstract types, hence the polymorphic
    ##   type specification "Matrix{T} where T<:Real" instead of the simpler Matrix{Real})
    ## FTdist is 1/4 * Freeman-Tukey distance
    ## -- multiply by 4 to get something with a chi-sq distribution with len(v)-1 d.f.
    #FTdist(v1::Matrix{T},v2::Matrix{T}) where T<:Real = sum((sqrt.(v1) .- sqrt.(v2)) .^ 2)
    ## zeros are common in target cells, and get penalized too harshly -- add 1 to relax this
    FTdist(v1::Matrix{T},v2::Matrix{T}) where T<:Real = sum((sqrt.(v1 .+ one(T)) .- sqrt.(v2 .+ one(T))) .^ 2)
    ## weighted
    #FTdist(v1::Matrix{T},v2::Matrix{T},wts::Matrix{Float64}) where T<:Real = sum(wts .* ((sqrt.(v1 .+ one(T)) .- sqrt.(v2 .+ one(T))) .^ 2))
    ## or, good ol' euclidean-squared
    #sqdist(v1::Matrix{T},v2::Matrix{T}) where T<:Real = sum((v1 .- v2) .^ 2)

    ## standard simulated annealing acceptance function
    function accept(neg_dE::Float64, T::Float64)
        if neg_dE >= 0
            return true
        else
            p = exp(neg_dE / T)
            return rand() < p
        end
    end

    ## function for stepping down the temperature; adaptive would probably be better?
    function sched(gen, temp, E0, E1, c)
        return temp * c
    end

    ## termination criteria
    function term(gen, E0, temp, max_gens, crit_val, report=1000)
        t = E0 < crit_val || gen > max_gens
        if gen % report == 0 || t
            println(E0, " ", gen, " ", temp)
        end
        return t
    end

    ## optimize with simulated annealing
    ##  returns a vector of indices from samples
    function anneal(glob_samp_ref::SharedMatrix{Int64}, mask::BitVector, targ::Matrix{Int64}, n::Int64, params::Dict{Symbol, R}) where R<:Real
        ## make a local copy of needed samples (this isn't done that often, and should be the fastest approach)
        samples = glob_samp_ref[mask, :]
        ## indices to sample
        ##  note, these index the local subset, not the shared samples
        idxs = axes(samples,1) 
        ## if no valid samples, return a bad score (this shouldn't happen)
        if isempty(idxs)
            return (Vector{Int64}(), 0, Inf, 0.0)
        end
        ## initialize with a random combination of samples
        c0 = rand(idxs,n)
        summary = summarize(c0, samples)
        E0 = FTdist(summary, targ)
        T = 0.5 * E0 ## set initial temperature based on starting distance, I guess?
        gen = 0

        done = false
        while !done
            gen += 1
            orig = mutate!(c0, idxs)
            summarize!(summary, c0, samples)
            E1 = FTdist(summary, targ)
            if accept(E0-E1, T)
                T = sched(gen, T, E0, E1, params[:cooldown])
                E0 = E1 ## only update the score if we accept, lol
            else
                revert!(c0, orig)
            end
            done = term(gen, E0, T, params[:maxgens], params[:critval], params[:report])
        end

        ## convert the answer back to global indices
        ## I wasn't really going to forget to do this, was I?
        res = findall(mask)[c0]

        ## force garbage collection, distributed GC isn't always smart
        GC.gc()

        return (res, gen, E0, T)
    end

    ## closes over params and a reference to shared matrix of samples
    ## returns a fn that uses anneal with given params, and can see the shared samples
    function annealer(shared_samples::SharedMatrix{Int64}, params::Dict{Symbol, R}) where R<:Real
        ## call this function to do the work:
        ##  mask = which samples from shared_samples to use for a given target
        function f(mask::BitVector, targ::Matrix{Int64}, n::Int64)
            return anneal(shared_samples, mask, targ, n, params)
        end
        return f
    end

## end @everywhere
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

## number of households in each cbg
function read_hh_counts()
    hhcdf = read_df("processed/hh_counts.csv"; types=Dict("Geo"=>String15))
    ## convert to dicts for easier lookup
    ##  note the .=> syntax; this broadcasts associations between two arrays, similar to dict(zip()) in python
    return Dict(hhcdf[:,1] .=> hhcdf[:,2])
end

## returns a sharedmatrix of samples -- can pass it to local parallel processes without copying data
function read_samples()
    puma_samples_all = read_df("processed/census_samples.csv"; types=Dict("SERIALNO"=>String15))
    hh_ids = puma_samples_all[:,1] ## for looking up the household serial numbers later
    shared_samples = convert(SharedMatrix, Matrix{Int64}(puma_samples_all[:,2:end]))
    return (shared_samples, hh_ids)	
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

## for finding samples with similar urbanization to target urbanization x
function urbanization_lookup(df::DataFrame, x::Float64)
    if x > 0.999
        ## if df has a missing cell, comparison will return "missing"
        ## -- coalesce() makes it return "false" instead
        return (coalesce.(df.U .> 0.999, false))
    elseif x < 0.334
        return (coalesce.(df.U .< 0.334, false))
    else
        a = x - 0.1; b = min(x+0.1, 0.999)
        return (coalesce.(a .< df.U .< b, false))
    end
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
function optimize(samples::SharedMatrix{Int64}, samp_masks::Vector{BitVector}, targs::Vector{Matrix{Int64}}, n_hhs::Vector{Int64}, params)
    a_fn = annealer(samples, params)
    ## use pmap for this, not @distributed for
    return pmap(a_fn, samp_masks, targs, n_hhs)
end

## like optimize, but overwrites previous results in vector "x" at the indices given by "rerun"
function reoptimize!(x, rerun::Vector{Int64}, samples::SharedMatrix{Int64}, 
    samp_masks::Vector{BitVector}, targs::Vector{Matrix{Int64}}, n_hhs::Vector{Int64}, params_r)

    ## make sure we found enough suitable samples for each target
    enough_samps = sum.(samp_masks) .> (n_hhs[rerun] .// 2)
    rerun = rerun[enough_samps]
    ## make sure there's actually something to rerun
    if lastindex(rerun) > 0
        a_fn = annealer(samples, params_r)
        x_r = pmap(a_fn, samp_masks, targs[rerun], n_hhs[rerun])
        ## only overwrite the ones whose score improved
        improved = [a[3] for a in x_r] .< [a[3] for a in x[rerun]]
        ## note .= syntax for broadcasting assignment to an array
        x[rerun[improved]] .= x_r[improved]
    end
    return nothing
end

## performs several optimization runs on the target cbg's in each county
## writes each county's results to a separate file
function process_counties(counties=[])

    samples, hh_ids = read_samples()
    cbg_puma, cbg_county, cbg_cbsa, cbg_urban = read_targ_geo()
    samp_geo_cols = Dict("SERIALNO"=>String15,"st_puma"=>String7,"cbsa"=>String7,"county"=>String7,"R"=>Float64,"U"=>Float64)
    samp_geo = read_df("processed/samp_geo.csv"; select=collect(keys(samp_geo_cols)), types=samp_geo_cols)
    ## samp geo indices match those in census_samples
    ## all(samp_geo.SERIALNO .== hh_ids)
    targs_all, geos_all, targ_colnames = read_targets()
    hh_counts = read_hh_counts()
    ## number of households in each target
    n_hhs_all = [hh_counts[g] for g in geos_all]
    ## each target's county
    county = [g[1:5] for g in geos_all]

    ## weight households more than individuals? (not currently implemented)
    #targ_weights = [ones(1,43).*2.0 ones(1,length(targ_colnames)-43)]
    dConfig = tryJSON("config.json")
    c_val::Float64 = get(dConfig, "CO_crit_val", 10.0)
    CO_cooldown::Float64 = get(dConfig, "CO_cooldown", 0.99)
    CO_cooldown_slow = 0.5 + 0.5 * CO_cooldown
    CO_maxgens::Int = get(dConfig, "CO_maxgens", 200000)
    CO_report = CO_maxgens + 1
    
    params = Dict(:maxgens => CO_maxgens, :critval => c_val, :cooldown => CO_cooldown, :report => CO_report)

    ## run all counties by default
    if isempty(counties)
        counties = unique(county)
    end

    mkpath("jlse/CO")
    for c in counties
        targs = targs_all[county .== c]
        geos = geos_all[county .== c]
        n_hhs = n_hhs_all[county .== c]

        println("\n puma \n")
        ## look up samples for each target based on target's puma code
        ## puma is the most local sample level, has the fewest samples available
        samp_masks = sample_lookup(samp_geo, :st_puma, [cbg_puma[g] for g in geos])
        x = optimize(samples, samp_masks, targs, n_hhs, params)

        println("\n county \n")
        ## rerun poor matches using county-level
        rerun = findall([a[3]>c_val for a in x])
        samp_masks = sample_lookup(samp_geo, :county, [cbg_county[g] for g in geos[rerun]])
        params_r = Dict(:maxgens => CO_maxgens, :critval => c_val, :cooldown => CO_cooldown, :report => CO_report)
        reoptimize!(x, rerun, samples, samp_masks, targs, n_hhs, params_r)

        println("\n cbsa \n")
        ## cbsa-level
        rerun = findall([a[3]>c_val for a in x])
        samp_masks = sample_lookup(samp_geo, :cbsa, [cbg_cbsa[g] for g in geos[rerun]])
        params_r = Dict(:maxgens => CO_maxgens, :critval => c_val, :cooldown => CO_cooldown, :report => CO_report)
        reoptimize!(x, rerun, samples, samp_masks, targs, n_hhs, params_r)
        
        println("\n urb \n")
        ## urbanization level, has the most samples; more likely to match but longer to search
        ##  (also the least associated with the target's local geography)
        rerun = findall([a[3]>c_val for a in x])
        samp_masks = sample_lookup(samp_geo, :U, [cbg_urban[g] for g in geos[rerun]])
        params_r = Dict(:maxgens => CO_maxgens, :critval => c_val, :cooldown => CO_cooldown_slow, :report => CO_report)
        reoptimize!(x, rerun, samples, samp_masks, targs, n_hhs, params_r)

        ## store results as dict keyed by cbg code; look up actual hh id's of sample indices
        scores = Dict(geos .=> [a[3] for a in x])
        households = Dict(geos .=> [hh_ids[a[1]] for a in x])

        serialize(abspath("jlse/CO/hh"*c*".jlse"),households)
        serialize(abspath("jlse/CO/hh"*c*"_scores.jlse"),scores)

        ## force garbage collection, distributed GC isn't always smart
        GC.gc()
    end

    return nothing
end


process_counties()


