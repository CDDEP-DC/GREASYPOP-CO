#=
Copyright 2023 Alexander Tulchinsky

This file is part of Greasypop.

Greasypop is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Greasypop is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with Greasypop. If not, see <https://www.gnu.org/licenses/>.
=#

using Random
using SparseArrays
using ProportionalFitting
using Logging
#using Statistics

include("utils.jl")
include("fileutils.jl")

## splits n into a vector that sums to n
##   by drawing from a lognormal dist (mu, sigma)
##   receives and updates a vector of unused draws
function split_lognormal!(n::Integer, mu::Float64, sigma::Float64, draws::Vector{I}) where I<:Integer
    sizes = Vector{Int64}()
    ## don't bother generating if less than:
    while n > 2
        ## try to use an unused draw
        i = findfirst(x->x<=n, draws)
        if isnothing(i)
            sz = ceil(Int64,rlogn(mu,sigma)) ## round up to avoid 0's
            ## if it's too big, save for later (otherwise we're cheating at the distribution)
            while sz > n 
                push!(draws,sz)
                sz = ceil(Int64,rlogn(mu,sigma)) ## round up
            end
        else
            sz = popat!(draws,i)
        end
        push!(sizes, sz)
        n -= sz
    end
    ## append remaining, if any
    if n > 0
        push!(sizes,n)
    end
    return sizes
end

## reads people generated in households.jl
## returns {residence geo => list of worker keys} for each category
function group_commuters_by_origin(cat_codes::Vector{S}) where {S<:AbstractString}
    ## within each category, group by cbg, randomize order
    cbgs = dser_path("jlse/cbg_idxs.jlse")

    ## only commuters are assigned to workplaces
    ## dataframe provides fast grouping
    df = DataFrame([(id=k[1], hh=k[2], cbg=k[3], income=v.com_inc, category=v.com_cat) 
                    for (k,v) in filterv(x->x.commuter, dser_path("jlse/people.jlse"))])
    df_grouped = groupby(df, [:category, :cbg])

    ## a dict for each category, containing a dict of cbg => list of workers
    ## each worker is identified by (id, household, cbg, income category)
    ##  -- the latter is for easier generating of income-assortative workplace networks
    worker_keys = Dict(kc => 
                        Dict(ko => Vector{Tuple{fieldtypes(Pkey)...,eltype(df.income)}}()
                        for ko in values(cbgs))
                    for kc in cat_codes)

    ## fill the dict from dataframe
    ## shuffle each list of workers so they can be assigned to workplaces in order
    for gk in keys(df_grouped)
        cat_code, cbg_code = cat_codes[gk[:category]], cbgs[gk[:cbg]]
        worker_keys[cat_code][cbg_code] = shuffle( [(r.id,r.hh,r.cbg,r.income) for r in eachrow(df_grouped[gk])] )
    end
    
    return worker_keys
end

## for generating data-less "dummy" people who commute from outside the synth pop
function dummy_gen_fn()
    dummy_idx = 0
    ## each dummy needs to be provided an income code for assortative network construction
    f = function(origin::AbstractString, inc_code::Integer)
        dummy_idx += 1
        ## diagnostics (dummy should only be created when origin is "outside")
        d_err = origin == "outside" ? 0 : 1
        ## dummies have no household or home cbg
        return (dummy_idx,0,0,UInt8(inc_code)), d_err
    end
    return f
end

## household sample summaries has # of commuters for every sample household
## files in CO/ have a list of households by geo code
## return a dict of {residence geo => # commuters} for each category in cat_codes
function read_workers_by_cat(cat_codes::Vector{<:AbstractString}, counties::Vector{<:AbstractString})
    cat_cols = ["com_ind_"*k for k in cat_codes]
    hh_samps = read_df("processed/hh_samples.csv"; select=[["SERIALNO","NP","com_LODES_low","com_LODES_high"]; cat_cols], types=Dict("SERIALNO"=>String15))
    hh_idx = Dict(hh_samps.SERIALNO .=> eachindex(hh_samps.SERIALNO))
    
    workers_by_cat_x_ori = Dict(k => Dict{String15,Int64}() for k in cat_codes)
    for co in counties
        cbg_dict = dser_path("jlse/CO/hh"*co*".jlse") ## read households generated for cbgs in county
        for (ori, hhvec) in cbg_dict ## for each cbg in county, look up # workers in hh sample summaries
            for (cat_code, cat_col) in zip(cat_codes, cat_cols) ## sum each category across households
                workers_by_cat_x_ori[cat_code][ori] = sum(skipmissing(hh_samps[[hh_idx[x] for x in hhvec], cat_col]))
            end
        end
    end
    return workers_by_cat_x_ori
end

## summary generated in households.jl has # of workers in each GQ
##
## assume that all residents of military quarters work at the quarters
## only civilians in non-inst GQs commute to jobs
## (this assumption is also in households.jl)
##
function read_gq_workers_by_cat(cat_codes::Vector{<:AbstractString})
    gq_df = dser_path("jlse/df_gq_summary.jlse")
    cat_cols = ["ind_"*k for k in cat_codes]
    
    gq_by_cat_x_ori = Dict(k => Dict{String15,Int64}() for k in cat_codes)
    for r in eachrow(gq_df)
        for (cat_code, cat_col) in zip(cat_codes, cat_cols)
            gq_by_cat_x_ori[cat_code][r.geo] = r[cat_col]
        end
    end
    return gq_by_cat_x_ori
end

## read commute matrix for category k
## data is in sparse format, so m and n are needed to determine dimensions
function read_od_matrix(k::AbstractString, m::Integer, n::Integer)
    df = read_df("processed/od_"*k*".csv.gz"; types=Dict("origin"=>UInt32,"dest"=>UInt32,"p"=>Float32))
    return sparse(df.origin, df.dest, df.p, m, n)
end

## counts of workers commuting from outside the synth area
function read_outside_origins(cat_codes::Vector{S}) where {S<:AbstractString}
    tmp_dict = let df = read_df("processed/work_cats_live_outside.csv"); Dict(df[!,1] .=> df[!,2]); end
    return Dict(k => round(Int,tmp_dict["C24030:"*k]) for k in cat_codes)
end

## calculate origin-destination counts for each category in cat_codes
## write to file (don't need them all in mem at once)
## also keep hh and gq separate for now, in case they need to be treated differently
function calc_od_counts(cat_codes::Vector{<:AbstractString}, counties::Vector{<:AbstractString})

    ## these vectors map row and col idxs in the od matrices to geo codes in the population
    origin_labels = let df = read_df("processed/od_rows_origins.csv"); df.origin; end
    dest_labels = let df = read_df("processed/od_columns_dests.csv"); df.dest; end
    n_rows = length(origin_labels)
    n_cols = length(dest_labels)
    origin_idx = Dict(origin_labels .=> eachindex(origin_labels))

    ## read # workers from hh sample summary, workers living in gq's from gq summary
    ## these are dicts of geo code => count for each category in cat_codes
    println("reading worker counts")
    hhw_by_cat_x_ori = read_workers_by_cat(cat_codes,counties)
    gqw_by_cat_x_ori = read_gq_workers_by_cat(cat_codes)

    println("# workers living in synth area = ", sum(values(reduce(mergewith(+), values(hhw_by_cat_x_ori)))) + 
                                                sum(values(reduce(mergewith(+), values(gqw_by_cat_x_ori)))))

    ## need # workers with origin "outside"; append to hh workers dict
    outside_by_cat = read_outside_origins(cat_codes)
    for k in cat_codes
        hhw_by_cat_x_ori[k]["outside"] = outside_by_cat[k]
    end

    println("# workers living outside = ", sum(values(outside_by_cat)))

    ## multiply workers by od matrix
    println("calculating origin-destination counts")
    test_tot = 0
    for k in cat_codes
        println("  for category ",k)
        M = read_od_matrix(k, n_rows, n_cols);
        hh_counts = SparseMatrixCSC{UInt32, UInt32}(spzeros(n_rows, n_cols))
        gq_counts = SparseMatrixCSC{UInt32, UInt32}(spzeros(n_rows, n_cols))
        for (code, rownum) in origin_idx
            hh_counts[rownum,:] = lrRound(M[rownum,:] .*  get(hhw_by_cat_x_ori[k],code,0))
            gq_counts[rownum,:] = lrRound(M[rownum,:] .*  get(gqw_by_cat_x_ori[k],code,0))
        end
        ser_path("jlse/od_counts_"*k*".jlse",(hh_counts, gq_counts))
        test_tot += sum(hh_counts) + sum(gq_counts)
    end

    println("# workers assigned to destinations = ", test_tot)
    return (origin_labels, dest_labels)
end


## employer size stats by county
function read_county_stats()
    cols = Dict("county"=>String7,"mu_ln"=>Float64,"sigma_ln"=>Float64)
    county_stats = read_df("processed/work_sizes.csv"; select=collect(keys(cols)), types=cols)
    return Dict(county_stats.county .=> zip(county_stats.mu_ln, county_stats.sigma_ln))
end

## n people working in schools and school locations
function read_school_info()
    ## find the closest cbg for each school
    distmat = read_df("processed/cbg_sch_distmat.csv"; types=Dict("GEOID"=>String15))
    closest_cbg = Dict(x => distmat[argmin(distmat[!,x]),"GEOID"] for x in String15.(names(distmat)[2:end]))
    
    ## number of teachers by school
    sch_cols = Dict("NCESSCH"=>String15,"TEACHERS"=>Int64,"STUDENTS"=>Int64)
    schools = read_df("processed/schools.csv"; select=collect(keys(sch_cols)), types=sch_cols)
    sch_n_teachers = Dict(schools.NCESSCH .=> schools.TEACHERS)

    return (sch_n_teachers, closest_cbg)
end

## n people working in gqs and gq locations
function read_gq_info()
    ## find cbg where each gq is located
    gqs = dser_path("jlse/gqs.jlse")
    cbgs = dser_path("jlse/cbg_idxs.jlse")
    gq_cbgs = Dict(k => cbgs[k[2]] for k in keys(gqs))
    ## inst qg's: assume 1 workers per 10 residents; non-inst, 1 per 50? mininum 2?
    dConfig = tryJSON("config.json")
    r_i::Float64 = float(get(dConfig, "inst_res_per_worker", 10))
    r_ni::Float64 = float(get(dConfig, "noninst_res_per_worker", 50))
    e_min::Int = get(dConfig, "min_gq_workers", 2)
    ni_types = Set([:milGQ, :ninst1864civ])
    
    gq_n_emps = Dict(k => 
                max(e_min, ceil(Int64, v.type in ni_types ? length(v.residents)/r_ni : length(v.residents)/r_i)) 
                for (k,v) in gqs)

    return (gq_n_emps, gq_cbgs)
end

## filter work destinations by geo code and boolean function
## dest_idx is dict of geo code => index
## return indices
function filter_dests(f, geo::Sa, dest_idx::Dict{Sb, I}) where {I<:Integer,Sa<:AbstractString,Sb<:AbstractString}
    return collect(values(filter( ((k,v),) -> (startswith(k,geo) && f(v)) , dest_idx)))
end

## commute origins for people working in schools and gq's
## modifies count_matrix
function pull_inst_workers!(count_matrix::Matrix{<:Integer}, 
    dest_idx::Dict{<:AbstractString, <:Integer}, origin_labels::Vector{Ko},
    workers_by_key::Dict{Ki, <:Integer}, loc_by_key::Dict{Ki, <:AbstractString}) where {Ki<:Any,Ko<:AbstractString}

    ## for each institution, pull n workers who have a suitable work _destination_
    #test = Dict(k=>[] for k in keys(workers_by_key))
    inst_emp_origins = Dict{Ki,Vector{Ko}}()
    for (inst_id, n) in workers_by_key
        cbg = loc_by_key[inst_id]
        ## find destination(s) with enough workers; try closest cbg first, then wider areas
        ## limit to same county
        ## employment numbers may not agree exactly with commute numbers
        ## also, official employment location in commute data may not be the inst's actual location
        ## draw as many workers as possible going to the actual location
        ## then make up the remainder from nearby locations
        ## this way, commuters are coming from approximately the right area
        geo_areas = [cbg,cbg[1:11],cbg[1:9],cbg[1:7],cbg[1:5]]
        ## column sums change on each loop
        colsums = vec(sum(count_matrix;dims=1))
        ## for each geo area, a list of destination indices (columns) with > 0 workers:
        dest_lists = [filter_dests(i->(colsums[i]>0), geo, dest_idx) for geo in geo_areas]
        ## shuffle each list, but preserve the geo area order:
        avail_cols = unique(reduce(vcat, map(shuffle, dest_lists)))
        ## draw from columns until enough workers
        o_idxs = Int[]
        for col in avail_cols
            draw_n = min(colsums[col], n)
            #push!(test[inst_id], (col,draw_n))
            ## pass a view of the column so the original matrix gets modified
            append!(o_idxs, drawCounts!(view(count_matrix,:,col), draw_n))
            n = n - draw_n
            if n < 1
                break
            end
        end
        if n > 0
            println("warning: inst $inst_id loc $cbg short by $n workers")
        end

        inst_emp_origins[inst_id] = origin_labels[o_idxs]
    end
    return inst_emp_origins
end

## group by destination, split into workplaces, assign origins
## modifies count_matrix in place
## returns a dict of workplace ids => employee cbgs
function generate_workplaces!(count_matrix::Matrix{<:Integer}, dest_idx::Dict{<:AbstractString, <:Integer}, 
    origin_labels::Vector{Ko}, county_stats::Dict{<:AbstractString, Tuple{Float64, Float64}},
    draws_by_county::Dict{<:AbstractString, Vector{I}},
    cat_idx::Integer) where {Ko<:AbstractString,I<:Integer}

    work_origins = Dict{WRKkey, Vector{Ko}}()
    for (co,draws) in draws_by_county
        (mu,sigma) = county_stats[co]
        ## work destinations in the county we just read the stats for
        dests = filterk(k->k[1:5]==co, dest_idx)
        for (dest_code, col) in dests
            n = sum(count_matrix[:,col]) ## number of workers in the dest
            if n > 0
                sizes = split_lognormal!(n,mu,sigma,draws)
                for (work_i, emp_size) in enumerate(sizes)
                    ## get rand sample of origins; pass a view so original matrix is modified
                    o_idxs = drawCounts!(view(count_matrix,:,col), emp_size)
                    ## create workplace and assign origins sampled
                    work_origins[(work_i, cat_idx, dest_code)] = origin_labels[o_idxs]
                end
            end
        end
        #println(draws)
    end
    return work_origins
end

## make a separate work destination for each person working outside the synth area
##  (no need to try to group them, as they have no work network anyway)
function generate_outside_workplaces(work_outside::Dict{K, <:Integer}, cat_idx::Integer) where {K<:Any}
    return Dict([WRKkey((i, cat_idx, "outside")) for i in 1:sum(values(work_outside))]  
        .=> map(x->Vector{K}([x]), reduce(vcat, [fill(k, v) for (k, v) in work_outside])))
end

## assigns people in workers_by_origin to employers in emp_origins
## modifies cidx_by_origin so that it can be called with several emp_origins on the same pop
## creates "dummies" -- workers from outside that don't exist in people data
## returns a dict of employer id => vector of worker ids, and some diagnostics
function assign_workers!(emp_origins::Dict{T,Vector{K}}, workers_by_origin::Dict{K, Vector{W}}, 
    cidx_by_origin::Dict{K, I}, dummy_fn::F) where {T<:Any,K<:Any,W<:Any,I<:Integer,F<:Function}

    n_by_origin = Dict(k => length(v) for (k,v) in workers_by_origin)
    ## p high income workers, for generating dummies
    p_LODES_high = sum(count.(x->x[4]==2, values(workers_by_origin))) / sum(values(n_by_origin))

    ## initialize empty lists
    workers = Dict(est_id=>Vector{W}() for est_id in keys(emp_origins))
    dummies = Vector{W}() ## keep track of dummies created

    missing_origin = 0
    ran_out = Dict{K,Int64}()
    for (e_id, origin_vec) in emp_origins
        for origin_key in origin_vec
            ## if the origin is not in cidx, it's outside the synth area and a dummy must be created
            if haskey(cidx_by_origin, origin_key)
                i = cidx_by_origin[origin_key] += 1 ## this doesn't look like it should be allowed, but it is lol
                if i > n_by_origin[origin_key]
                    ran_out[origin_key] = get!(ran_out, origin_key, 0) + 1 ## this shouldn't happen
                else
                    push!(workers[e_id], (workers_by_origin[origin_key][i]))
                end
            else
                ## create dummy
                inc_code = rand() < p_LODES_high ? 2 : 1
                dum::W, d_err::Int = dummy_fn(origin_key, inc_code)
                push!(dummies, dum)
                push!(workers[e_id], dum)
                ## diagnostics (all origins except "outside" should have been in cidx)
                missing_origin += d_err
            end
        end
    end
    return (workers, dummies, missing_origin, ran_out)
end


function generate_jobs_and_workers()

    wp_codes = tryJSON("processed/codes.json")
    ind_codes::Vector{String} = get(wp_codes, "ind_codes", String[])
    ind_idxs = Dict(ind_codes .=> eachindex(ind_codes))
    ser_path("jlse/wp_cat_codes.jlse",ind_idxs)
    counties = let cbgs = dser_path("jlse/cbg_idxs.jlse"); unique(map(x->x[1:5], values(cbgs))); end

    println("reading commuters in synth pop")
    worker_keys = group_commuters_by_origin(ind_codes)
    ## if origin is outside the synth area, will generate a dummy instead of assigning a resident
    dummy_fn = dummy_gen_fn()

    ## calculate origin-destination counts and save to files
    ## "labels" vectors map row and col idxs in the od matrices to geo codes in the population
    ## workplaces will be generated based on the od matrices
    ## then connected to people/locations in the synth pop using the label indices
    origin_labels, dest_labels = calc_od_counts(ind_codes, counties)
    ser_path("jlse/od_labels.jlse",(origin_labels,dest_labels))

    dest_idx = Dict(dest_labels .=> eachindex(dest_labels)) ## for easy lookup

    ## mean and sd of ln(workplace size) for each county
    county_stats = read_county_stats()
    ## adjust stats
    ##  slightly overproduces large employers compared to CBP stats
    ##  (but unlisted employers should probably skew large)
    stats2 = deepcopy(county_stats)
    for (k,v) in stats2
        stats2[k] = (v[1], v[2]+0.1)
    end

    ## save employer size draws by county, to make best use of county-level stats
    draws_by_county = Dict(k => Vector{Int}() for k in counties)

    ## special treatment for these:
    school_teacher_category = "EDU"
    gq_employee_category = "ADM_MIL"
    ## could make gq workers more accurate by handling each type of gq separately
    ##   but there are not very many of them, and _most_ will be prison workers (ind #92)
    (sch_n_emps, sch_cbgs) = read_school_info()
    (gq_n_emps, gq_cbgs) = read_gq_info()

    ## data structures for collecting results
    sch_workers = Dict()
    gq_workers = Dict()
    company_workers = Dict()
    outside_workers = Dict()
    dummies = Dict()
    missing_origins = Dict()
    ran_out = Dict()

    ## do the following by category
    for ckey in ind_codes

        println("generating workplaces for category "*ckey)

        ## id's of synth pop residents to assign to workplaces
        workers_by_origin = worker_keys[ckey]
        ## instead of removing workers as they're assigned, just keep pointers to the first available worker
        cidx_by_origin = Dict(keys(workers_by_origin) .=> 0)

        ## origin-destination matrix for current category
        hh_sparse, gq_sparse = dser_path("jlse/od_counts_"*ckey*".jlse");
        od_counts = Matrix(hh_sparse+gq_sparse)
        ## pull off the last column (work destination outside the synth area)
        work_outside_counts = od_counts[:,end]
        od_counts = od_counts[:,1:end-1]
        ## by origin code, # people working outside the synth area
        work_outside = Dict(origin_labels .=> work_outside_counts)

        ## check totals (before modifying od_counts)
        n_jobs_in_synth_area = sum(od_counts)
        n_jobs_outside_area = sum(work_outside_counts)
        n_jobs_od_total = n_jobs_in_synth_area + n_jobs_outside_area
        n_com_from_synth_area = sum(od_counts[1:end-1,:]) + sum(work_outside_counts)
        n_com_from_outside = sum(od_counts[end,:])

        ## pull out workers for schools and gqs for each destination (except "outside")
        ##   (assumes workers from out of state can work in public schools)
        ##   samples worker origins in proportion to counts 
        ##   currently does not consider worker income, age, etc.
        if ckey == school_teacher_category
            sch_emp_origins = pull_inst_workers!(od_counts, dest_idx, origin_labels, sch_n_emps, sch_cbgs)
            (sch_workers[ckey], dummies["sch"*ckey], missing_origins["sch"*ckey], ran_out["sch"*ckey]) = assign_workers!(
                sch_emp_origins, workers_by_origin, cidx_by_origin, dummy_fn)
            n_sch_jobs_generated = sum(length.(values(sch_emp_origins)))
            n_sch_workers_assigned = sum(length.(values(sch_workers[ckey])))
        else
            n_sch_jobs_generated = 0
            n_sch_workers_assigned = 0
        end

        if ckey == gq_employee_category
            gq_emp_origins = pull_inst_workers!(od_counts, dest_idx, origin_labels, gq_n_emps, gq_cbgs)
            (gq_workers[ckey], dummies["gq"*ckey], missing_origins["gq"*ckey], ran_out["gq"*ckey]) = assign_workers!(
                gq_emp_origins, workers_by_origin, cidx_by_origin, dummy_fn)
            n_gq_jobs_generated = sum(length.(values(gq_emp_origins)))
            n_gq_workers_assigned = sum(length.(values(gq_workers[ckey])))
        else
            n_gq_jobs_generated = 0
            n_gq_workers_assigned = 0
        end

        ## create workplaces for remaining workers for each destination (except "outside")
        ## and assign workers to those
        work_origins = generate_workplaces!(od_counts, dest_idx, origin_labels, stats2, draws_by_county, ind_idxs[ckey])
        (company_workers[ckey], dummies[ckey], missing_origins[ckey], ran_out[ckey]) = assign_workers!(
            work_origins, workers_by_origin, cidx_by_origin, dummy_fn);

        ## check results
        n_unused_draws = sum(length.(values(draws_by_county)))
        n_jobs_generated_within = sum(length.(values(work_origins))) + n_sch_jobs_generated + n_gq_jobs_generated
        n_workers_living_within = sum(length.(values(workers_by_origin)))
        println("  # jobs within synth area = $n_jobs_in_synth_area outside = $n_jobs_outside_area total = $n_jobs_od_total")
        println("  # commuting from outside = $n_com_from_outside resident workers needed = $n_com_from_synth_area")
        println("  # jobs generated = $n_jobs_generated_within workers in synth area = $n_workers_living_within")
        println("  (# unused wp size draws = $n_unused_draws)")
        println("  # workers assigned to jobs = " ,sum(length.(values(company_workers[ckey]))) +  n_gq_workers_assigned +  n_sch_workers_assigned)

        ## don't forget to assign people to work_outside origins
        n_by_origin = Dict(k => length(v) for (k,v) in workers_by_origin)
        unused = Dict(k => max(0,(v - cidx_by_origin[k])) for (k,v) in n_by_origin)
        println("  # remaining workers = ", sum(values(unused)))

        (outside_workers[ckey], _, missing_origins["out"*ckey], ran_out["out"*ckey]) = assign_workers!(
            generate_outside_workplaces(work_outside, ind_idxs[ckey]), workers_by_origin, cidx_by_origin, dummy_fn);
        ## check results
        println("  # jobs created outside synth area = ", sum(length.(values(outside_workers[ckey]))))
    end

    ## merge categories, save to file for next step
    println("writing results to file")
    suffix = ""
    ser_path("jlse/work_dummies"*suffix*".jlse", reduce(vcat, values(dummies)))
    ser_path("jlse/sch_workers"*suffix*".jlse",reduce(vecmerge, values(sch_workers)))
    ser_path("jlse/gq_workers"*suffix*".jlse",reduce(vecmerge, values(gq_workers)))
    ser_path("jlse/company_workers"*suffix*".jlse",reduce(vecmerge, values(company_workers)))
    ser_path("jlse/outside_workers"*suffix*".jlse",reduce(vecmerge, values(outside_workers)))

    ## check results
    #reduce(+, values(missing_origins))
    #reduce(mergewith(+), values(ran_out))
    company_worker_counts = length.(values(reduce(vecmerge, values(company_workers))))
    sch_worker_counts = length.(values(reduce(vecmerge, values(sch_workers))))
    gq_worker_counts = length.(values(reduce(vecmerge, values(gq_workers))))
    outside_worker_counts = length.(values(reduce(vecmerge, values(outside_workers))))
    origin_worker_counts = length.(values(reduce(vecmerge, values(worker_keys))))
    n_workers_from_outside = sum(values(read_outside_origins(ind_codes)))
    println("")
    println("# workers living in synth pop = ", sum(origin_worker_counts))
    println("# workers commuting from outside = ",  n_workers_from_outside)
    println("total workers = ", sum(origin_worker_counts) + n_workers_from_outside)
    println("# company employees = ", sum(company_worker_counts))
    println("# school employees = ", sum(sch_worker_counts), " expected ",sum(values(sch_n_emps)))
    println("# gq employees = ", sum(gq_worker_counts), " expected ",sum(values(gq_n_emps)))
    println("# working outside synth pop = ", sum(outside_worker_counts))
    println("total assigned to jobs = ", sum(company_worker_counts)+sum(sch_worker_counts)+sum(gq_worker_counts)+sum(outside_worker_counts))
    println("")
    println("generated establishment sizes (other than schools and group quarters):")
    println("size<5: ",count(x->x<5, company_worker_counts))
    println("4<size<10: ",count(x->(4<x<10), company_worker_counts))
    println("9<size<20: ",count(x->(9<x<20), company_worker_counts))
    println("19<size<50: ",count(x->(19<x<50), company_worker_counts))
    println("49<size<100: ",count(x->(49<x<100), company_worker_counts))
    println("99<size<250: ",count(x->(99<x<250), company_worker_counts))
    println("249<size<500: ",count(x->(249<x<500), company_worker_counts))
    println("499<size<1000: ",count(x->(499<x<1000), company_worker_counts))
    println("999<size<1500: ",count(x->(999<x<1500), company_worker_counts))
    println("1499<size<2500: ",count(x->(1499<x<2500), company_worker_counts))
    println("2499<size<5000: ",count(x->(2499<x<5000), company_worker_counts))
    println("size>4999: ",count(x->x>4999, company_worker_counts))

    return nothing
end


## generate commute matrices by industry using IPF
## based on census and LODES data
## needed before workplaces can be generated
function generate_commute_matrices()
    println("generating commute matrices")
    Logging.disable_logging(Logging.Info)

    ind_codes::Vector{String} = let wp_codes = tryJSON("processed/codes.json"); get(wp_codes, "ind_codes", String[]); end;

    ## n of workers by industry at each origin, from census data (assumed true)
    (total_by_ori, m_ind_ori) = let io_df = read_df("processed/work_io_sums.csv"; types=Dict("Geo"=>String15));
        (sum(Matrix(io_df[!,2:end]);dims=2), 
        permutedims( rowRound(Matrix(io_df[!,2:end])) , (2,1) ));
    end;

    ## n workers commuting from each origin to each dest; census counts * proprtions from LODES OD data
    (origin_idxs, dest_idxs, m_dest_ori) = let od_df = read_df("processed/work_od_prop.csv"; types=Dict("Geo"=>String15))
        (od_df[!,:Geo], names(od_df)[2:end],
        sparse(permutedims( rowRound(total_by_ori .* Matrix(od_df[!,2:end])) , (2,1) )));
    end;

    ## p ind at each dest, estimated from WAC data
    m_ind_dest_p = let id_est_df = read_df("processed/work_id_est_sums.csv"; types=Dict(1=>String15));
        m_ind_dest = permutedims( Matrix(id_est_df[!,2:end]), (2,1) );
        m_ind_dest ./ sum(m_ind_dest; dims=1); ## p ind at each dest
    end;

    ## for storing results
    res_iod = [spzeros(Float32, length(origin_idxs), length(dest_idxs)) for i in ind_codes];

    for (o,x) in enumerate(origin_idxs)
        o % 100 == 0 && println(o, "/", length(origin_idxs))

        ## row sum targets = workers in each industry at this origin
        ind_margin = m_ind_ori[:,o]
        ## column sum targets = workers commuting to each destination from this origin
        ## only need to consider non-zero columns
        d_idxs, d_margin = findnz(m_dest_ori[:,o])
        if isempty(d_idxs)
            ## no commute data exists for this origin; have to make something up
            ## most likely, nobody lives there; show a warning if they do
            sum(ind_margin) > 0 && println("warning: no commute data for ",x,", ",sum(ind_margin)," workers affected")
            ## anyway, set destination equal to origin, or to a random dest preferably in same county
            new_m = fill(1.0,(length(ind_margin),1))
            d_idxs = Int[something(findfirst(x->x==origin_idxs[o], dest_idxs),
                                    rand(first_nonempty([findall(x->x[1:5]==origin_idxs[o][1:5], dest_idxs)
                                                        ,findall(x->true, dest_idxs)])))]
        else
            ## initial matrix based on p each industry at each destination, while preserving origin-destination counts
            ## IPF will make it also preserve by-industry counts
            ## this can be done separately for each origin; overall totals by destination and by industry will be correct
            preserve_od_vals = (d_margin' .* m_ind_dest_p[:, d_idxs]);
            ## do IPF; handles zeros poorly, so replace with some small value
            init_m = max.(preserve_od_vals,0.00001);
            fac = ipf(init_m, [ind_margin, d_margin], maxiter=500, tol=1e-6);
            ## result = proportion going to each destination (by industry)
            new_m = Array(fac) .* init_m 
            new_margin = sum(new_m; dims=2) ## will be same as ind_margin if IPF converged
            new_m = new_m ./ new_margin
            ## for industries with 0 workers at this location, still need some kind of commute data
            ## assume they follow overall proportions from OD counts
            new_m[isapprox.(vec(new_margin), 0.0), :] .= d_margin' / sum(d_margin)
        end
        ## store results
        for (i,c) in enumerate(ind_codes)
            res_iod[i][o,d_idxs] .= new_m[i,:]
        end
    end

    println("writing commute matrices")
    write_df("processed/od_rows_origins.csv", DataFrame(:idx=>eachindex(origin_idxs),:origin=>origin_idxs))
    write_df("processed/od_columns_dests.csv", DataFrame(:idx=>eachindex(dest_idxs),:dest=>dest_idxs))
    for (i,k) in enumerate(ind_codes)
        write_df("processed/od_"*k*".csv.gz", res_iod[i], [:origin,:dest,:p]; compress=true)
    end

    return nothing
end










