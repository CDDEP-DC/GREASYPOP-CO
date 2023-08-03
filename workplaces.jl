#=
Copyright 2023 Alexander Tulchinsky

This file is part of Greasypop.

Greasypop is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Greasypop is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with Greasypop. If not, see <https://www.gnu.org/licenses/>.
=#

using Random
#using Statistics

include("utils.jl")
include("fileutils.jl")

## splits n into a vector that sums to n
##   by drawing from a lognormal dist (mu, sigma)
##   receives and updates a vector of unused draws
function split_lognormal!(n::Int64, mu::Float64, sigma::Float64, draws::Vector{Int64})
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

## loop through cbgs within counties and add number of workers to dict
function read_workers_by_cbg(origin_codes)
    ## household sample summaries has # of workers for every sample household
    hh_samps = read_df("processed/hh_samples.csv"; types=Dict("SERIALNO"=>String15))
    hh_idx = Dict(hh_samps.SERIALNO .=> eachindex(hh_samps.SERIALNO))
    workers_by_cbg = Dict{String15,Int64}()
    counties = unique(map(x->x[1:5], origin_codes))
    for co in counties
        ## read households generated for cbgs in county
        cbg_dict = dser_path("jlse/CO/hh"*co*".jlse")

        ## for each cbg in county, look up # workers in hh sample summaries
        for c in filter(k->k[1:5]==co, origin_codes)
            if haskey(cbg_dict,c) ## some cbgs don't have households
                hhvec = cbg_dict[c] ## household id's in this cbg
                hh_workers = Vector{Int64}()
                for x in hhvec ## add number of workers from each household (only commuters count)
                    n_work = hh_samps[hh_idx[x],:commuter]
                    if !ismissing(n_work) 
                        push!(hh_workers, n_work)
                    end
                end
                workers_by_cbg[c] = sum(hh_workers)
            end
        end
    end
    return workers_by_cbg
end

## origin codes are a tuple of (cbg, income code)
function read_workers_cbg_x_inc(origin_codes)
    ## household sample summaries has # of workers for every sample household
    hh_samps = read_df("processed/hh_samples.csv"; types=Dict("SERIALNO"=>String15))
    hh_idx = Dict(hh_samps.SERIALNO .=> eachindex(hh_samps.SERIALNO))
    codes = Dict("inc_low"=>:com_LODES_low, "inc_high"=>:com_LODES_high) ## column corresponding to each income code
    workers_cbg_x_inc = Dict{Tuple{String15,String15},Int64}()
    counties = unique(map(x->x[1][1:5], origin_codes))
    for co in counties
        ## read households generated for cbgs in county
        cbg_dict = dser_path("jlse/CO/hh"*co*".jlse")

        ## for each cbg in county, look up # workers in hh sample summaries
        for (c,i_code) in filter(k->k[1][1:5]==co, origin_codes)
            if haskey(cbg_dict,c) ## some cbgs don't have households
                hhvec = cbg_dict[c] ## household id's in this cbg
                hh_workers = Vector{Int64}()
                for x in hhvec ## add number of workers from each household (only commuters count)
                    n = hh_samps[hh_idx[x],codes[i_code]]
                    push!(hh_workers, coalesce(n, 0))
                end
                workers_cbg_x_inc[(c,i_code)] = sum(hh_workers)
            end
        end
    end
    return workers_cbg_x_inc
end

## workers living in group quarters, calculated in households.jl
function read_gq_workers()
    gq_df = dser_path("jlse/df_gq_summary.jlse")
    gq_workers_by_cbg = Dict{String15,Int64}()
    for r in eachrow(gq_df)
        if r.commuter > 0
            gq_workers_by_cbg[String15(r.geo)] = r.commuter
        end
    end
    return gq_workers_by_cbg
end

function read_gq_workers_by_inc()
    gq_df = dser_path("jlse/df_gq_summary.jlse")
    gq_w_cbg_x_inc = Dict{Tuple{String15,String15},Int64}()
    codes = Dict("inc_low"=>:com_LODES_low, "inc_high"=>:com_LODES_high) ## column corresponding to each income code
    for r in eachrow(gq_df)
        for (i_code, col_name) in codes
            if r[col_name] > 0
                gq_w_cbg_x_inc[(r.geo,i_code)] = r[col_name]
            end
        end
    end
    return gq_w_cbg_x_inc
end

## OD data for workers commuting from outside the synth area
function read_outside_origins()
    df_live_outside = read_df("processed/work_locs_live_outside.csv"; 
            select=[:w_cbg,:S000,:h_state], 
            types=Dict([:w_cbg,:S000,:h_state] .=> [String15,Int64,String3]))
    ## only count people living in neighboring states (assume others work remotely)
    dConfig = tryJSON("config.json")
    v = get(dConfig, "commute_states", nothing)
    if !isnothing(v)
        neigh_states = Set(String.(v))
        df_live_outside = filter(r->in(r.h_state,neigh_states), df_live_outside)
    end
    df_by_dest = combine(groupby(df_live_outside, :w_cbg), :S000 => sum => :n)
    return Dict(df_by_dest.w_cbg .=> df_by_dest.n)
end

## in OD data, using SE01 + SE02 for low income, SE03 for high income
function read_outside_by_inc()
    df_live_outside = read_df("processed/work_locs_live_outside.csv"; 
            select=[:w_cbg,:SE01,:SE02,:SE03,:h_state], 
            types=Dict([:w_cbg,:SE01,:SE02,:SE03,:h_state] .=> [String15,Int64,Int64,Int64,String3]))
    ## only count people living in neighboring states (assume others work remotely)
    dConfig = tryJSON("config.json")
    v = get(dConfig, "commute_states", nothing)
    if !isnothing(v)
        neigh_states = Set(String.(v))
        df_live_outside = filter(r->in(r.h_state,neigh_states), df_live_outside)
    end
    df_by_dest = combine(groupby(df_live_outside, :w_cbg), :SE01 => sum, :SE02 => sum, :SE03 => sum)
    df_by_dest[!,:inc_low] = df_by_dest[!,:SE01_sum] .+ df_by_dest[!,:SE02_sum]
    df_by_dest[!,:inc_high] = df_by_dest[!,:SE03_sum]
    df_by_dest = stack(df_by_dest[!,[:w_cbg,:inc_low,:inc_high]], 2:3)
    return Dict(zip(df_by_dest.w_cbg, String15.(df_by_dest[:,2])) .=> df_by_dest.value)
end

## number of teachers by school
function read_sch_teachers()
    sch_cols = Dict("NCESSCH"=>String15,"TEACHERS"=>Int64,"STUDENTS"=>Int64)
    schools = read_df("processed/schools.csv"; select=collect(keys(sch_cols)), types=sch_cols)
    return Dict(schools.NCESSCH .=> schools.TEACHERS)
end

## filter work destinations by geo code and boolean function
function filter_dests(f, geo::AbstractString, dest_idx::Dict{String, Int64})
    return filter( ((k,v),) -> (startswith(k,geo) && f(v)) , dest_idx)
end

## pulls teachers from likely work destinations based on numbers in sch_n_teachers
## modifies count_matrix in-place
## return dict of school id => employee origins
function pull_teachers!(count_matrix::Matrix{Int64}, dest_idx::Dict{String, Int64}, origin_labels::Vector{K}) where K<:Any
    ## find the closest cbg for each school
    distmat = read_df("processed/cbg_sch_distmat.csv"; types=Dict("GEOID"=>String15))
    closest_cbg = Dict(x => string(distmat[argmin(distmat[!,x]),"GEOID"]) for x in names(distmat))

    sch_n_teachers = read_sch_teachers()
    sch_emp_origins = Dict{String15,Vector{K}}()
    ## for each school, pull n teachers who have a suitable work _destination_
    for (school_id, n) in sch_n_teachers
        cbg = closest_cbg[school_id]
        tract = cbg[1:11]
        subcty = cbg[1:9]
        cty = cbg[1:5]
        ## find a destination with enough workers; try closest cbg first, then wider areas
        opts = first_nonempty([filter_dests(v->sum(count_matrix[:,v])>=n, geo, dest_idx) for geo in [cbg,tract,subcty,cty]])
        ## if multiple suitable options, choose at random
        (dest,col) = rand(opts)
        ## pass a view of the chosen column so the original matrix gets modified
        o_idxs = drawCounts!(view(count_matrix,:,col), n)
        sch_emp_origins[school_id] = origin_labels[o_idxs]
    end
    return sch_emp_origins    
end

## pulls gq employees from likely work destinations based on numbers in gq_n_emps
##  (this is people working at group quarters)
## modifies count_matrix in-place
## return dict of gq id => employee origins
function pull_gq_workers!(count_matrix::Matrix{Int64}, dest_idx::Dict{String, Int64}, origin_labels::Vector{K}) where K<:Any
    ## find cbg where each gq is located
    gqs = dser_path("jlse/gqs.jlse")
    cbgs = dser_path("jlse/cbg_idxs.jlse")
    gq_cbgs = Dict(k => cbgs[k[2]] for k in keys(gqs))
    ## inst qg's: assume 1 workers per 10 residents; non-inst, 1 per 50? mininum 2?
    dConfig = tryJSON("config.json")
    r_i::Float64 = float(get(dConfig, "inst_res_per_worker", 10))
    r_ni::Float64 = float(get(dConfig, "noninst_res_per_worker", 50))
    e_min::Int = get(dConfig, "min_gq_workers", 2)
    
    gq_n_emps = Dict(k => 
        max(e_min, ceil(Int64, v.type == :noninst1864 ? length(v.residents)/r_ni : length(v.residents)/r_i)) 
        for (k,v) in gqs)
    
    gq_emp_origins = Dict{GQkey,Vector{K}}()
    ## for each gq, pull n workers who have a suitable work _destination_
    for (gq_id, n) in gq_n_emps
        cbg = gq_cbgs[gq_id]
        tract = cbg[1:11]
        subcty = cbg[1:9]
        cty = cbg[1:5]
        ## find a destination with enough workers; try closest cbg first, then wider areas
        opts = first_nonempty([filter_dests(v->sum(count_matrix[:,v])>=n, geo, dest_idx) for geo in [cbg,tract,subcty,cty]])
        ## if multiple suitable options, choose at random
        (dest,col) = rand(opts)
        ## pass a view of the chosen column so the original matrix gets modified
        o_idxs = drawCounts!(view(count_matrix,:,col), n)
        gq_emp_origins[gq_id] = origin_labels[o_idxs]
    end
    return gq_emp_origins    
end

## allocate workers to destinations for each cbg, based on commute matrix
function allocate_workers(od_matrix::Matrix{Float64}, od_idx::Dict{K, Int64}, workers_by_cbg::Dict{K, Int64}) where K<:Any
    od_counts = zeros(Int64, size(od_matrix))
    for (code, rownum) in od_idx 
        ## note, od matrix contains cbgs that we didn't create; e.g, not enough households
        if haskey(workers_by_cbg, code)
            od_counts[rownum,:] = lrRound(od_matrix[rownum,:] .* workers_by_cbg[code])
        end    
    end
    return od_counts
end

## employer size stats by county
function read_county_stats()
    cols = Dict("county"=>String7,"mu_ln"=>Float64,"sigma_ln"=>Float64)
    county_stats = read_df("processed/work_sizes.csv"; select=collect(keys(cols)), types=cols)
    return Dict(county_stats.county .=> zip(county_stats.mu_ln, county_stats.sigma_ln))
end

## group by destination, split into workplaces, assign origins
## modifies count_matrix in place
## returns a dict of workplace ids => employee cbgs
function generate_workplaces!(count_matrix::Matrix{Int64}, dest_idx::Dict{String, Int64}, 
    origin_labels::Vector{K}, county_stats::Dict{String7, Tuple{Float64, Float64}}, inc_seg::Bool=true) where K<:Any

    if inc_seg ## generate income-segregated workplaces
        dConfig = tryJSON("config.json")
        p_mixed_workers::Float64 = get(dConfig, "p_workers_in_mixed_workplaces", 0.5) ## places a certain p workers into mixed workplaces
        wp_types = UInt8[0,1,2]
        ## create 3 separate OD matrices
        counts_mixed = round.(Int, count_matrix .* p_mixed_workers)
        idxs_low = findall(x->x[2]=="inc_low", origin_labels)
        idxs_high = findall(x->x[2]=="inc_high", origin_labels)
        remaining = count_matrix .- counts_mixed
        counts_low = zeros(Int, size(remaining))
        counts_low[idxs_low,:] .= remaining[idxs_low,:]
        counts_high = zeros(Int, size(remaining))
        counts_high[idxs_high,:] .= remaining[idxs_high,:]
        matrices = [counts_mixed, counts_low, counts_high]
    else
        wp_types = UInt8[0]
        matrices = [count_matrix]
    end

    work_origins = Dict{WRKkey, Vector{K}}()
    counties = unique(map(x->x[1:5], collect(keys(dest_idx))))
    for co in counties
        ## throw out unused draws between counties
        draws = Vector{Int64}()
        (mu,sigma) = county_stats[co]
        ## work destinations in the county we just read the stats for
        dests = filterk(k->k[1:5]==co, dest_idx)
        for (dest_code, col) in dests
            for (typecode, M) in zip(wp_types, matrices)
                n = sum(M[:,col]) ## number of workers in the dest
                if n > 0
                    sizes = split_lognormal!(n,mu,sigma,draws)
                    for (work_i, emp_size) in enumerate(sizes)
                        ## get rand sample of origins; pass a view so original matrix is modified
                        o_idxs = drawCounts!(view(M,:,col), emp_size)
                        ## create workplace and assign origins sampled
                        work_origins[(work_i, typecode, dest_code)] = origin_labels[o_idxs]
                    end
                end
            end
        end
        println(draws)
    end
    return work_origins
end

## make a separate work destination for each person working outside the synth area
##  (no need to try to group them, as they have no work network anyway)
function generate_outside_workplaces(work_outside::Dict{K,Int}) where K<:Any
    return Dict([WRKkey((i, 0, "outside")) for i in 1:sum(values(work_outside))]  
        .=> map(x->Vector{K}([x]), reduce(vcat, [fill(k, v) for (k, v) in work_outside])))
end

## people generated in households.jl
function read_workers()
    people = dser_path("jlse/people.jlse")
    ## just commuters
    return collect(keys(filterv(x->x.commuter, people)))
end

function read_workers_by_inc()
    people = dser_path("jlse/people.jlse")
    ## return a tuple for each person: (person id, hh id, cbg id, commuter income cat)
    ## first 3 come from the person key; for income cat use 1 = low, 2 = high
    ## filter out non-commuters
    return filter(x->x[4]>0, [(k..., UInt8(1*v.com_LODES_low + 2*v.com_LODES_high)) for (k,v) in people])
end

function group_workers_by_cbg(workers::Vector{W}, origin_labels::Vector{K}) where {W,K}
    ## group by cbg, randomize order
    cbgs = dser_path("jlse/cbg_idxs.jlse")
    ##  (dataframe has fast grouping)
    df_by_origin = groupby(DataFrame(workers),"3")
    workers_by_origin = Dict{K, Vector{W}}()
    for gk in keys(df_by_origin)
        cbg_geo = cbgs[gk["3"]]
        workers_by_origin[cbg_geo] = shuffle( [Tuple(r) for r in eachrow(df_by_origin[gk])] )
    end
    return workers_by_origin
end

function group_workers_by_cbg_x_inc(workers::Vector{W}, origin_labels::Vector{K}) where {W,K}
    ## group by cbg and income, randomize order
    cbgs = dser_path("jlse/cbg_idxs.jlse")
    ##  (dataframe has fast grouping)
    df_by_origin = groupby(DataFrame(workers),["3","4"])
    inc_codes = Dict(UInt8(1)=>"inc_low", UInt8(2)=>"inc_high")
    workers_by_origin = Dict{K, Vector{W}}()
    for gk in keys(df_by_origin)
        cbg_geo, i_code = cbgs[gk["3"]], inc_codes[gk["4"]]
        workers_by_origin[(cbg_geo,i_code)] = shuffle( [Tuple(r) for r in eachrow(df_by_origin[gk])] )
    end
    return workers_by_origin
end

function dummy_gen_fn(by_inc::Bool=false)
    dummy_idx = 0
    inc_codes = Dict("inc_low"=>UInt8(1), "inc_high"=>UInt8(2))
    if by_inc
        f = function(origin::Tuple{String15,String15})
            dummy_idx += 1
            ## diagnostics (dummy should only be created when origin is "outside")
            d_err = origin[1] == "outside" ? 0 : 1
            ## dummies have no household or home cbg
            ## we have dummies' income from the commute data; include this in the key
            return (dummy_idx,0,0,get(inc_codes,origin[2],UInt8(0))), d_err
        end
    else
        f = function(origin::String15)
            dummy_idx += 1
            ## diagnostics (dummy should only be created when origin is "outside")
            d_err = origin == "outside" ? 0 : 1
            ## dummies have no household or home cbg
            return (dummy_idx,0,0), d_err
        end
    end
    return f
end

## assigns people in workers_by_origin to employers in emp_origins
## modifies cidx_by_origin so that it can be called with several emp_origins on the same pop
## creates "dummies" -- workers from outside that don't exist in people data
## returns a dict of employer id => vector of worker ids, and some diagnostics
function assign_workers!(emp_origins::Dict{T,Vector{K}}, workers_by_origin::Dict{K, Vector{W}}, 
    cidx_by_origin::Dict{K, Int64}, dummy_fn::F) where {T<:Any,K<:Any,W<:Any,F<:Function}

    n_by_origin = Dict(k => length(v) for (k,v) in workers_by_origin)
    ## initialize empty lists
    workers = Dict{T, Vector{W}}()
    for est_id in keys(emp_origins)
        workers[est_id] = Vector{W}()
    end
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
                dum::W, d_err::Int = dummy_fn(origin_key)
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
    dConfig = tryJSON("config.json")
    inc_seg = Bool(get(dConfig, "income_segregated_workplaces", 1))

    ## read commute matrix
    filename = inc_seg ? "processed/work_od_matrix.csv" : "processed/work_od_matrix_no_inc.csv"
    od_matrix = read_df(filename; types=Dict("h_cbg"=>String15))
    if inc_seg
        od_start_col = 3
        origin_labels = collect(zip(od_matrix.h_cbg,Vector{String15}(od_matrix[:,2]))) ## row labels are a tuple of the first two columns (cbg, income_label)
    else
        od_start_col = 2
        origin_labels = od_matrix.h_cbg ## row labels are cbgs
    end
    od_idx = Dict(origin_labels .=> axes(od_matrix,1)) ## for easy lookup
    dest_labels = names(od_matrix)[od_start_col:end-1] ## save column headings, except last one which is "outside"
    dest_idx = Dict(dest_labels .=> eachindex(dest_labels)) ## for easy lookup
    
    ## read # workers from hh sample summary, add workers living in gq's
    fn_read_wrk = inc_seg ? read_workers_cbg_x_inc : read_workers_by_cbg
    fn_read_gq = inc_seg ? read_gq_workers_by_inc : read_gq_workers
    total_w_by_origin = mergewith(+, fn_read_wrk(origin_labels), fn_read_gq())
    println("# workers living in synth area = ", sum(values(total_w_by_origin)))

    ## multiply workers by od matrix
    ## (no need to worry about "secret jobs", which are only missing from the OD raw data)
    od_counts = allocate_workers(Matrix{Float64}(od_matrix[!,od_start_col:end]), od_idx, total_w_by_origin)
    println("# allocated to destinations = ", sum(od_counts))

    ## pull off the last column (destination outside the synth area)
    ## -- can't generate workplaces for these
    work_outside_counts = od_counts[:,end]
    od_counts = od_counts[:,1:end-1]

    ## read workers living outside synth area
    ## append to bottom of od_counts, following same order as existing matrix
    last_inside_idx = lastindex(origin_labels) ## save idx before modifying
    if inc_seg
        live_outside = read_outside_by_inc()
        od_counts = vcat(od_counts, 
                transpose([get(live_outside, (k,"inc_low"), 0) for k in dest_labels]), 
                transpose([get(live_outside, (k,"inc_high"), 0) for k in dest_labels]))
        push!(origin_labels, ("outside","inc_low"), ("outside","inc_high")) ## add labels for bottom rows
    else
        live_outside = read_outside_origins()
        od_counts = vcat(od_counts, transpose([get(live_outside, k, 0) for k in dest_labels]))
        push!(origin_labels, "outside") ## add label for bottom row   
    end
    println("# jobs within synth area = ", sum(od_counts))

    ## pull out workers for schools and gqs for each destination (except "outside")
    ##   (assumes workers from out of state can work in public schools)
    ##   samples worker origins in proportion to counts, without special consideration for income
    ##   (at schools and gqs, high/low income workers are mixed together in the same proportion as the destination's overall workforce) 
    sch_emp_origins = pull_teachers!(od_counts, dest_idx, origin_labels)
    gq_emp_origins = pull_gq_workers!(od_counts, dest_idx, origin_labels)

    ## create workplaces for remaining workers for each destination (except "outside")
    ## and assign origins to those
    ## mean and sd of ln(workplace size) for each county
    county_stats = read_county_stats()
    ## adjust stats
    ##  slightly overproduces large employers compared to CBP stats
    ##  (but unlisted employers should probably skew large)
    stats2 = deepcopy(county_stats)
    for (k,v) in stats2
        stats2[k] = (v[1], v[2]+0.1)
    end

    work_origins = generate_workplaces!(od_counts, dest_idx, origin_labels, stats2, inc_seg)

    ## by cbg, # people working outside the synth area
    ## could create a separate dummy workplace for each person?
    ##  (no reason to group them; they will get infected at work randomly, not through the network)
    work_outside = Dict(origin_labels[1:last_inside_idx] .=> work_outside_counts)

    ## read synthetic people generated in households.jl
    ## workers have income code appended to person key; will make it easier to build assortative network
    workers = inc_seg ? read_workers_by_inc() : read_workers()

    ## check totals
    println("# ppl working within synth area = ", length(workers) - sum(work_outside_counts) + sum(values(live_outside)))
    println("# jobs generated in synth area = ", (sum(length.(values(work_origins))) + 
        sum(length.(values(sch_emp_origins))) + 
        sum(length.(values(gq_emp_origins))))) ## jobs inside synth area

    ## group people by origin, shuffle, and assign to origins
    fn_grp_wrk = inc_seg ? group_workers_by_cbg_x_inc : group_workers_by_cbg
    workers_by_origin = fn_grp_wrk(workers, origin_labels)

    ## instead of removing workers as they're assigned, just keep pointers to the first available worker
    cidx_by_origin = Dict(keys(workers_by_origin) .=> 0)
    dummy_fn = dummy_gen_fn(inc_seg)

    ## assign workers from synth pop to employer origins
    (sch_workers, sch_dummies, missing_origins, ran_out) = assign_workers!(sch_emp_origins, workers_by_origin, cidx_by_origin, dummy_fn);
    println("# workers assigned to schools = ",sum(length.(values(sch_workers))))
    (gq_workers, gq_dummies, missing_origins, ran_out) = assign_workers!(gq_emp_origins, workers_by_origin, cidx_by_origin, dummy_fn);
    println("# workers assigned to group quarters = ",sum(length.(values(gq_workers))))
    (company_workers, company_dummies, missing_origins, ran_out) = assign_workers!(work_origins, workers_by_origin, cidx_by_origin, dummy_fn);
    println("# workers assigned to 'companies' = ",sum(length.(values(company_workers))))
    ## check results
    println("total workers assigned = " ,(sum(length.(values(company_workers))) + 
        sum(length.(values(gq_workers))) + 
        sum(length.(values(sch_workers)))))

    ## don't forget to assign people to work_outside origins
    ##
    ## you weren't going to forget, were you?
    ## TODO (maybe) push ~50k work-from-home ppl to the end so they get assigned outside
    ##   (otherwise could be too many wfh in the in-area workplaces)
    n_by_origin = Dict(k => length(v) for (k,v) in workers_by_origin)
    unused = Dict(k => max(0,(v - cidx_by_origin[k])) for (k,v) in n_by_origin)
    println("# remaining workers = ", sum(values(unused)))

    (outside_workers, _, missing_origins, ran_out) = assign_workers!(generate_outside_workplaces(work_outside), workers_by_origin, cidx_by_origin, dummy_fn);
    ## check results
    println("# jobs created outside synth area = ", sum(length.(values(outside_workers))))

    ## save to file for next step
    suffix = "" # inc_seg ? "" : "_no_inc"
    ser_path("jlse/work_dummies"*suffix*".jlse",[sch_dummies; gq_dummies; company_dummies])
    ser_path("jlse/sch_workers"*suffix*".jlse",sch_workers)
    ser_path("jlse/gq_workers"*suffix*".jlse",gq_workers)
    ser_path("jlse/company_workers"*suffix*".jlse",company_workers)
    ser_path("jlse/outside_workers"*suffix*".jlse",outside_workers)

    # performance testing
    println("generated establishment sizes (other than schools and group quarters):")
    println("size<5: ",length(filterv(x->length(x)<5, work_origins)))
    println("4<size<10: ",length(filterv(x->(4<length(x)<10), work_origins)))
    println("9<size<20: ",length(filterv(x->(9<length(x)<20), work_origins)))
    println("19<size<50: ",length(filterv(x->(19<length(x)<50), work_origins)))
    println("49<size<100: ",length(filterv(x->(49<length(x)<100), work_origins)))
    println("99<size<250: ",length(filterv(x->(99<length(x)<250), work_origins)))
    println("249<size<500: ",length(filterv(x->(249<length(x)<500), work_origins)))
    println("499<size<1000: ",length(filterv(x->(499<length(x)<1000), work_origins)))
    println("999<size<1500: ",length(filterv(x->(999<length(x)<1500), work_origins)))
    println("1499<size<2500: ",length(filterv(x->(1499<length(x)<2500), work_origins)))
    println("2499<size<5000: ",length(filterv(x->(2499<length(x)<5000), work_origins)))
    println("size>4999: ",length(filterv(x->(length(x)>4999), work_origins)))

    return nothing
end
