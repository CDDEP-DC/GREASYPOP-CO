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
                for x in hhvec ## add number of workers from each household
                    n_work = hh_samps[hh_idx[x],:has_job]
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

## workers living in group quarters, calculated in households.jl
function read_gq_workers()
    gq_df = dser_path("jlse/df_gq_summary.jlse")
    gq_workers_by_cbg = Dict{String15,Int64}()
    for r in eachrow(gq_df)
        if r.working > 0
            gq_workers_by_cbg[String15(r.geo)] = r.working
        end
    end
    return gq_workers_by_cbg
end

## allocate workers to destinations for each cbg, based on commute matrix
function allocate_workers(od_matrix::Matrix{Float64}, od_idx::Dict{String15, Int64}, workers_by_cbg::Dict{String15, Int64})
    od_counts = zeros(Int64, size(od_matrix))
    for (code, rownum) in od_idx 
        ## note, od matrix contains cbgs that we didn't create; e.g, not enough households
        if haskey(workers_by_cbg, code)
            od_counts[rownum,:] = lrRound(od_matrix[rownum,:] .* workers_by_cbg[code])
        end    
    end
    return od_counts
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
## return dict of school id => employee cbgs
function pull_teachers!(count_matrix::Matrix{Int64}, dest_idx::Dict{String, Int64}, origin_labels::Vector{String15})
    ## find the closest cbg for each school
    distmat = read_df("processed/cbg_sch_distmat.csv"; types=Dict("GEOID"=>String15))
    closest_cbg = Dict(x => string(distmat[argmin(distmat[!,x]),"GEOID"]) for x in names(distmat))

    sch_n_teachers = read_sch_teachers()
    sch_emp_origins = Dict{String15,Vector{String15}}()
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
## return dict of gq id => employee cbgs
function pull_gq_workers!(count_matrix::Matrix{Int64}, dest_idx::Dict{String, Int64}, origin_labels::Vector{String15})
    ## find cbg where each gq is located
    gqs = dser_path("jlse/gqs.jlse")
    cbgs = dser_path("jlse/cbg_idxs.jlse")
    gq_cbgs = Dict(k => cbgs[k[2]] for k in keys(gqs))
    ## inst qg's: assume 1 workers per 10 residents; non-inst, 1 per 50? mininum 2?
    dConfig = tryJSON("config.json")
    r_i = get(dConfig, "inst_res_per_worker", 10)
    r_ni = get(dConfig, "noninst_res_per_worker", 50)
    e_min = get(dConfig, "min_gq_workers", 2)
    
    gq_n_emps = Dict(k => 
        max(e_min, ceil(Int64, v.type == :noninst1864 ? length(v.residents)/r_ni : length(v.residents)/r_i)) 
        for (k,v) in gqs)
    
    gq_emp_origins = Dict{GQkey,Vector{String15}}()
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

## employer size stats by county
function read_county_stats()
    cols = Dict("county"=>String7,"mu_ln"=>Float64,"sigma_ln"=>Float64)
    county_stats = read_df("processed/work_sizes.csv"; select=collect(keys(cols)), types=cols)
    return Dict(county_stats.county .=> zip(county_stats.mu_ln, county_stats.sigma_ln))
end

## group by destination, split into workplaces, assign origins
## modifies count_matrix in place
## returns a dict of workplace ids => employee cbgs
function generate_workplaces!(count_matrix::Matrix{Int64}, dest_idx::Dict{String, Int64}, origin_labels::Vector{String15}, county_stats::Dict{String7, Tuple{Float64, Float64}})
    work_origins = Dict{WRKkey, Vector{String15}}()

    counties = unique(map(x->x[1:5], collect(keys(dest_idx))))
    for co in counties
        ## throw out unused draws between counties
        draws = Vector{Int64}()
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
                    work_origins[(work_i, dest_code)] = origin_labels[o_idxs]
                end
            end
        end
        println(draws)
    end
    return work_origins
end

## people generated in households.jl
function read_workers()
    people = dser_path("jlse/people.jlse")
    ## just working people
    return filterv(x->x.working, people)    
end

## assigns people in workers_by_cbg to employers in emp_origins
## modifies cidx_by_cbg so that it can be called with several emp_origins on the same pop
## returns a dict of employer id => vector of person ids, and some diagnostics
function assign_workers!(emp_origins::Dict{T,Vector{String15}}, 
    workers_by_cbg::Dict{String15, Vector{Pkey}}, 
    cidx_by_cbg::Dict{String15, Int64}) where T

    n_by_cbg = Dict(k => lastindex(v) for (k,v) in workers_by_cbg)
    workers = Dict{T, Vector{Pkey}}()
    for est_id in keys(emp_origins)
        workers[est_id] = Vector{Pkey}()
    end
    
    dummies = 0
    missing_origin = 0
    ran_out = Dict{String15,Int64}()
    for (e_id, origins) in emp_origins
        for h_cbg in origins
            if h_cbg == "outside"
                dummies += 1
                i = cidx_by_cbg["outside"] += 1
                push!(workers[e_id], (i,0,0)) ## dummies have no household or home cbg
            else
                if haskey(cidx_by_cbg, h_cbg)
                    i = cidx_by_cbg[h_cbg] += 1 ## this doesn't look like it should be allowed, but it is lol
                    if i > n_by_cbg[h_cbg]
                        ran_out[h_cbg] = get!(ran_out, h_cbg, 0) + 1 ## this shouldn't happen
                    else
                        push!(workers[e_id], (workers_by_cbg[h_cbg][i]))
                    end
                else
                    missing_origin += 1
                end
            end
        end
    end
    return (workers, dummies, missing_origin, ran_out)
end


function generate_jobs_and_workers()
    ## read commute matrix
    od_matrix = read_df("processed/work_od_matrix.csv"; types=Dict("h_cbg"=>String15))
    origin_labels = od_matrix.h_cbg ## save row labels
    origin_idx = Dict(od_matrix.h_cbg .=> eachindex(od_matrix.h_cbg)) ## for easy lookup
    dest_labels = names(od_matrix)[2:end-1] ## save column headings, except last one which is "outside"
    dest_idx = Dict(dest_labels .=> eachindex(dest_labels)) ## for easy lookup
    ## convert to matrix for easier math
    od_matrix = Matrix{Float64}(od_matrix[!,2:end])

    ## read # workers from hh sample summary
    hh_workers_by_cbg = read_workers_by_cbg(origin_labels)

    ## add workers living in gq's
    gq_workers_by_cbg = read_gq_workers()
    tt_workers_by_cbg = mergewith(+,hh_workers_by_cbg,gq_workers_by_cbg)
    println("# workers living in synth area = ", sum(values(tt_workers_by_cbg)))

    ## multiply workers by od matrix
    ## (no need to worry about "secret jobs", which are only missing from the OD raw data)
    od_counts = allocate_workers(od_matrix, origin_idx, tt_workers_by_cbg)
    println("# allocated to destinations = ", sum(od_counts))

    ## pull off the last column (destination outside the synth area)
    ## -- can't generate workplaces for these
    work_outside_counts = od_counts[:,end]
    od_counts = od_counts[:,1:end-1]

    ## read workers living outside synth area
    live_outside = read_outside_origins()
    ## append to bottom of od_counts
    od_counts = vcat(od_counts, transpose([get(live_outside, k, 0) for k in dest_labels]))
    push!(origin_labels, "outside") ## add label for bottom row
    println("# jobs within synth area = ", sum(od_counts))

    ## pull out workers for schools and gqs for each destination (except "outside")
    ##   (assumes workers from out of state can work in public schools)
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

    work_origins = generate_workplaces!(od_counts, dest_idx, origin_labels, stats2)

    ## by cbg, # people working outside the synth area
    ## could create a separate dummy workplace for each person?
    ##  (no reason to group them; they will get infected at work randomly, not through the network)
    work_outside = Dict(origin_labels[1:end-1] .=> work_outside_counts)

    ## group people by cbg, shuffle, and assign to origins
    cbgs = dser_path("jlse/cbg_idxs.jlse")

    ## read synthetic people generated in households.jl
    workers = read_workers()

    ## check totals
    println("# ppl working within synth area = ", length(workers) - sum(work_outside_counts) + sum(values(live_outside)))
    println("# jobs generated in synth area = ", (sum(length.(values(work_origins))) + 
        sum(length.(values(sch_emp_origins))) + 
        sum(length.(values(gq_emp_origins))))) ## jobs inside synth area

    ## group by cbg, randomize order
    ##  (dataframe has fast grouping)
    df_by_cbg = groupby(DataFrame(keys(workers)),"3")
    workers_by_cbg = Dict{String15, Vector{Pkey}}()
    for gk in keys(df_by_cbg)
        cbg_idx = gk["3"]
        cbg_geo = cbgs[cbg_idx]
        workers_by_cbg[cbg_geo] = shuffle(Tuple.(eachrow(df_by_cbg[gk])))
    end

    ## instead of removing workers as they're assigned, just keep pointers to the first available worker
    cidx_by_cbg = Dict(keys(workers_by_cbg) .=> 0)
    ## allow sequential numbering of workers living outside the synth area
    cidx_by_cbg[String15("outside")] = 0

    ## assign workers from synth pop to employer origins
    (sch_workers, dummies, missing_origins, ran_out) = assign_workers!(sch_emp_origins, workers_by_cbg, cidx_by_cbg);
    println("# workers assigned to schools = ",sum(length.(values(sch_workers))))
    (gq_workers, dummies, missing_origins, ran_out) = assign_workers!(gq_emp_origins, workers_by_cbg, cidx_by_cbg);
    println("# workers assigned to group quarters = ",sum(length.(values(gq_workers))))
    (company_workers, dummies, missing_origins, ran_out) = assign_workers!(work_origins, workers_by_cbg, cidx_by_cbg);
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
    n_by_cbg = Dict(k => length(v) for (k,v) in workers_by_cbg)
    unused = Dict(k => max(0,(v - cidx_by_cbg[k])) for (k,v) in n_by_cbg)
    println("# remaining workers = ", sum(values(unused)))

    ## make a separate work destination for each person working outside the synth area
    ##  (probably no need to try to group them, as they have no work network anyway)
    (outside_workers, dummies, missing_origins, ran_out) = assign_workers!(
        Dict([WRKkey((i, "outside")) for i in 1:sum(values(work_outside))]  
                .=> map(x->Vector{String15}([x]), reduce(vcat, [fill(k, v) for (k, v) in work_outside]))),
        workers_by_cbg, cidx_by_cbg)

    ## check results
    println("# jobs created outside synth area = ", sum(length.(values(outside_workers))))

    ## save to file for next step
    ser_path("jlse/sch_workers.jlse",sch_workers)
    ser_path("jlse/gq_workers.jlse",gq_workers)
    ser_path("jlse/company_workers.jlse",company_workers)
    ser_path("jlse/outside_workers.jlse",outside_workers)

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
