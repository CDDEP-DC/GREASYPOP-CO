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

function read_counties()
    cbg_geo_cols = Dict("Geo"=>String15,"st_puma"=>String7,"cbsa"=>String7,"county"=>String7,"R"=>Float64,"U"=>Float64)
    cbg_geo = read_df("processed/cbg_geo.csv"; select=collect(keys(cbg_geo_cols)), types=cbg_geo_cols)
    return unique(cbg_geo.county)
end

## for looking up row # by hh serial
function read_hh_serials()
    hh_samps = read_df("processed/hh_samples.csv"; types=Dict("SERIALNO"=>String15))
    return Dict(hh_samps.SERIALNO .=> eachindex(hh_samps.SERIALNO))
end

## read person samples into a dataframe
function read_psamp_df(ind_codes::Vector{<:AbstractString}, additional_traits::Vector{<:AbstractString})
    nonbool_cols = ["SERIALNO","AGEP","sch_grade"]
    bool_cols = ["commuter","has_job","com_LODES_low","com_LODES_high"]
    ind_cols = ["ind_"*k for k in ind_codes]
    typedict = Dict(["SERIALNO"=>String15;
                    [x=>Bool for x in bool_cols];
                    ## additional traits are all bools for now
                    [x=>Bool for x in additional_traits];
                    [x=>Bool for x in ind_cols]
                    ])
    df = read_df("processed/p_samples.csv"; select=[nonbool_cols; bool_cols; additional_traits; ind_cols], types=typedict)
    return df[:, [nonbool_cols; bool_cols; additional_traits; ind_cols]]
end

## for looking up all person samps by hh serial
## (this way is much faster than a search loop through a million serials)
function people_by_serial(p_samps::DataFrame)
    ## make a column with the row #'s, group by serial
    p_samps_by_hh = groupby(insertcols(p_samps, 1, :idx => 1:nrow(p_samps)), :SERIALNO)
    ## Ref() is a trick of the combine function (found this in the dataframes.jl docs)
    df = combine(p_samps_by_hh, :idx => Ref => :idxs)
    return Dict(df.SERIALNO .=> collect.(df.idxs))
end

## returns a vector of number employed by industry index from dataframe row r
function row_gq_employment(n::Integer, jobtype::Symbol, ind_codes::Vector{<:AbstractString}, r::DataFrameRow)::Vector{Int}
    if jobtype == :none || n < 1
        return [0 for x in ind_codes]
    else
        pref = Dict(:civ=>"civ_ind_",:mil=>"mil_ind_")
        return int.([r[pref[jobtype]*k] for k in ind_codes])
    end
end

## for evaluating results
function cbg_summary(cbg_code, cbg_dict, hh_samps, hh_idx)
    hhvec = cbg_dict[cbg_code]
    cbg_samps = empty(hh_samps)
    for x in hhvec
        push!(cbg_samps, hh_samps[hh_idx[x],:])
    end
    return combine(cbg_samps, Not(:SERIALNO) .=> x->sum(skipmissing(x)); renamecols=false)[1,:]
end


function generate_group_quarters(dConfig::Dict, cbgs::Dict{<:AbstractString, I}, 
    cbg_indexer::Indexer{I}, ind_codes::Vector{<:AbstractString}) where {I<:Integer}

    min_gq_residents::Int = get(dConfig, "min_gq_residents", 20)
    add_trait_cols::Vector{String} = get(dConfig, "additional_traits", String[])

    ## loop through gq, create individuals
    ## -- assume only 18-64 noninst have jobs
    ## -- assume children in gq don't attend public schools
    ## will need to add workers to gq, and generate a within-org network
    df_gq_cols = ["Geo", "group quarters:", "group quarters:under 18", "group quarters:18 to 64", 
        "group quarters:65 and over", "p_u18_inst", "p_18_64_inst", "p_65o_inst", "p_18_64_noninst_civil", 
        "p_18_64_noninst_mil", "commuter_p|ninst1864civ", "work_from_home_p|ninst1864civ", 
        "com_LODES_low_p|ninst1864civ", "com_LODES_high_p|ninst1864civ", "commuter_p|milGQ", 
        "work_from_home_p|milGQ", "com_LODES_low_p|milGQ", "com_LODES_high_p|milGQ"]

    df_gq = read_df("processed/group_quarters.csv"; select=df_gq_cols, types=Dict("Geo"=>String15))

    ## assume 5 gq's max per cbg:
    ## -- u18 inst, 18-64 inst, 18-64 noninst civilian, 18-64 noninst military, 65+ inst
    ## -- ignore noninst for u18 and 65+
    ## -- ignore gq with less than min_gq_size people
    gq_types = [:instu18, :inst1864, :ninst1864civ, :milGQ, :inst65o]
    assumed_ages = [15, 30, 30, 30, 75] ## ages unknown (could generate these if needed)
    job_types = [:none, :none, :civ, :mil, :none] ## assume only 18-64 noninst have jobs
    transform!(df_gq, ["group quarters:under 18","p_u18_inst"] => ((a,b)->thresh.(int.(a.*b),min_gq_residents)) => "pop_instu18")
    transform!(df_gq, ["group quarters:18 to 64","p_18_64_inst"] => ((a,b)->thresh.(int.(a.*b),min_gq_residents)) => "pop_inst1864")
    transform!(df_gq, ["group quarters:18 to 64","p_18_64_noninst_civil"] => ((a,b)->thresh.(int.(a.*b),min_gq_residents)) => "pop_ninst1864civ")
    transform!(df_gq, ["group quarters:18 to 64","p_18_64_noninst_mil"] => ((a,b)->thresh.(int.(a.*b),min_gq_residents)) => "pop_milGQ")
    transform!(df_gq, ["group quarters:65 and over","p_65o_inst"] => ((a,b)->thresh.(int.(a.*b),min_gq_residents)) => "pop_inst65o")

    ## read gq worker counts by industry, which were derived from census data
    df_civil_emp = rename(x->replace(x,"C24030:"=>"civ_ind_","C24010:"=>"civ_occ_"), 
                    read_df("processed/gq_civilian_workers.csv"; types=Dict("Geo"=>String15)))
    df_mil_emp = rename(x->replace(x,"C24030:"=>"mil_ind_","C24010:"=>"mil_occ_"), 
                    read_df("processed/gq_military_workers.csv"; types=Dict("Geo"=>String15)))

    ## not all of those are commuters -- assume p(work from home) is the same across industries
    ## randomly assign some people as wfh; do same for income category
    transform!(df_gq, ["commuter_p|ninst1864civ","work_from_home_p|ninst1864civ"] => ((a,b)->a./(a.+b)) => "commuter_p|civ_worker")
    transform!(df_gq, ["commuter_p|milGQ","work_from_home_p|milGQ"] => ((a,b)->a./(a.+b)) => "commuter_p|mil_worker")
    transform!(df_gq, ["com_LODES_high_p|ninst1864civ","com_LODES_low_p|ninst1864civ"] => ((a,b)->a./(a.+b)) => "LODES_high|civ_commuter")
    transform!(df_gq, ["com_LODES_high_p|milGQ","com_LODES_low_p|milGQ"] => ((a,b)->a./(a.+b)) => "LODES_high|mil_commuter")
    df_gq = innerjoin(df_gq, df_civil_emp, df_mil_emp; on=:Geo)

    gqs = Dict{GQkey, GQres}()
    gq_people = Dict{Pkey, PersonData}()

    for r in eachrow(df_gq)
        ## group quarters includes some cbgs without households, need to add them
        cbg_index = cbg_indexer(cbgs, r.Geo)
        ## pop of each gq type in the cbg (0 if below threshhold)
        gq_pops = [r["pop_"*string(x)] for x in gq_types]
        ## employment category counts; a vector for each gq type
        emp_stats = map((a,b)->row_gq_employment(a,b,ind_codes,r), gq_pops, job_types)
        ## for creating a unique key for each person
        p_idxs = ranges(gq_pops)

        ##
        ## assume that all residents of military quarters work at the quarters
        ## only civilians in non-inst GQs commute to jobs
        ## (this assumption is also in workplaces.jl)
        ##
        commuter_p = Dict(:instu18 => 0.0, :inst1864 => 0.0, 
                        :ninst1864civ => r["commuter_p|civ_worker"], 
                        :milGQ => 0.0, :inst65o => 0.0)

        LODES_high_p = Dict(:instu18 => 0.0, :inst1864 => 0.0, 
                        :ninst1864civ => r["LODES_high|civ_commuter"], 
                        :milGQ => r["LODES_high|mil_commuter"], :inst65o => 0.0)

        for (t_idx, t_code) in enumerate(gq_types)    
            if gq_pops[t_idx] > 0 
                ## use 0 as hh index for gq people (so looking up household will return nothing)
                pkeys = [(p_i,0,cbg_index) for p_i in p_idxs[t_idx]]
                ## create group quarter entry and add to dict
                gqs[(t_idx,cbg_index)] = GQres(t_code, pkeys)
                ## for mapping person index to employment category
                emp_idxs = cumsum(emp_stats[t_idx])
                ## create people and add to dict
                for (i,k) in enumerate(pkeys)
                    ## indices mapped to emp cats based on category counts; missing = no job
                    emp_cat = something(findfirst(x -> x >= i, emp_idxs), missing)
                    has_job = !ismissing(emp_cat)
                    is_commuter = has_job ? (rand() < commuter_p[t_code]) : false
                    ## income category, for commuters only
                    inc_cat = is_commuter ? (rand() < LODES_high_p[t_code] ? 2 : 1) : missing
                    ## employment category is only for assigning commuters to workplaces
                    emp_cat = is_commuter ? emp_cat : missing
                    ## using 0 for household index and sample#; age,sex,race,etc. unknown (could generate these if needed)
                    gq_people[k] = PersonData((0,cbg_index), 0, assumed_ages[t_idx],
                                has_job, is_commuter, emp_cat, inc_cat,
                                missing, [missing for x in add_trait_cols]...)
                end
            end
        end
    end

    ## write summary statistics for all gq's; will be used to assign ppl in gq to jobs
    df_gq_summary = DataFrame([
                    "geo" => String15[]; 
                    [string(x) => Int64[] for x in gq_types];
                    #["civ_ind_"*i => Int64[] for i in ind_codes];
                    #["mil_ind_"*i => Int64[] for i in ind_codes]
                    ["ind_"*i => Int64[] for i in ind_codes]
                    ])

    for c in df_gq.Geo
        row_summ = Dict{String,Any}("geo" => c)
        merge!(row_summ, Dict(string.(gq_types) .=> 0))
        #merge!(row_summ, Dict(["civ_ind_"*i for i in ind_codes] .=> 0))
        #merge!(row_summ, Dict(["mil_ind_"*i for i in ind_codes] .=> 0))
        merge!(row_summ, Dict(["ind_"*i for i in ind_codes] .=> 0))
        cbg_index = cbgs[c]
        for (t_idx, t_code) in enumerate(gq_types)
            gq = get(gqs, (t_idx,cbg_index), missing)
            if !ismissing(gq)
                ppl = [gq_people[k] for k in gq.residents]
                row_summ[string(gq.type)] = length(ppl)
                if gq.type == :ninst1864civ
                    #merge!(row_summ, Dict(["civ_ind_"*i for i in ind_codes] .=> counts([coalesce(x.com_cat,0) for x in ppl], 1:length(ind_codes))))
                    merge!(row_summ, Dict(["ind_"*i for i in ind_codes] .=> counts([coalesce(x.com_cat,0) for x in ppl], 1:length(ind_codes))))
                #elseif gq.type == :milGQ
                #    merge!(row_summ, Dict(["mil_ind_"*i for i in ind_codes] .=> counts([coalesce(x.com_cat,0) for x in ppl], 1:length(ind_codes))))
                end
            end
        end
        push!(df_gq_summary, row_summ)
    end
    ser_path("jlse/df_gq_summary.jlse", df_gq_summary)

    return (cbgs, gqs, gq_people)
end


function generate_people()

    dConfig = tryJSON("config.json")
    additional_traits::Vector{String} = get(dConfig, "additional_traits", String[])
    wp_codes = tryJSON("processed/codes.json")
    ind_codes::Vector{String} = get(wp_codes, "ind_codes", String[])

    counties = read_counties()
    hh_idx = read_hh_serials() ## for linking household to row # in hh samps
    println("reading person samples")
    p_samps = read_psamp_df(ind_codes, additional_traits) ## for looking up person traits from sample data
    p_idx = people_by_serial(p_samps) ## for linking people in households to row #s in p_samps

    ind_colnames = Symbol.(["ind_"*k for k in ind_codes]) ## industry code columns in p_samps
    ind_col_idxs = Dict(ind_colnames .=> eachindex(ind_colnames)) ## assign an integer index to each, in order

    ## pre-compute some traits so we don't have to do it inside the loop
    ## each person has only one industry; append its code and index to each record
    transform!(p_samps, AsTable(ind_colnames) => ByRow(first_true) => "ind_code")
    transform!(p_samps, "ind_code" => ByRow(k->get(ind_col_idxs,k,missing)) => "ind_cat_idx")
    ## this determines job category for commuters
    transform!(p_samps, ["commuter", "ind_cat_idx"] => 
                        ByRow((a,b)-> a ? b : missing) => 
                        "com_cat")

    ## same for income categories (but these already exclude non-commuters)
    income_colnames = [:com_LODES_low, :com_LODES_high]
    income_col_idxs = Dict(:com_LODES_low => 1, :com_LODES_high => 2)
    transform!(p_samps, AsTable(income_colnames) => ByRow(first_true) => "income_code")
    transform!(p_samps, "income_code" => ByRow(k->get(income_col_idxs,k,missing)) => "com_inc")

    add_trait_cols = Symbol.(additional_traits)

    cbgs = Dict{String15, CBGkey}() ## assign an index to each cbg processed
    cbg_indexer = Indexer{CBGkey}()
    households = Dict{Hkey, Household}() ## create a unique hh id for each hh
    people = Dict{Pkey, PersonData}() ## create unique person id for each person
    
    println("generating people")
    for c in counties
        println("county ",c)
        cbg_hhs = dser_path("jlse/CO/hh"*c*".jlse");

        for (cbg_code, hh_vec) in cbg_hhs ## vector of households in each cbg
            cbg_i = cbg_indexer(cbgs, cbg_code)
            for (hh_i, hh_serial) in enumerate(hh_vec)
                hh_key = (hh_i, cbg_i)
                p_vec = p_idx[hh_serial] ## person sample indices in each household
                for (p_i, r) in enumerate(p_vec) ## create each person from data in sample df
                    people[(p_i,hh_i,cbg_i)] = PersonData(
                        hh_key, 
                        r, 
                        p_samps[r,:AGEP], 
                        p_samps[r,:has_job], 
                        p_samps[r,:commuter], 
                        p_samps[r,:com_cat],
                        p_samps[r,:com_inc],
                        p_samps[r,:sch_grade],
                        [p_samps[r,x] for x in add_trait_cols]...)
                end
                households[hh_key] = Household(hh_idx[hh_serial], [(i,hh_i,cbg_i) for i in eachindex(p_vec)])
            end
        end    
    end

    println("generating group quarters")
    cbgs, gqs, gq_people = generate_group_quarters(dConfig, cbgs, cbg_indexer, ind_codes)
    people = merge(gq_people, people)

    ## write to files for next step
    println("writing people to file")
    ser_path("jlse/cbg_idxs.jlse", Dict([v=>k for (k,v) in cbgs]))
    ser_path("jlse/hh.jlse", households)
    ser_path("jlse/gqs.jlse", gqs)
    ser_path("jlse/people.jlse", people)

    return nothing
end

