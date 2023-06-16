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
function read_psamp_df()
    p_samp_sel = ["SERIALNO","AGEP","SEX","commuter","has_job","job_listed","sch_grade"]
    return read_df("processed/p_samples.csv"; select=p_samp_sel, types=Dict("SERIALNO"=>String15))
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

## assume 4 gq's max per cbg:
## -- u18 inst, 18-64 inst, 18-64 noninst, 65+ inst
## -- ignore noninst for u18 and 65+
## -- ignore gq with less than 20 people
function row_gq_pops(r::DataFrameRow)
    instu18 = int(r["group quarters:under 18"] * r["p_u18_inst"])
    inst1864 = int(r["group quarters:18 to 64"] * r["p_18_64_inst"])
    noninst1864 = int(r["group quarters:18 to 64"] * (1.0 - r["p_18_64_inst"]))
    inst65o = int(r["group quarters:65 and over"] * r["p_65o_inst"])

    dConfig = tryJSON("config.json")
    r_m = get(dConfig, "min_gq_residents", 20)

    return [instu18 >= r_m ? instu18 : 0, 
        inst1864 >= r_m ? inst1864 : 0, 
        noninst1864 >= r_m ? noninst1864 : 0, 
        inst65o >= r_m ? inst65o : 0]
end

## assume only 18-64 noninst have jobs
function row_gq_employment(n::Int64, havejobs::Bool, r::DataFrameRow)
    if havejobs
        n_com_list = int(n * r["p_com_list|noninst1864"])
        n_com_ulist = int(n * r["p_com_ulist|noninst1864"])
        n_wfh_list = int(n * r["p_wfh_list|noninst1864"])
        n_wfh_ulist = int(n * r["p_wfh_ulist|noninst1864"])
        n_not_working = int(n) - (n_com_list + n_com_ulist + n_wfh_list + n_wfh_ulist)
        return (n_com_list, n_com_ulist, n_wfh_list, n_wfh_ulist, n_not_working)
    else
        return (0, 0, 0, 0, n)
    end
end

## generate employment fields based on # in each category
function employment_values(cl,cu,wl,wu,nw)
    vcl = [trues(cl); falses(cu); falses(wl); falses(wu); falses(nw)]
    vcu = [falses(cl); trues(cu); falses(wl); falses(wu); falses(nw)]
    vwl = [falses(cl); falses(cu); trues(wl); falses(wu); falses(nw)]
    vwu = [falses(cl); falses(cu); falses(wl); trues(wu); falses(nw)]

    vw = vcl .| vcu .| vwl .| vwu ##working=true
    vc = vcl .| vcu ##commuter=true
    vl = Vector{Union{Missing,Bool}}(vcl .| vwl) ##job_listed=true
    vl[.!vw] .= missing ##job_listed should be unknown when not working
    return (vw, vc, vl)
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

## summary will be used to assign ppl in gq to jobs
function gq_summary(cbg_code, df_gq, df_idx)
    if haskey(df_idx, cbg_code)
        r = df_gq[df_idx[cbg_code],:]
        (instu18, inst1864, noninst1864, inst65o) = row_gq_pops(r)
        (com_list, com_ulist, wfh_list, wfh_ulist, nw) = row_gq_employment(noninst1864, true, r)
        return (Dict(
            "instu18"=> instu18, "inst1864"=> inst1864,
            "noninst1864"=> noninst1864, "inst65o" => inst65o,
            "commuter" => com_list + com_ulist, "wfh" => wfh_list + wfh_ulist,
            "working" => com_list + com_ulist + wfh_list + wfh_ulist,  "not_working" => nw))
    else
        return nothing
    end
end


## generate people, households, group quarters, based on samples generated in CO.jl
## (also create and store integer key for each cbg)
function generate_people()

    counties = read_counties()
    hh_idx = read_hh_serials()
    p_samps = read_psamp_df()
    p_idx = people_by_serial(p_samps)

    ## by cbg
    ## create a unique hh id for each hh, link it to corresponding row in hh_samps
    ## create person id for each person in hh, link it to the unique hh id and to row in p_samps
    ## link hh id to all person ids

    cbgs = Dict{CBGkey, String15}()
    households = Dict{Hkey, Household}()
    people = Dict{Pkey, PersonData}()
    cbg_i = 0

    for c in counties
        cbg_hhs = dser_path("jlse/CO/hh"*c*".jlse");

        for (cbg_code, hh_vec) in cbg_hhs
            cbg_i += 1
            cbgs[cbg_i] = cbg_code
            for (hh_i, hh_serial) in enumerate(hh_vec)
                hh_key = (hh_i, cbg_i)
                p_vec = p_idx[hh_serial]
                for (p_i, r) in enumerate(p_vec)
                    people[(p_i,hh_i,cbg_i)] = PersonData(hh_key, r, 
                        p_samps[r,:AGEP], 
                        (p_samps[r,:SEX]==2), 
                        (p_samps[r,:has_job]==1), 
                        (p_samps[r,:commuter]==1), 
                        (p_samps[r,:job_listed]==1), 
                        p_samps[r,:sch_grade])
                end
                households[hh_key] = Household(hh_idx[hh_serial], [(i,hh_i,cbg_i) for i in eachindex(p_vec)])
            end
        end    
    end

    ## loop through gq, create individuals
    ## -- assume only 18-64 noninst have jobs
    ## -- assume children in gq don't attend public schools
    ## will need to add workers to gq, and generate a within-org network

    df_gq = read_df("processed/group_quarters.csv"; types=Dict("Geo"=>String15))
    gqs = Dict{GQkey, GQres}()
    gq_types = [:instu18, :inst1864, :noninst1864, :inst65o]
    assumed_ages = [15, 30, 30, 75] ## ages unknown (could generate these if needed)
    havejobs = [false,false,true,false] ## assume only 18-64 noninst have jobs

    ## for associating gq's with existing cbgs
    cbg_rev = Dict([v=>k for (k,v) in cbgs])

    for r in eachrow(df_gq)

        cbg_code = r.Geo
        ## group quarters includes some cbgs without households, need to add them
        if haskey(cbg_rev, cbg_code)
            cbg_index = cbg_rev[cbg_code]
        else
            cbg_i += 1
            cbgs[cbg_i] = cbg_code
            cbg_rev[cbg_code] = cbg_i
            cbg_index = cbg_i
        end

        gq_pops = row_gq_pops(r) ## returns 0 if below threshhold
        p_idxs = ranges(gq_pops)
        emp_stats = map((a,b)->row_gq_employment(a,b,r), gq_pops, havejobs)

        for (t_idx, t_code) in enumerate(gq_types)    
            if gq_pops[t_idx] > 0 
                ## use 0 as hh index for gq people (so looking up household will return nothing)
                pkeys = [(p_i,0,cbg_index) for p_i in p_idxs[t_idx]]
                ## create group quarter entry and add to dict
                gqs[(t_idx,cbg_index)] = GQres(gq_types[t_idx], pkeys)
                ## "..." expands a tuple to multiple function args
                vWorking,vCommuter,vListed = employment_values(emp_stats[t_idx]...)
                ## create people and add to dict
                for (i,k) in enumerate(pkeys)
                    ## using 0 for household index and sample#; age and sex unknown (could generate these if needed)
                    people[k] = PersonData((0,cbg_index), 0, assumed_ages[t_idx], missing, 
                                vWorking[i], vCommuter[i], vListed[i], missing)
                end
            end
        end
    end

    ## write to files for next step
    ser_path("jlse/cbg_idxs.jlse", cbgs)
    ser_path("jlse/hh.jlse", households)
    ser_path("jlse/gqs.jlse", gqs)
    ser_path("jlse/people.jlse", people)

    ## create summary statistics for all gq's; will be used to assign ppl in gq to jobs
    gq_idx = Dict(df_gq.Geo .=> eachindex(df_gq.Geo))
    df_gq_summary = DataFrame(geo = String15[],
                    instu18 = Int64[], inst1864 = Int64[], noninst1864 = Int64[], inst65o = Int64[],
                    working = Int64[],not_working = Int64[],commuter = Int64[],wfh = Int64[])
    for c in df_gq.Geo
        push!(df_gq_summary, gq_summary(c, df_gq, gq_idx); cols=:subset)
    end
    df_gq_summary.geo = df_gq.Geo
    ser_path("jlse/df_gq_summary.jlse", df_gq_summary)

    return nothing
end

