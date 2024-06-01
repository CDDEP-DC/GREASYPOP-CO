#=
Copyright 2023 Alexander Tulchinsky

This file is part of Greasypop.

Greasypop is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Greasypop is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with Greasypop. If not, see <https://www.gnu.org/licenses/>.
=#

include("utils.jl")
include("fileutils.jl")

mkpath("pop_export")

println("households and people")

d = dser_path("jlse/cbg_idxs.jlse")
df = DataFrame(sort([
            (cbg_id=Int64(k), cbg_geocode=String(v))
            for (k,v) in d
            ], by=x->x.cbg_id))
write_df("pop_export/cbg_idxs.csv",df)

d = dser_path("jlse/hh.jlse")
df = DataFrame(sort([
            (hh_id=Int(k[1]), cbg_id=Int(k[2]), sample_index=Int(v.sample), n_people=length(v.people))
            for (k,v) in d
            ], by=x->(x.cbg_id,x.hh_id)))
write_df("pop_export/hh.csv",df)

d = dser_path("jlse/people.jlse")
df = DataFrame(sort([
            (p_id=Int(k[1]), hh_id=Int(k[2]), cbg_id=Int(k[3]), sample_index=mcon(Int,v.sample),
            age=mcon(Int,v.age), female=mcon(Int,v.female), working=mcon(Int,v.working), commuter=mcon(Int,v.commuter), 
            commuter_income_category=mcon(Int,v.com_inc) , commuter_workplace_category=mcon(Int,v.com_cat),
			race_black_alone=mcon(Int,v.race_black_alone), white_non_hispanic=mcon(Int,v.white_non_hispanic), hispanic=mcon(Int,v.hispanic),
            sch_grade=mcon(String,v.sch_grade))
            for (k,v) in d
            ], by=x->(x.cbg_id,x.hh_id,x.p_id)))
write_df("pop_export/people.csv",df)

d = nothing
GC.gc()

println("schools")

d = dser_path("jlse/sch_students.jlse")
df = DataFrame(sort([
            (sch_code=String(k), p_id=Int(v[1]), hh_id=Int(v[2]), cbg_id=Int(v[3]))
            for (k,v) in dflat(d)
            ], by=x->x.sch_code))
write_df("pop_export/sch_students.csv",df)

d = dser_path("jlse/sch_workers.jlse")
df = DataFrame(sort([
            (sch_code=String(k), p_id=Int(v[1]), hh_id=Int(v[2]), cbg_id=Int(v[3]))
            for (k,v) in dflat(d)
            ], by=x->x.sch_code))
write_df("pop_export/sch_workers.csv",df)

println("group quarters")

d = dser_path("jlse/gqs.jlse")
df = DataFrame(sort([
            (gq_id=Int(k[1]), cbg_id=Int(k[2]), gq_type=String(v.type), n_residents=length(v.residents))
            for (k,v) in d
            ], by=x->(x.cbg_id,x.gq_id)))
write_df("pop_export/gqs.csv",df)

d = Dict(k=>v.residents for (k,v) in d)
df = DataFrame(sort([
            (gq_id=Int(k[1]), cbg_id=Int(k[2]), p_id=Int(v[1]), hh_id=Int(v[2]))
            for (k,v) in dflat(d)
            ], by=x->(x.cbg_id,x.gq_id)))
write_df("pop_export/gq_residents.csv",df)

println("workplaces")

d = dser_path("jlse/gq_workers.jlse")
df = DataFrame(sort([
            (gq_id=Int(k[1]), gq_cbg_id=Int(k[2]), p_id=Int(v[1]), p_hh_id=Int(v[2]), p_cbg_id=Int(v[3]))
            for (k,v) in dflat(d)
            ], by=x->(x.gq_cbg_id,x.gq_id)))
write_df("pop_export/gq_workers.csv",df)

d = dser_path("jlse/company_workers.jlse")
df = DataFrame(sort([
            (employer_geo_code=String(k[3]), employer_type=Int(k[2]), employer_num=Int(k[1]), p_id=Int(v[1]), p_hh_id=Int(v[2]), p_cbg_id=Int(v[3]))
            for (k,v) in dflat(d)
            ], by=x->(x.employer_geo_code,x.employer_type,x.employer_num)))
write_df("pop_export/company_workers.csv",df)

d = dser_path("jlse/outside_workers.jlse")
df = DataFrame(sort([
            (p_id=Int(v[1]), p_hh_id=Int(v[2]), p_cbg_id=Int(v[3]))
            for (k,v) in dflat(d)
            ], by=x->(x.p_cbg_id,x.p_hh_id,x.p_id)))
write_df("pop_export/outside_workers.csv",df)

d = nothing
GC.gc()

println("done")
