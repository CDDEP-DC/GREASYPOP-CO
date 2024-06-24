### Geographically REAlistic SYnthetic POPulation using Combinatorial Optimization

![logo](greasypop-logo.png)


* generates a synthetic population (people, households, schools, workplaces) from US census data for a specified region, at census block group (CBG) resolution
* generates a synthetic contact network of regular household, school, and work contacts
* this version groups people into workplaces by industry ( for previous version(s) see Releases --> )

citation: Tulchinsky, A. Y., Haghpanah, F., Hamilton, A., Kipshidze, N., & Klein, E. Y. (2024). Generating geographically and economically realistic large-scale synthetic contact networks: A general method using publicly available data (arXiv:2406.14698). arXiv. http://arxiv.org/abs/2406.14698

## files in this project

    config.json - specify the region for which to generate the synth pop; misc other settings
    census.py - processes input data; converts census and PUMS to a common format; extracts school and workplace data needed for synthesis
	CO.jl - performs combinatorial optimization: selects households for each CBG from microdata samples (PUMS)
	synthpop.jl - script that calls functions for population synthesis from the files below
	households.jl - fills each household from CO.jl with people generated from PUMS data; also creates group quarters (GQ)
	schools.jl - reads school data prepared by census.py and assigns students created in households.jl
	workplaces.jl - creates workplaces based on data from census.py and assigns workers created in households.jl; also assigns teachers to schools and staff to GQ's
	netw.jl - generates synthetic contact network
	utils.jl, fileutils.jl - various utility functions
	export_synthpop.jl - exports synth pop to csv
	export_network.jl - exports contact network to mtx

# how to use

### 1. prepare data

### note: currently only works with data from 2010 - 2019 (format changed in 2020)

## into folder "census"
#### create one sub-folder for each geographic area whose census data you will download; sub-folder names don't matter
#### (if you're only using data from one US state, make one sub-folder for it)
#### into each sub-folder, place the following data tables (from [data.census.gov](https://data.census.gov/))

    (ACS* = ACS 5yr survey, census block group (CBG) level, from year ####)
    (DEC* = decennial census tables from preceding census, having same cbg boundaries)

    ACSDT5Y####.B01001-Data.csv
    ACSDT5Y####.B09018-Data.csv
    ACSDT5Y####.B09019-Data.csv
    ACSDT5Y####.B09020-Data.csv
    ACSDT5Y####.B09021-Data.csv
    ACSDT5Y####.B11004-Data.csv
    ACSDT5Y####.B11012-Data.csv
    ACSDT5Y####.B11016-Data.csv
    ACSDT5Y####.B19001-Data.csv
    ACSDT5Y####.B22010-Data.csv
    ACSDT5Y####.B23009-Data.csv
    ACSDT5Y####.B23025-Data.csv
    ACSDT5Y####.B25006-Data.csv
    ACSDT5Y####.B11001H-Data.csv
    ACSDT5Y####.B11001I-Data.csv
    ACSDT5Y####.C24010-Data.csv
    ACSDT5Y####.C24030-Data.csv
    DECENNIALSF1####.P43-Data.csv


## into folder "pums"
#### PUMS data for the same 5-yr period as ACS
#### from https://www2.census.gov/programs-surveys/acs/data/pums/

    psam_h??.* and psam_p??.*
    for each state you want to draw samples from
    (these are provided inside zip files named csv_h??.zip and csv_p??.zip)

## into folder "geo"
#### from https://www.census.gov/programs-surveys/geography/guidance/geo-areas/pumas.html

    census tract to PUMA relationship file, *Census_Tract_to*PUMA*.*
    (census boundaries were changed in 2020; choose the year corresponding to ACS year)

#### from geocorr https://mcdc.missouri.edu/applications/geocorr2018.html (if using < 2020 ACS)
#### or https://mcdc.missouri.edu/applications/geocorr2022.html (if using >= 2020 ACS)

    puma to county, rename to *puma_to_county*.*
    puma to cbsa, rename to *puma_to_cbsa*.*
    puma to urban-rural portion, rename to *puma_urban_rural*.*
    cbg to cbsa, rename *cbg_to_cbsa*.*
    cbg to urban-rural portion, rename to *cbg_urban_rural*.*

#### cbg lat-long coords from https://www2.census.gov/geo/tiger/TIGER####/BG/ where #### is year

    tl####_??_bg.zip where ?? is the FIPS code for each state in the synth area


## into folder "work"
#### origin-destination work commute data from https://lehd.ces.census.gov/data/
#### use the version that has the same boundaries as the ACS data (v7 for < 2020; v8 for >= 2020)
#### use JT01, "primary" jobs (because JT00 counts 2+ jobs for the same individual)

    main file for every state in the synth area, named *od_main_JT01*.csv.gz
    aux file for every state in the synth area, named *od_aux_JT01*.csv.gz
    if many people from your synth area commute to other states, also get the *aux* file for those states

#### workplace area characteristics (WAC) data from same site

    one file for each state in the synth area, named *wac_S000_JT01*.csv.gz

#### employer size data from https://www.census.gov/programs-surveys/cbp/data/datasets.html
#### 2016 complete county file (more complete than later data)

    cbp16co.zip


## into folder "school"
#### from https://nces.ed.gov/programs/edge/Geographic/SchoolLocations

    school locations: EDGE_GEOCODE_PUBLICSCH_*.xlsx
    GIS data: folder Shapefile_SCH

#### from https://nces.ed.gov/ccd/files.asp
#### (choose "Nonfiscal" and Level = "School" from the dropdown options)

    info about grades offered: "Directory" file ccd_sch_029*.csv or .zip
    enrollment data: "Membership" file ccd_sch_052*.csv or .zip
    number of teachers: "Staff" file ccd_sch_059*.csv or .zip


### 2. edit config.json

    geos: list of areas to include in the synth pop
    only CBGs starting with these strings will be included
    (two chars for state FIPS, 5 chars for county, more for sub-county; any combination is ok)

    inc_adj: current year ADJINC from PUMS "Data Dictionary" at https://www.census.gov/programs-surveys/acs/microdata/documentation.html
    inc_cats: arbitrary labels for income categories
    inc_cols: corresponding sets of columns from ACS table B19001

    commute_states: state FIPS codes that are within commute distance of synth pop (include synth pop states themselves)

### 3. install python and julia libraries:

    python 3.9.16, pandas 1.5.3, numpy 1.24.3, geopandas 0.12.2, shapely 2.0.1
    julia 1.9.0, CSV v0.10.10, DataFrames v1.5.0, Graphs v1.8.0, InlineStrings v1.4.0, JSON v0.21.4, MatrixMarket v0.4.0, StatsBase v0.33.21, ProportionalFitting v0.3.0

### 4. run scripts:

    python census.py
    julia -p auto CO.jl 
        (searches for optimal combination of samples to match census data, takes a while)
        (uses multiple local processors; "-p auto" uses all available cores)
    julia synthpop.jl

### 5. (optional) export population and/or network to csv
#### if continuing in julia, the population and contact network are serialized in folder "jlse"
#### otherwise, run export script(s):

    julia export_synthpop.jl
    julia export_network.jl

#### exports appear in folder "pop_export"
#### network is exported as a sparse matrix in Matrix Market native exchange format https://math.nist.gov/MatrixMarket/formats.html#MMformat

The file adj_mat_keys maps the indices of the contact matrix to the people in people.csv. **NOTE** The indices in the .mtx files begin at 1. If you are reading the matrix into Juila (or R), everything will work as expected. If you read it into Python using scipy.io.mmread, it will automatically subtract 1 from all the index values to make it 0-indexed. In adj_mat_keys, refer to the column (index_one or index_zero) corresponding to how the matrix ends up getting indexed. (In the older version, subtract 1 from the "index" column if your matrix becomes 0-indexed.)

Keep in mind that this is not a complete contact network for a population; it only describes contacts *within* households, group quarters, schools, and workplaces. You will probably need to generate other types of contacts depending on what you're using this for. The file adj_out_workers lists people who work outside of the synthesized area; they have jobs but are not part of any workplace network. The file adj_dummy_keys lists people who live outside but work within the synthesized area; they belong to a workplace network but are not part of any household.

