### Geographically REAlistic SYnthetic POPulation using Combinatorial Optimization

![logo](greasypop-logo.png)


* generates a synthetic population (people, households, schools, workplaces) from US census data for a specified region, at census block group (CBG) resolution
* generates a synthetic contact network of regular household, school, and work contacts

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

## into folder "census"
#### create one sub-folder for each US state whose data you want to use (name of sub-folder doesn't matter)
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
    DECENNIALSF1####.P32-Data.csv
    DECENNIALSF1####.P43-Data.csv


## into folder "pums"
#### PUMS data from https://www.census.gov/programs-surveys/acs/microdata.html

    psam_h??.* and psam_p??.*
    for each state you want to draw samples from


## into folder "geo"
#### from https://www.census.gov/programs-surveys/geography/guidance/geo-areas/pumas.html

    census tract to PUMA relationship file, *Census_Tract_to*PUMA*.*

#### from geocorr https://mcdc.missouri.edu/applications/geocorr2018.html

    puma to county, rename to *puma_to_county*.*
    puma to cbsa, rename to *puma_to_cbsa*.*
    puma to urban-rural portion, rename to *puma_urban_rural*.*
    cbg to cbsa, rename *cbg_to_cbsa*.*
    cbg to urban-rural portion, rename to *cbg_urban_rural*.*

#### cbg lat-long coords from https://www2.census.gov/geo/tiger/TIGER####/BG/ where #### is year

    tl####_??_bg.zip where ?? is the FIPS code for each state in the synth area


## into folder "work"
#### origin-destination work commute data from https://lehd.ces.census.gov/data/
#### use the version that has the same census blocks as the ACS data (v. 7 for 2019)
#### use JT01, "primary" jobs (because JT00 counts 2+ jobs for the same individual)

    main file for every state in the synth area, named *od_main_JT01*.csv.gz
    aux file for every state in the synth area, named *od_aux_JT01*.csv.gz
    optionally, aux files for other states to capture commute patterns outside the above state(s)


#### employer size data from https://www.census.gov/programs-surveys/cbp/data/datasets.html
#### 2016 complete county file (more complete than 2019 data)

    cbp16co.zip


## into folder "school"
#### from https://nces.ed.gov/programs/edge/Geographic/SchoolLocations

    school locations: EDGE_GEOCODE_PUBLICSCH_*.xlsx
    GIS data: folder Shapefile_SCH

#### from https://nces.ed.gov/ccd/files.asp

    info about grades offered: "Directory" file ccd_sch_029*.csv or .zip
    enrollment data: "Membership" file ccd_sch_052*.csv or .zip
    number of teachers: "Staff" file ccd_sch_059*.csv or .zip


### 2. edit config.json

    geos: list of areas to include in the synth pop
    can be state or county FIPS codes (or any subset of cbg code starting with state FIPS)

    inc_adj: current year ADJINC from PUMS data dictionary https://www.census.gov/programs-surveys/acs/microdata.html
    inc_cats: arbitrary labels for income categories
    inc_cols: corresponding sets of columns from ACS table B19001

    commute_states: state FIPS codes that are within commute distance of synth pop (include synth pop states themselves)

### 3. install python and julia libraries:

    python 3.9.16, pandas 1.5.3, numpy 1.24.3, geopandas 0.12.2, shapely 2.0.1
    julia 1.9.0, CSV v0.10.10, DataFrames v1.5.0, Graphs v1.8.0, InlineStrings v1.4.0, JSON v0.21.4, MatrixMarket v0.4.0, StatsBase v0.33.21

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

Note this is not a complete contact network for a population; it only describes contacts *within* households, group quarters, schools, and workplaces. You will probably need to generate other types of contacts depending on what you're using this for.

