library(here)
library(rjson)

key = "YOUR_CENSUS_API_KEY"

# Read in the functions
source(here("pull_datasets.R"))

# which data to download from geo.json
jlist <- fromJSON(file="geo.json")
main_fips <- unique(substr(jlist$geos, 1, 2))
main_abbr = fips_info(main_fips)$abbr
use_pums = if(is.null(jlist$use_pums)) main_abbr else fips_info(jlist$use_pums)$abbr
aux_abbr = if(is.null(jlist$commute_states)) vector("character") else fips_info(jlist$commute_states[!(jlist$commute_states %in% main_fips)])$abbr

# year is hardcoded for now
main_year = 2019
decennial_year = 2010

# tables needed in census.py
acs_required = c("B01001", "B09018", "B09019", "B09020", "B09021", "B11004", "B11012", 
                "B11016", "B19001", "B22010", "B23009", "B23025", "B25006", "B11001H", 
                "B11001I", "C24010", "C24030")

dec_required = c("P43")

# This function pulls the aggregated census data
# You need to provide a vector of state codes, the year, and your census API Key
# will get stored in the "census" folder, in a subfolder for each state
message("pulling census data for ",toString(main_fips))
pull_census_data(state_fips = main_fips,
                 year_ACS = main_year, 
                 year_DEC = decennial_year, 
                 ACS_table_codes = acs_required, 
                 DEC_table_codes = dec_required, 
                 key = key)

# This function pulls the PUMS data
# You need to provide a vector of state two letter codes, the year, and your census API Key
# These will get stored in the "pums" folder in two CSVs: (1) Household-level data (2) Person-level data
message("downloading PUMS data for ",toString(use_pums))
pull_pums_data(states = use_pums, year = main_year)

# This function pulls the shapefiles for a given state/s
# You need to simply provide the state two letter codes and year
# stored in "geo" folder
message("downloading shapefiles for ",toString(main_fips))
download_shapefiles(main_fips, main_year)

# This function pulls a series of LODES datasets
# You need to provide a vector of the main and auxiliary state codes and the year
# stored in "work" folder
message("pulling LODES for ",toString(main_abbr)," + aux-only for ",toString(aux_abbr))
pull_LODES(main_abbr, aux_abbr, main_year)
