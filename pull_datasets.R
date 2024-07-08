#########
# Data Automation

library(censusapi)
library(data.table)
library(tidyverse)
library(tidycensus)
library(tigris)
library(sf)
library(lehdr)
library(usmap)

## tries to download a file with 1 hr timeout, retries 3 times
try_download <- function(src, dst) {
    options(timeout=3600)
    retries <- 3
    status <- 1
    while ((status != 0) && (retries > 0)) {
        status <- tryCatch({
            download.file(src, dst, mode = "wb")
        }, warning=function(e){-1}, error=function(e){-1})
        retries <- retries - 1
    }
    if (status != 0) {
        cat("download failed: ",src,"\n")
        quit()
    }
    return(status)
}

# PULL CENSUS DATA -------------
# This function will accept a vector of state codes and then pull the relevant data using the Censusapi library
# Note, you will need to provide a key

pull_census_data <- function(state_fips, year_ACS, year_DEC, ACS_table_codes, DEC_table_codes, key) {
 
    ACS_metadata <- listCensusMetadata(
        name = "acs/acs5",
        vintage = year_ACS,
        type = "variables"
    )

    DEC_metadata <- listCensusMetadata(
        name = "dec/sf1",
        vintage = year_DEC,
        type = "variables"
    )

  for(state_i in state_fips){
    message("state = ",state_i)

    # Table Codes
    table_codes = data.frame(table_codes = paste0("group(",ACS_table_codes,")"),
                             table_name = paste0("ACSDT5Y",year_ACS,".",ACS_table_codes,"-Data"))
    
    for(row_i in 1:nrow(table_codes)){
        message("downloading ",table_codes[row_i,"table_name"])
      
      # Make the API call and get the data
      data <- getCensus(
        name = "acs/acs5",
        vintage = year_ACS,
        vars = c(table_codes[row_i,"table_codes"]),
        region = "block group:*",
        regionin = paste0("state:", state_i, " county:*"),
        key = key
      )
      message("got ",dim_desc(data))
      
      data <- data %>%
        select(-c("state","county","tract","block_group"))
      
      data <- data %>%
        select(tail(names(.), 2), everything())
      
      data <- data %>%
        mutate_all(as.character)  
      
      data_labels <- ACS_metadata %>%
        filter(name %in% colnames(data)) %>%
        select(name,label)
      
      # Create a named vector with column labels
      label_row <- setNames(as.character(data_labels$label), data_labels$name)
      
      # Ensure all columns in data have a label, fill missing labels with column names
      all_labels <- setNames(colnames(data), colnames(data))
      all_labels[names(label_row)] <- label_row
      
      # Convert the named vector to a data frame with one row
      label_df <- as.data.frame(t(all_labels), stringsAsFactors = FALSE)
      
      data_with_labels <- bind_rows(label_df, data)
      
      data_with_labels <- data_with_labels %>%
        select(-ends_with("EA"), -ends_with("M"), -ends_with("MA"))
      
      destination_folder <- here("census",toupper(fips_info(state_i)$abbr))  # Replace with your desired folder path
      
      # Create the destination folder if it doesn't exist
      if (!dir.exists(destination_folder)) {
        dir.create(destination_folder, recursive = TRUE)
      }
      
      file_name <- paste0("/",table_codes[row_i,"table_name"],".csv")
      
      message("writing ",paste0(destination_folder,file_name))
      write.table(data_with_labels, file = paste0(destination_folder,file_name), 
                  row.names = F, 
                  col.names = T, 
                  sep=",")
      
    }
    
    rm(row_i, table_codes)

    # Table Codes
    table_codes = data.frame(table_codes = paste0("group(",DEC_table_codes,")"),
                             table_name = paste0("DECENNIALSF1",year_DEC,".",DEC_table_codes,"-Data"))
    
    for(row_i in 1:nrow(table_codes)){
        message("downloading ",table_codes[row_i,"table_name"])
      
      # Make the API call and get the data
      data <- getCensus(
        name = "dec/sf1",
        vintage = year_DEC,
        vars = c(table_codes[row_i,"table_codes"]),
        region = "block group:*",
        regionin = paste0("state:", state_i, " county:*"),
        key = key
      )
      message("got ",dim_desc(data))
      
      data <- data %>%
        select(-c("state","county","tract","block_group"))
      
      data <- data %>%
        select(tail(names(.), 2), everything())
      
      data <- data %>%
        mutate_all(as.character)  
      
      data_labels <- DEC_metadata %>%
        filter(name %in% colnames(data)) %>%
        select(name,label)
      
      # Create a named vector with column labels
      label_row <- setNames(as.character(data_labels$label), data_labels$name)
      
      # Ensure all columns in data have a label, fill missing labels with column names
      all_labels <- setNames(colnames(data), colnames(data))
      all_labels[names(label_row)] <- label_row
      
      # Convert the named vector to a data frame with one row
      label_df <- as.data.frame(t(all_labels), stringsAsFactors = FALSE)
      
      data_with_labels <- bind_rows(label_df, data)
      
      data_with_labels <- data_with_labels %>%
        select(-ends_with("ERR"))
      
      destination_folder <- here("census",toupper(fips_info(state_i)$abbr))  # Replace with your desired folder path
      
      # Create the destination folder if it doesn't exist
      if (!dir.exists(destination_folder)) {
        dir.create(destination_folder, recursive = TRUE)
      }
      
      file_name <- paste0("/",table_codes[row_i,"table_name"],".csv")
      
      message("writing ",paste0(destination_folder,file_name))
      write.table(data_with_labels, file = paste0(destination_folder,file_name), 
                  row.names = F, 
                  col.names = T, 
                  sep=",")
      
    }
    
    
  }
 
}

# PULL PUMS Microdata -------------

pull_pums_data <- function(states, year) {
  
  states <- tolower(states)
  
  file_urls = data.frame(state = character(),
                         url = character());
  
  i=1
  
  # Set the URL of the file to download
  for(state in states){
    
    file_urls[i:(i+1),1] = state;
    
    file_urls[i,2] = paste0("https://www2.census.gov/programs-surveys/acs/data/pums/",year,"/5-Year/csv_h",state,".zip")
    file_urls[i+1,2] = paste0("https://www2.census.gov/programs-surveys/acs/data/pums/",year,"/5-Year/csv_p",state,".zip")
    i = i+2
  }
  
  for(state_i in states){
    
    urls_list <- file_urls %>% filter(state == !!state_i) %>% pull(url)

    for(url in urls_list){
      
      download_url <- url
      
      # Specify the destination folder and file name
      destination_folder <- here("pums")  # Replace with your desired folder path
      
      file_name <- basename(download_url)
      destination_file <- file.path(destination_folder, file_name)
      
      # Create the destination folder if it doesn't exist
      if (!dir.exists(destination_folder)) {
        dir.create(destination_folder, recursive = TRUE)
      }
      
      # Download the file
      message("downloading ",download_url)
      try_download(download_url, destination_file)
      
      # the actual data file is one of the files inside the zip
      unzip(destination_file, exdir = destination_folder)
      
      # Remove the zip file
      unlink(destination_file)
      
      # Print a message indicating the download is complete
      message("Download complete: ", destination_file)
    }

  }
}


## get shapefiles from census web server

download_shapefiles <- function(state_fips, year) {
   
  file_urls = data.frame(state = character(),
                         url = character());
  
  # Set the URL of the file to download
  for(i in 1:length(state_fips)){
    state = state_fips[i]
    file_urls[i,1] = state;
    file_urls[i,2] = paste0("https://www2.census.gov/geo/tiger/TIGER",year,"/BG/tl_",year,"_",state,"_bg.zip")
  }
  
  for(state_i in state_fips){
    
    urls_list <- file_urls %>% filter(state == !!state_i) %>% pull(url)

    for(url in urls_list){
      
      download_url <- url
      
      # Specify the destination folder and file name
      destination_folder <- here("geo")  # Replace with your desired folder path
      
      file_name <- basename(download_url)
      destination_file <- file.path(destination_folder, file_name)
      
      # Create the destination folder if it doesn't exist
      if (!dir.exists(destination_folder)) {
        dir.create(destination_folder, recursive = TRUE)
      }
      
      # Download the file
      message("downloading ",download_url)
      try_download(download_url, destination_file)
      
      ## unzip, but keep the zipped file also
      unzip(destination_file, exdir = destination_folder)

      # Print a message indicating the download is complete
      message("Download complete: ", destination_file)
    }

  }
}



## get shapefiles from API

pull_shape_files <- function(states, year, folder = "Shapefiles") {

  for(state_i in states) {

  # Ensure the folder exists
  dir.create(here("Data", folder, state_i), showWarnings = FALSE, recursive = TRUE)
  
  # Download the shapefile for census block groups for the given state and year
  block_group_shapefile <- tigris::block_groups(state = state_i, year = year)
  block_group_shapefile_path <- here("Data", folder, state_i, paste0("block_group_shapefile_", state_i, "_", year, ".shp"))
  st_write(block_group_shapefile, block_group_shapefile_path, driver = "ESRI Shapefile")

  # Download the shapefile for census block group population centroids for the given state and year
  
  #https://www2.census.gov/geo/docs/reference/cenpop2010/blkgrp/CenPop2010_Mean_BG24.txt
  
  if(year < 2020){
    year_pop <- 2010
  }else if(year >= 2020){
    year_pop <- 2020
  }

  download_url <- paste0("https://www2.census.gov/geo/docs/reference/cenpop",  year_pop ,"/blkgrp/CenPop",  year_pop ,"_Mean_BG", fips(state_i),".txt")
  
  file_name <- basename(download_url)
  
  try_download(download_url, here("Data", folder, state_i, file_name))
  
  }
  
}

# PULL LEHD LODES Data -------------
pull_LODES = function(states_main, states_aux, year){
  
  if(year >= 2020){
    version <- "LODES8";
  }else if(year < 2020){
    version <- "LODES7";
  }
  
  # Function to ensure the directory exists
  ensure_directory <- function(dir) {
    if (!dir.exists(dir)) {
      dir.create(dir, showWarnings = FALSE, recursive = TRUE)
    }
  }
  # For the states on the "main" list, download OD main JT01, OD aux JT01, and WAC S000 JT01
  for (state_i in states_main) {
    state_dir <- here("work")
    ensure_directory(state_dir)
    
    # Download and save OD main JT01
    message("downloading lodes od main ",state_i)
    df <- grab_lodes(state = state_i, 
                     year = year, 
                     version = version,
                     lodes_type = "od", 
                     job_type = "JT01",
                     state_part = "main",
                     agg_geo = "bg",
                     use_cache = FALSE)
    setnames(df, c("w_bg","h_bg"), c("w_geocode","h_geocode"))
    outfile = file.path(state_dir, paste0(state_i, "_od_main_JT01_", year, ".csv.gz"))
    message("writing ",outfile)
    fwrite(df, file = outfile, compress="gzip")
    
    # Download and save OD aux JT01
    message("downloading lodes od aux ",state_i)
    df <- grab_lodes(state = state_i, 
                     year = year, 
                     version = version,
                     lodes_type = "od", 
                     job_type = "JT01",
                     state_part = "aux",
                     agg_geo = "bg",
                     use_cache = FALSE)
    setnames(df, c("w_bg","h_bg"), c("w_geocode","h_geocode"))
    outfile = file.path(state_dir, paste0(state_i, "_od_aux_JT01_", year, ".csv.gz"))
    message("writing ",outfile)
    fwrite(df, file = outfile, compress="gzip")
    
    # Download and save WAC S000 JT01
    message("downloading lodes wac ",state_i)
    df <- grab_lodes(state = state_i, 
                     year = year, 
                     lodes_type = "wac", 
                     version = version,
                     job_type = "JT01",
                     segment = "S000",
                     agg_geo = "bg",
                     use_cache = FALSE)
    setnames(df, c("w_bg"), c("w_geocode"))
    outfile = file.path(state_dir, paste0(state_i, "_wac_S000_JT01_", year, ".csv.gz"))
    message("writing ",outfile)
    fwrite(df, file = outfile, compress="gzip")
  }
  
  # For the states on the "aux" list just download OD aux JT01
  for (state_i in states_aux) {
    state_dir <- here("work")
    ensure_directory(state_dir)
    
    # Download and save OD aux JT01
    message("downloading lodes od aux ",state_i)
    df <- grab_lodes(state = state_i, 
                     year = year, 
                     version = version,
                     lodes_type = "od", 
                     job_type = "JT01",
                     state_part = "aux",
                     agg_geo = "bg",
                     use_cache = FALSE)
    setnames(df, c("w_bg","h_bg"), c("w_geocode","h_geocode"))
    outfile = file.path(state_dir, paste0(state_i, "_od_aux_JT01_", year, ".csv.gz"))
    message("writing ",outfile)
    fwrite(df, file = outfile, compress="gzip")
  }
  
  message("All LODES datasets have been downloaded")
  
}

