'''
Copyright 2023 Alexander Tulchinsky

This file is part of Greasypop.

Greasypop is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Greasypop is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with Greasypop. If not, see <https://www.gnu.org/licenses/>.
'''

import pandas as pd
import numpy as np
from numpy import array as arr
import geopandas as gpd
from shapely.geometry.point import Point
import os
import shutil
from glob import glob
import json

import ipfn

def tryJSON(filename):
    try:
        with open(filename, 'r') as f:
            d = json.load(f)
    except Exception as e:
        print("warning: ",e)
        d = {}
    return d

## round a pandas Series to integers while preserving sum
## (using largest-remainder method)
def lrRound(v):
    vrnd =  v.apply(np.floor)
    verr = v - vrnd
    vrem = np.int64(np.round(sum(v) - sum(vrnd)))
    vidxs = verr.index[np.flip(np.argsort(verr.values))]
    for i in range(vrem):
        vrnd[vidxs[i]] += 1    
    return vrnd

def read_census(file_list, geos, usecols, columns):
    dfs = []
    for f in file_list:
        df = pd.read_csv(f, index_col=0, header=1, usecols=usecols)
        df.rename(inplace=True, columns=columns)
        dfs.append(df)
    df_all = pd.concat(dfs)
    df_all['Geo'] = df_all.index.map(lambda x: x[9:])
    df_all.set_index('Geo',drop=True,inplace=True)
    if geos is not None:
        df_all = df_all[df_all.index.map(lambda x: np.any([x.startswith(g) for g in geos]))].copy(deep=True)
    return df_all

def read_acs(table,geos=None):
    filematch = ''.join(['*.', table, '-Data.*'])
    with os.scandir("census") as d:
        files = [glob(os.path.join(f,filematch))[0] for f in d 
                 if f.is_dir() and not f.name.startswith(".")]
    return read_census(files, geos, 
                       (lambda x: (x=='Geography') or (str.split(x,'!!')[0]=='Estimate')),
                       (lambda x: ''.join([table,':',*str.split(x,'!!')[2:]])))

def read_decennial(table,geos=None):
    filematch = ''.join(['*.', table, '-Data.*'])
    with os.scandir("census") as d:
        files = [glob(os.path.join(f,filematch))[0] for f in d 
                 if f.is_dir() and not f.name.startswith(".")]
    return read_census(files, geos,
                       (lambda x: (x=='Geography') or (str.split(x,'!!')[0]=='Total')),
                       (lambda x: ':'.join([table,*str.split(x,'!!')[1:]])))

def read_pums(prefix, dtypes):
    with os.scandir("pums") as d:
        files = [f for f in d if f.is_file() and f.name.startswith(prefix)]
    dfs = [pd.read_csv(f,usecols=list(dtypes.keys()),dtype=dtypes) for f in files]
    return pd.concat(dfs)

## read geography shapefiles
def read_cbg_geo(geos=None):
    files = glob(os.path.join("geo","tl_*_bg.zip"))
    cbg_geo = pd.concat([gpd.read_file(f) for f in files]).set_index('GEOID')
    if geos is not None:
        geo_mask = cbg_geo.index.map(lambda x: np.any([x.startswith(g) for g in geos]))
        cbg_geo = cbg_geo[geo_mask].copy(deep=True)
    cbg_geo['county'] = cbg_geo['STATEFP'] + cbg_geo['COUNTYFP']
    ## centroid of each cbg, in projection 18N coords
    cbg_geo['cent'] = cbg_geo['geometry'].to_crs(epsg=32618).centroid
    return cbg_geo

## read LODES od data
def read_work_commute():
    odcols = ['w_geocode','h_geocode','S000','SE01','SE02','SE03']
    odtypes = {'w_geocode':str,'h_geocode':str,'S000':"Int64",'SE01':"Int64",'SE02':"Int64",'SE03':"Int64"}
    main_files = glob(os.path.join("work","*od_main_JT01*.[c|z]*"))
    aux_files = glob(os.path.join("work","*od_aux_JT01*.[c|z]*"))
    dfs = []
    for f in main_files:
        df = pd.read_csv(f,usecols=odcols,dtype=odtypes)
        df["main"] = True
        dfs.append(df)
    for f in aux_files:
        df = pd.read_csv(f,usecols=odcols,dtype=odtypes)
        df["main"] = False
        dfs.append(df)
    od = pd.concat(dfs,ignore_index=True)
    od['w_cbg'] = od['w_geocode'].apply(lambda x: x[0:12])
    od['h_cbg'] = od['h_geocode'].apply(lambda x: x[0:12])
    od['w_county'] = od['w_geocode'].apply(lambda x: x[0:5])
    od['h_county'] = od['h_geocode'].apply(lambda x: x[0:5])
    od['w_state'] = od['w_geocode'].apply(lambda x: x[0:2])
    od['h_state'] = od['h_geocode'].apply(lambda x: x[0:2])
    return od

def read_cbp(counties):
    cbp_dtype = {'fipstate':str,'fipscty':str,'naics':str,'emp':"Int64",'est':"Int64",
        'n1_4':"Int64",'n5_9':"Int64",'n10_19':"Int64",'n20_49':"Int64",'n50_99':"Int64",'n100_249':"Int64",
        'n250_499':"Int64",'n500_999':"Int64",'n1000_1':"Int64",'n1000_2':"Int64",'n1000_3':"Int64",'n1000_4':"Int64"}
    cbp_US = pd.read_csv(glob(os.path.join("work","cbp*co.[z|t|c]??"))[0],
                        usecols=list(cbp_dtype.keys()),dtype=cbp_dtype)
    cbp_US['county'] = cbp_US['fipstate'] + cbp_US['fipscty']
    ## this counts employees, regardless of state of residence
    ## incomplete, but we're only using it to estimate avg employer sizes
    cbp = cbp_US[(cbp_US['naics']=='------')]
    return cbp[cbp['county'].isin(counties)].set_index('county').iloc[:,3:].copy(deep=True)

def read_sch_data(geos=None):
    sch_geo = pd.read_excel(glob(os.path.join("school","EDGE_GEOCODE_PUBLICSCH_*.xlsx"))[0],dtype=str) ## has correct locations
    sch_dir = pd.read_csv(glob(os.path.join("school","ccd_*029*.[c|z]??"))[0],dtype=str,encoding='Latin-1') ## has errors in location
    sch_staff = pd.read_csv(glob(os.path.join("school","ccd_*059*.[c|z]??"))[0],dtype=str,encoding='Latin-1') ## full-time-equivalent # teachers
    sch_mem_raw = pd.read_csv(glob(os.path.join("school","ccd_*052*.[c|z]??"))[0],dtype=str,encoding='Latin-1') ## reported student counts
    sch_mem = sch_mem_raw[sch_mem_raw['TOTAL_INDICATOR']=='Education Unit Total'].copy(deep=True)
    sch_mem['STUDENTS'] = sch_mem['STUDENT_COUNT'].astype("Int64")
    sch_geo.set_index('NCESSCH',inplace=True)
    sch_dir.set_index('NCESSCH',inplace=True)
    sch_staff.set_index('NCESSCH',inplace=True)
    sch_mem.set_index('NCESSCH', inplace=True)
    ## field codes explained at nces.ed.gov
    scols = ['STFIP', 'CNTY', 'NMCNTY', 'LAT', 'LON', 'SCH_NAME', 'LEAID',
        'UPDATED_STATUS', 'UPDATED_STATUS_TEXT',
        'SCH_TYPE', 'SCH_TYPE_TEXT', 'NOGRADES', 'G_PK_OFFERED', 'G_KG_OFFERED',
        'G_1_OFFERED', 'G_2_OFFERED', 'G_3_OFFERED', 'G_4_OFFERED',
        'G_5_OFFERED', 'G_6_OFFERED', 'G_7_OFFERED', 'G_8_OFFERED',
        'G_9_OFFERED', 'G_10_OFFERED', 'G_11_OFFERED', 'G_12_OFFERED',
        'GSLO', 'GSHI', 'LEVEL', 'TEACHERS', 'STUDENTS']
    ## prefer columns based on join order
    schools = sch_geo.join(sch_dir, rsuffix='_').join(sch_staff, rsuffix='_').join(sch_mem, rsuffix='_').loc[:,scols]
    if geos is not None:
        geo_mask = np.any([schools['CNTY'].str.startswith(g) for g in [s[0:5] for s in geos]], axis=0)
    else:
        geo_mask = np.ones(schools.shape[0]).astype(bool)
    schools = schools.loc[geo_mask
        & schools['UPDATED_STATUS'].isin(['1','3','4','5','8']) ## removed closed and inactive schools
        & (schools['LEVEL'] != 'Ungraded')
        & (schools['LEVEL'] != 'Not reported')
        & (schools['LEVEL'] != "Not applicable")
        & (schools['NOGRADES'] != 'Yes') ## can't assign students w/o grade levels
        & (schools['SCH_TYPE'] == '1') ## regular schools only
        ].copy(deep=True)
    ## convert Yes/No to bool
    for x in ['G_PK_OFFERED', 'G_KG_OFFERED',
        'G_1_OFFERED', 'G_2_OFFERED', 'G_3_OFFERED', 'G_4_OFFERED',
        'G_5_OFFERED', 'G_6_OFFERED', 'G_7_OFFERED', 'G_8_OFFERED',
        'G_9_OFFERED', 'G_10_OFFERED', 'G_11_OFFERED', 'G_12_OFFERED']:
        schools[x] = schools[x].map({'Yes':True,'No':False})
    ## round to int
    schools['TEACHERS'] = schools['TEACHERS'].astype(float).round().astype("Int64")
    ## if # teachers or students missing, replace with mean for that school type
    ## (or overall mean if that's not available)
    t_na = schools['TEACHERS'].isna()
    s_na = schools['STUDENTS'].isna()
    mean_teachers = np.int64(np.round(schools['TEACHERS'].mean()))
    mean_students = np.int64(np.round(schools['STUDENTS'].mean()))
    t_mean = schools[['LEVEL','TEACHERS']].groupby(['LEVEL'],group_keys=True).agg({'TEACHERS':"mean"}).fillna(mean_teachers)
    s_mean = schools[['LEVEL','STUDENTS']].groupby(['LEVEL'],group_keys=True).agg({'STUDENTS':"mean"}).fillna(mean_students)
    t_mean["TEACHERS"] = t_mean["TEACHERS"].astype(float).round().astype("Int64")
    s_mean["STUDENTS"] = s_mean["STUDENTS"].astype(float).round().astype("Int64")
    schools.loc[t_na,'TEACHERS'] = t_mean.loc[schools['LEVEL']].values[t_na]
    schools.loc[s_na,'STUDENTS'] = s_mean.loc[schools['LEVEL']].values[s_na]
    return schools


##
## PUMS data from https://www.census.gov/programs-surveys/acs/microdata.html
##
## need psam_h##.* and psam_p##.* where ## is state code
## put into folder "pums"
def read_psamp(LODES_cutoff, ind_codes, occ_codes):

    ## read individual PUMS data
    psamp_dtype = {'SERIALNO':str,'PUMA':str,'ST':str,'PWGTP':"Int64",
                'AGEP':"Int64",'SEX':str,'RELSHIPP':str,
                'WAGP':"Int64",'PINCP':"Int64",'PERNP':"Int64",'COW':str,'POWPUMA':str,'POWSP':str,'JWTRNS':str,
                'WKL':str,'WKW':str,'WRK':str,'ESR':str,'NAICSP':str,'SOCP':str,
                'SCH':str,'SCHG':str,'SCHL':str,
                'ESP':str,'SFN':str,'SFR':str,
                'CIT':str,'FER':str,'LANX':str,'DIS':str,'RAC1P':str,'HISP':str,'PAP':"Int64"}

    psamp = read_pums("psam_p",psamp_dtype)

    psamp['st_puma'] = psamp['ST'] + psamp['PUMA']
    psamp['sch_grade'] = psamp['SCHG'].map(dict(zip([str(x).rjust(2,'0') for x in range(1,17)], ['p','k',*[str(x) for x in range(1,13)],'c','g'])))

    ## using 2-digit industry and occupation codes
    psamp["industry"] = psamp["NAICSP"].fillna("").str.slice(0,2)
    psamp["occupation"] = psamp["SOCP"].fillna("").str.slice(0,2)

    ## all columns added below are meant to be aggregated as sums
    ## (to use in matching synth pop to census sums,
    ##  or when sums are needed for generating workplaces, schools, etc.)
    pstart_idx = len(psamp.columns)

    psamp['age_u6'] = psamp['AGEP'] < 6
    psamp['age_u18'] = psamp['AGEP'] < 18
    psamp['age_6_11'] = (psamp['AGEP'] < 12) & ~psamp['age_u6']
    psamp['age_6_17'] = psamp['age_u18'] & ~psamp['age_u6']
    psamp['age_12_17'] = psamp['age_6_17'] & ~psamp['age_6_11']
    psamp['own_ch_any_age'] = psamp['RELSHIPP'].isin(['25','26','27'])
    psamp['own_ch_u18'] = psamp['own_ch_any_age'] & psamp['age_u18']
    psamp['hholder'] = psamp['RELSHIPP'] == '20' ## there's exactly one '20' per household serialno
    psamp['partner'] = psamp['RELSHIPP'].isin(['21','22','23','24'])
    psamp['child_of_hh'] = psamp['RELSHIPP'].isin(['25','26','27','35'])
    psamp['other_rel'] = psamp['RELSHIPP'].isin(['28','29','30','31','32','33'])
    psamp['non_rel'] = psamp['RELSHIPP'].isin(['34','36'])
    ## census excludes householder and partners from counts of children in households
    psamp['ch_u18_in_hh'] = psamp['age_u18'] & (psamp['child_of_hh'] | psamp['other_rel'] | psamp['non_rel'])
    psamp['grandch_u18'] = psamp['age_u18'] & (psamp['RELSHIPP'] == '30')
    psamp['age_18_34'] = (psamp['AGEP'] < 35) & ~psamp['age_u18']
    psamp['age_65o'] = psamp['AGEP'] > 64
    psamp['age_35_64'] = (psamp['AGEP'] > 34) & ~psamp['age_65o']
    psamp['sch_pre_1'] = psamp['SCHG'].isin(['01','02'])
    psamp['sch_1_4'] = psamp['SCHG'].isin(['03','04','05','06'])
    psamp['sch_5_8'] = psamp['SCHG'].isin(['07','08','09','10'])
    psamp['sch_9_12'] = psamp['SCHG'].isin(['11','12','13','14'])
    psamp['sch_college'] = psamp['SCHG'] == '15'
    psamp['sch_public'] = psamp['SCH'] == '2'
    psamp['sch_private'] = psamp['SCH'] == '3'
    psamp['esp_2p_2w'] = psamp['ESP'] == '1'
    psamp['esp_2p_1w'] = psamp['ESP'].isin(['2','3'])
    psamp['esp_2p_nw'] = psamp['ESP'] == '4'
    psamp['esp_1p_1w'] = psamp['ESP'].isin(['5','7'])
    psamp['esp_1p_nw'] = psamp['ESP'].isin(['6','8'])
    psamp['male'] = psamp['SEX'] == '1'
    psamp['female'] = psamp['SEX'] == '2'
    psamp['employed'] = psamp['ESR'].isin(['1','2']) ## note, this means civilian employed
    psamp['unemployed'] = psamp['ESR'] == '3'
    psamp['armed_forces'] = psamp['ESR'].isin(['4','5'])
    psamp['in_lf'] = psamp['ESR'].isin(['1','2','3','4','5'])
    psamp['nilf'] = psamp['ESR'] == '6'
    psamp['has_job'] = psamp['ESR'].isin(['1','2','4','5'])
    psamp['work_from_home'] = psamp['has_job'] & (psamp['JWTRNS'] == '11')
    ## and assume a person with a job is a commuter unless they explicitly work from home
    psamp['commuter'] = psamp['has_job'] & ~psamp['work_from_home']
    psamp['has_disability'] = psamp['DIS'] == '1'
    psamp['age_u3'] = psamp['AGEP'] < 3
    psamp['age_3_5'] = psamp['age_u6'] & ~psamp['age_u3']
    psamp['edu_not_hsgrad_age_25o'] = (psamp['AGEP'] > 24) & (psamp['SCHL'].isin(['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15']))
    psamp['edu_hs_or_somecoll_age_25o'] = (psamp['AGEP'] > 24) & (psamp['SCHL'].isin(['16','17','18','19','20']))
    psamp['edu_bach_or_higher_age_25o'] = (psamp['AGEP'] > 24) & (psamp['SCHL'].isin(['21','22','23','24']))
    psamp['race_white_alone'] = psamp['RAC1P'] == '1'
    psamp['race_black_alone'] = psamp['RAC1P'] == '2'
    psamp['race_amerindian_or_alaskan'] = psamp['RAC1P'].isin(['3','4','5'])
    psamp['race_asian_alone'] = psamp['RAC1P'] == '6'
    psamp['race_pacific_alone'] = psamp['RAC1P'] == '7'
    psamp['race_other_alone'] = psamp['RAC1P'] == '8'
    psamp['race_two_or_more'] = psamp['RAC1P'] == '9'
    psamp['hispanic'] = psamp['HISP'] != '01'
    psamp['white_non_hispanic'] = psamp['race_white_alone'] & ~psamp['hispanic']
    ## will mark householder race when joined with household table
    for x in ["white_alone","black_alone","amerindian_or_alaskan","asian_alone","pacific_alone","other_alone","two_or_more"]:
        psamp['_'.join(['hh_race',x])] = psamp['hholder'] & psamp['_'.join(['race',x])]
    ## likewise for ethnicity
    for x in ["hispanic", "white_non_hispanic"]:
        psamp['_'.join(['hh',x])] = psamp['hholder'] & psamp[x]

    for x in ['u6', '6_11', '12_17', 'u3', '3_5']:
        psamp['_'.join(['own_ch',x])] = psamp['_'.join(['age',x])] & psamp['own_ch_any_age']

    for x in ['age_18_34', 'age_35_64', 'age_65o']:
        for y in ['hholder','partner','child_of_hh','other_rel','non_rel']:
            psamp['_'.join([x,y])] = psamp[x] & psamp[y]

    ## ESP counts only own children of householder, or children in subfamilies
    ## (matches the logic of table B23008)
    for x in ['esp_2p_2w','esp_2p_1w','esp_2p_nw','esp_1p_1w','esp_1p_nw']:
        for y in ['age_u6','age_6_17']:
            psamp['_'.join([x,y])] = psamp[x] & psamp[y]

    psamp = psamp.copy() ## defrag the dataframe, lol

    ## relate job income to categories in LODES commute statistics
    ## only count commuters
    ## WAGP is 0 for business owners, use PERNP? or PINCP?
    psamp['com_LODES_low'] = psamp['commuter'] & (psamp['PINCP'] < LODES_cutoff)
    psamp['com_LODES_high'] = psamp['commuter'] & (psamp['PINCP'] >= LODES_cutoff)

    ## counts of workers in each industry and occ, for matching to census sums
    for k in ind_codes:
        psamp['ind_'+k] = psamp['has_job'] & (psamp['industry'].isin(ind_codes[k][0]))

    for k in occ_codes:
        psamp['occ_'+k] = psamp['has_job'] & (psamp['occupation'].isin(occ_codes[k][0]))

    ## counts of commuters in each ind and occ, for generating workplaces
    ## (only commuters will be assigned to workplaces)
    for k in ind_codes:
        psamp['com_ind_'+k] = psamp['commuter'] & (psamp['industry'].isin(ind_codes[k][0]))

    for k in occ_codes:
        psamp['com_occ_'+k] = psamp['commuter'] & (psamp['occupation'].isin(occ_codes[k][0]))

    ## sum individual column counts in each household
    pcols = ['SERIALNO','PWGTP','WAGP','PINCP','PERNP','PAP', *psamp.columns[pstart_idx:]]
    ptotals = psamp[pcols].groupby('SERIALNO',group_keys=True).agg(sum)

    return psamp.copy(), ptotals

##
## returns household sums merged with individual sums
##
def read_hsamp(ptotals, ADJINC, inc_cats, inc_cols):

    ## read household PUMS data
    hsamp_dtype = {'SERIALNO':str,'PUMA':str,'ST':str,'NP':"Int64",'TYPE':str,'CPLT':str,'FES':str,'FS':str,
                'HHT':str,'HHT2':str,'HINCP':"Int64",'HUGCL':str,'HUPAC':str,'HUPAOC':str,'HUPARC':str,
                'MULTG':str,'NOC':"Int64",'NPF':"Int64",'NPP':str,'NR':str,'NRC':"Int64",'PARTNER':str,
                'PSF':str,'R65':str,'WIF':str,'WORKSTAT':str,
                'ACCESS':str,'HHL':str,'WKEXREL':str}

    hsamp = read_pums("psam_h",hsamp_dtype)
    hsamp = hsamp[(hsamp['NP']>0) & (hsamp['TYPE'] == '1')]

    ## pre-compute census columns so we can generate them quickly from a PUMS subsample:
    hsamp['fam_married'] = hsamp['HHT'] == '1'
    hsamp['fam_m_unmar'] = hsamp['HHT'] == '2'
    hsamp['fam_f_unmar'] = hsamp['HHT'] == '3'
    hsamp['fam_unmar'] = hsamp['fam_m_unmar'] | hsamp['fam_f_unmar']
    hsamp['fam_hh'] = hsamp['HHT'].isin(['1','2','3'])
    hsamp['non_fam_hh'] = hsamp['HHT'].isin(['4','5','6','7'])
    hsamp['w_own_ch_u6_only'] = hsamp['HUPAOC'] == '1'
    hsamp['w_own_ch_6_17_only'] = hsamp['HUPAOC'] == '2'
    hsamp['w_own_ch_u6_and_6_17'] = hsamp['HUPAOC'] == '3'
    hsamp['w_own_ch_u18'] = hsamp['HUPAOC'].isin(['1','2','3'])
    hsamp['no_own_ch_u18'] = hsamp['HUPAOC'] == '4'
    hsamp['w_rel_ch_u6_only'] = hsamp['HUPARC'] == '1'
    hsamp['w_rel_ch_6_17_only'] = hsamp['HUPARC'] == '2'
    hsamp['w_rel_ch_u6_and_6_17'] = hsamp['HUPARC'] == '3'
    hsamp['w_rel_ch_u18'] = hsamp['HUPARC'].isin(['1','2','3'])
    hsamp['no_rel_ch_u18'] = hsamp['HUPARC'] == '4'
    hsamp['fam_work0'] = hsamp['WIF'] == '0'
    hsamp['fam_work1'] = hsamp['WIF'] == '1'
    hsamp['fam_work2'] = hsamp['WIF'] == '2'
    hsamp['fam_work3o'] = hsamp['WIF'] == '3'
    hsamp['partner_hh_ch_u18'] = hsamp['HHT2'].isin(['01','03'])
    hsamp['partner_hh_no_ch_u18'] = hsamp['HHT2'].isin(['02','04'])
    hsamp['hh_f_alone'] = hsamp['HHT2'] == '05'
    hsamp['hh_f_ch_u18'] = hsamp['HHT2'] == '06'
    hsamp['hh_f_other_rel'] = hsamp['HHT2'] == '07'
    hsamp['hh_f_nonrel'] = hsamp['HHT2'] == '08'
    hsamp['hh_m_alone'] = hsamp['HHT2'] == '09'
    hsamp['hh_m_ch_u18'] = hsamp['HHT2'] == '10'
    hsamp['hh_m_other_rel'] = hsamp['HHT2'] == '11'
    hsamp['hh_m_nonrel'] = hsamp['HHT2'] == '12'
    hsamp['partner_hh'] = hsamp['partner_hh_ch_u18'] | hsamp['partner_hh_no_ch_u18']
    hsamp['hh_alone'] = hsamp['hh_f_alone'] | hsamp['hh_m_alone']
    hsamp['hh_single_ch_u18'] = hsamp['hh_f_ch_u18'] | hsamp['hh_m_ch_u18']
    hsamp['hh_single_other_rel'] = hsamp['hh_f_other_rel'] | hsamp['hh_m_other_rel']
    hsamp['hh_rel_no_partner'] = hsamp['hh_f_ch_u18'] | hsamp['hh_f_other_rel'] | hsamp['hh_m_ch_u18'] | hsamp['hh_m_other_rel']
    hsamp['hh_nonrel_only'] = hsamp['hh_f_nonrel'] | hsamp['hh_m_nonrel']
    hsamp['snap'] = hsamp['FS'] == '1'
    hsamp['h_internet_sub'] = hsamp['ACCESS'] == '1'
    hsamp['h_internet_nosub'] = hsamp['ACCESS'] == '2'
    hsamp['h_no_internet'] = hsamp['ACCESS'] == '3'
    hsamp['h_lang_english_only'] = hsamp['HHL'] == '1'
    hsamp['h_lang_spanish'] = hsamp['HHL'] == '2'
    hsamp['h_lang_other'] = hsamp['HHL'].isin(['3','4','5'])
    hsamp['h_with_grandparent'] = hsamp['HUGCL'] == '1'

    # Household income (past 12 months, use ADJINC to adjust HINCP to constant dollars)
    income = hsamp['HINCP'] * ADJINC
    inc_cut = [int(x.split()[0].replace(",","").replace("$","")) for x in [c[0] for c in inc_cols][1:]]
    hsamp[inc_cats[0]] = (income < inc_cut[0])
    for i,x in enumerate(inc_cats[1:-1], 1):
        hsamp[x] = (income >= inc_cut[i-1]) & (income < inc_cut[i])
    hsamp[inc_cats[-1]] = (income >= inc_cut[-1])

    ## crosses
    for x in ['w_own_ch_u18','no_own_ch_u18']:
        for y in ['married','unmar']:
            for z in ['fam_work0','fam_work1','fam_work2','fam_work3o']:
                hsamp['_'.join([x,y,z])] = hsamp[x] & hsamp['_'.join(['fam',y])] & hsamp[z]

    for n in range(2,7):
        hsamp['_'.join(['fam_hh', str(n)])] = hsamp['fam_hh'] & (hsamp['NP'] == n)
    hsamp['fam_hh_7o'] = hsamp['fam_hh'] & (hsamp['NP'] > 6)

    for n in range(1,7):
        hsamp['_'.join(['non_fam_hh', str(n)])] = hsamp['non_fam_hh'] & (hsamp['NP'] == n)
    hsamp['non_fam_hh_7o'] = hsamp['non_fam_hh'] & (hsamp['NP'] > 6)

    for x in ['fam_married','fam_unmar']:
        for y in ['w_own_ch_u6_only','w_own_ch_6_17_only','w_own_ch_u6_and_6_17','no_own_ch_u18']:
            hsamp['_'.join([x,y])] = hsamp[x] & hsamp[y]

    for x in ['fam_married','fam_unmar']:
        for y in ['w_rel_ch_u6_only','w_rel_ch_6_17_only','w_rel_ch_u6_and_6_17','no_rel_ch_u18']:
            hsamp['_'.join([x,y])] = hsamp[x] & hsamp[y]

    ## convert bools to int
    start_idx = len(hsamp_dtype.keys())
    hsamp[hsamp.columns[start_idx:]] = hsamp[hsamp.columns[start_idx:]].astype('Int64')

    ## join household and individual sums by serial#
    ##   a sum of samples from this table should have columns matching every census table above
    hsamp = hsamp.set_index('SERIALNO').join(ptotals)

    ## create logic to match census table B09002
    for x in ['own_ch_u6', 'own_ch_6_11', 'own_ch_12_17', 'own_ch_u3', 'own_ch_3_5']: ## counts of children
        for y in ['fam_married', 'fam_unmar']: ## 0 or 1 for family type
            hsamp['_'.join([x,'in',y])] = hsamp[x] * hsamp[y]

    ## update age-category totals to match census B09021:
    #   "If a householder has no spouse or unmarried partner present, they will be shown in 'other relatives' if they have at least one relative present, or in 'other nonrelatives' if no relatives are present.
    for x in ['age_18_34', 'age_35_64', 'age_65o']:
        hsamp['_'.join([x,'alone'])] = hsamp['hh_alone'] * hsamp['_'.join([x,'hholder'])]
        hsamp['_'.join([x,'partner'])] += hsamp['partner_hh'] * hsamp['_'.join([x,'hholder'])]
        hsamp['_'.join([x,'other_rel'])] += hsamp['hh_rel_no_partner'] * hsamp['_'.join([x,'hholder'])]
        hsamp['_'.join([x,'non_rel'])] += hsamp['hh_nonrel_only'] * hsamp['_'.join([x,'hholder'])]

    ## if anyone in hh has disability
    hsamp['h_1_or_more_with_disab'] = (hsamp['has_disability'] > 0).astype("Int64")
    ## if anyone in hh receives snap or public assistance income
    hsamp['snap_or_pap'] = ((hsamp['snap'] == 1) | (hsamp['PAP'] > 0)).astype("Int64")
    ## by family size
    for n in range(2,7):
        ## 1 * 1 = 1
        hsamp['_'.join(['fam_hh', str(n), 'snap_pap'])] = hsamp['_'.join(['fam_hh', str(n)])] * hsamp['snap_or_pap']
    hsamp['fam_hh_7o_snap_pap'] = hsamp['fam_hh_7o'] * hsamp['snap_or_pap']

    return hsamp.copy()


def generate_samples(sample_columns, ADJINC, inc_cats, inc_cols, LODES_cutoff, ind_codes, occ_codes, more_summary_cols):

    print("reading pums data")
    psamp, ptotals = read_psamp(LODES_cutoff, ind_codes, occ_codes)
    hsamp = read_hsamp(ptotals, ADJINC, inc_cats, inc_cols)

    ## csv of household samples
    ## these will be recombined to generate the synth pop
    print("writing samples")
    hsamp[sample_columns].to_csv(os.path.join('processed','census_samples.csv'))

    ## separate file with PUMA and other geo data for each record in census samples
    ##
    ## from geocorr https://mcdc.missouri.edu/applications/geocorr2018.html
    ## need:
    ##  puma to county, rename to *puma_to_county*.*
    ##  puma to cbsa, rename to *puma_to_cbsa*.*
    ##  puma to urban-rural portion, rename to *puma_urban_rural*.*
    ## put into folder "geo"
    print("reading geo crosswalks")

    samp_geo = hsamp[['PUMA','ST']].reset_index().copy(deep=True)
    samp_geo['st_puma'] = samp_geo['ST'] + samp_geo['PUMA']
    ## simpler: place each puma in its primary cbsa
    file = glob(os.path.join("geo","*puma_to_cbsa*.*"))[0]
    puma_to_cbsa = pd.read_csv(file,dtype=str,skiprows=[1],usecols=["state","puma12","cbsa","afact"])
    puma_to_cbsa['st_puma'] = puma_to_cbsa['state'] + puma_to_cbsa['puma12']
    puma_to_cbsa.loc[puma_to_cbsa['cbsa']==" ", 'cbsa'] = "none"
    puma_to_cbsa = puma_to_cbsa.groupby('st_puma',group_keys=True).apply(lambda g: g[g['afact'] == g['afact'].max()]).reset_index(drop=True)
    puma_to_cbsa = puma_to_cbsa[['st_puma','cbsa']]
    ## can only find samples by cbsa in the synth area, which is fine
    samp_geo = samp_geo.merge(puma_to_cbsa, how='left', on='st_puma')
    ## some pumas span counties; associate each with its primary county
    ## -- we're only looking up samples by county when it contains multiple pumas
    file = glob(os.path.join("geo","*puma_to_county*.*"))[0]
    puma_to_county = pd.read_csv(file,dtype=str,skiprows=[1],usecols=["state","puma12","county","afact"])
    puma_to_county['st_puma'] = puma_to_county['state'] + puma_to_county['puma12']
    puma_to_county = puma_to_county.groupby('st_puma',group_keys=True).apply(lambda g: g[g['afact'] == g['afact'].max()]).reset_index(drop=True)
    puma_to_county = puma_to_county[['st_puma','county']]
    samp_geo = samp_geo.merge(puma_to_county, how='left', on='st_puma')
    file = glob(os.path.join("geo","*puma_urban_rural*.*"))[0]
    puma_ur = pd.read_csv(file,dtype=str,skiprows=[1],usecols=["state","puma12","ur","afact"])
    puma_ur['st_puma'] = puma_ur['state'] + puma_ur['puma12']
    puma_ur = puma_ur.pivot(index='st_puma', columns='ur', values='afact').fillna(0)
    puma_ur.reset_index(inplace=True)
    samp_geo = samp_geo.merge(puma_ur, how='left', on='st_puma')
    samp_geo.set_index('SERIALNO', inplace=True, verify_integrity=True)
    samp_geo.to_csv(os.path.join('processed','samp_geo.csv'))

    print("writing sample summaries")
    ## summary of household PUMS samples
    ## will be used to look up synth pop totals, after sample households are selected
    h_summcols = ['NP','HINCP','NOC','NPF','NRC','PWGTP','WAGP','PINCP','PERNP',
        'age_u6', 'age_u18', 'age_6_11', 'age_6_17',
        'age_12_17', 'own_ch_any_age', 'own_ch_u18', 'hholder', 'partner',
        'child_of_hh', 'other_rel', 'non_rel', 'ch_u18_in_hh',
        'grandch_u18', 'age_18_34', 'age_65o', 'age_35_64', 'sch_pre_1',
        'sch_1_4', 'sch_5_8', 'sch_9_12', 'sch_college', 'esp_2p_2w',
        'esp_2p_1w', 'esp_2p_nw', 'esp_1p_1w', 'esp_1p_nw', 'male',
        'female', 'employed', 'unemployed', 'armed_forces', 'in_lf',
        'nilf', 'work_from_home', 'commuter', 'has_job',
        'own_ch_u6', 'own_ch_6_11', 'own_ch_12_17',
        'esp_2p_2w_age_u6', 'esp_2p_2w_age_6_17', 'esp_2p_1w_age_u6',
        'esp_2p_1w_age_6_17', 'esp_2p_nw_age_u6', 'esp_2p_nw_age_6_17',
        'esp_1p_1w_age_u6', 'esp_1p_1w_age_6_17', 'esp_1p_nw_age_u6', 'esp_1p_nw_age_6_17',
        ## commuter sums by household, for generating workplaces:
        'com_LODES_low', 'com_LODES_high',
        *['com_ind_'+k for k in ind_codes.keys()],
        *['com_occ_'+k for k in occ_codes.keys()]]
    hsamp[h_summcols].to_csv(os.path.join('processed','hh_samples.csv'))

    ## summary of individual samples
    ## will be used to look up attributes of individuals selected for synth pop
    ## and for generating group quarters
    p_rawcols = ['SERIALNO','PWGTP','PUMA','ST','RELSHIPP','AGEP','SEX','WAGP','PINCP','PERNP',
                'COW','JWTRNS','POWPUMA','POWSP','SCH','SCHG','SCHL','WKL','WKW','ESR','NAICSP','SOCP',
                'sch_grade','st_puma']
    p_summcols = [*p_rawcols,'work_from_home','commuter','has_job','com_LODES_low','com_LODES_high','armed_forces',
                  ## all employed, by industry and occ (will be used to generate GQ employment data)
                  *['ind_'+k for k in ind_codes.keys()], *['occ_'+k for k in occ_codes.keys()],
                  *more_summary_cols]
    p_summary = psamp[p_summcols].reset_index(drop=True).copy(deep=True)
    col_idx = len(p_rawcols)
    p_summary[p_summary.columns[col_idx:]] = p_summary[p_summary.columns[col_idx:]].astype("Int64")
    p_summary.to_csv(os.path.join('processed','p_samples.csv'))

    ## merge geo data into return val
    p_summary = p_summary.merge(puma_to_cbsa, how='left', on='st_puma').merge(puma_to_county, how='left', on='st_puma')
    p_summary['cbsa'] = p_summary['cbsa'].fillna("none")

    return p_summary


## puma and other geo data for cbg's in the synth area
##
## from https://www.census.gov/programs-surveys/geography/guidance/geo-areas/pumas.html
##  need census tract to PUMA, filename *Census_Tract_to*PUMA*.*
##
## from geocorr https://mcdc.missouri.edu/applications/geocorr2018.html
## need:
##  cbg to cbsa, rename *cbg_to_cbsa*.*
##  cbg to urban-rural portion, rename to *cbg_urban_rural*.*
##
## put into folder "geo"
def read_geo_xwalk(cbg_index):

    file = glob(os.path.join("geo","*Census_Tract_to*PUMA*.*"))[0]
    tpum = pd.read_csv(file,dtype=str)
    tpum['st_puma'] = tpum['STATEFP']+tpum['PUMA5CE']
    tpum['tract'] = tpum['STATEFP']+tpum['COUNTYFP']+tpum['TRACTCE']
    ## note, population 0 cbgs are not in geocorr
    cbg_geo = pd.DataFrame({'Geo':cbg_index, 
                            'tract':cbg_index.map(lambda x: x[0:-1]),
                            'county':cbg_index.map(lambda x: x[0:5])})
    cbg_geo = cbg_geo.merge(tpum, how='left', on='tract').set_index('Geo', verify_integrity=True)
    file = glob(os.path.join("geo","*cbg_to_cbsa*.*"))[0]
    cbg_to_cbsa = pd.read_csv(file,dtype=str,skiprows=[1],usecols=["county","tract","bg","cbsa"])
    cbg_to_cbsa['Geo'] = cbg_to_cbsa['county'] + \
        cbg_to_cbsa['tract'].map(lambda x: x[0:4]) + cbg_to_cbsa['tract'].map(lambda x: x[5:]) + \
        cbg_to_cbsa['bg']
    cbg_to_cbsa.set_index('Geo', inplace=True, verify_integrity=True)
    file = glob(os.path.join("geo","*cbg_urban_rural*.*"))[0]
    cbg_ur = pd.read_csv(file,dtype=str,skiprows=[1],usecols=["county","tract","bg","ur","pop10","afact"])
    cbg_ur['Geo'] = cbg_ur['county'] + \
        cbg_ur['tract'].map(lambda x: x[0:4]) + cbg_ur['tract'].map(lambda x: x[5:]) + \
        cbg_ur['bg']
    cbg_ur = cbg_ur[['Geo','ur','afact']]
    cbg_ur = cbg_ur.pivot(index='Geo', columns='ur', values='afact').fillna(0)
    cbg_geo = cbg_geo.join(cbg_to_cbsa['cbsa']).join(cbg_ur)
    cbg_geo.loc[cbg_geo['cbsa']==" ", 'cbsa'] = "none"

    return cbg_geo


##
## group quarters
##
## group quarters by age groups, based on ACS
def generate_gq(geos, df_adults_in_hh, geo_xwalk, p_summary, ind_codes, occ_codes):

    print("generating GQ data")

    # everyone, by age group
    B01001 = read_acs('B01001',geos)
    ## everyone, incl group quarters
    B09019 = read_acs('B09019',geos)
    ## adults 65+ incl gq
    B09020 = read_acs('B09020',geos)

    B01001['n_total'] = B01001['B01001:']
    B01001['under18_total'] = B01001['B01001:Male:Under 5 years'] + B01001['B01001:Male:5 to 9 years'] \
        + B01001['B01001:Male:10 to 14 years'] + B01001['B01001:Male:15 to 17 years'] \
        + B01001['B01001:Female:Under 5 years'] + B01001['B01001:Female:5 to 9 years'] \
        + B01001['B01001:Female:10 to 14 years'] + B01001['B01001:Female:15 to 17 years']
    B01001['adult_total'] = B01001['n_total'] - B01001['under18_total']

    df_gq = B01001.join(df_adults_in_hh['B09021:'].rename('adult_total_in_hh')) \
                .join(B09019['B09019:In households:'].rename('n_total_in_hh')) \
                .join(B09019['B09019:In group quarters'].rename('n_group_quarters')) \
                .join(B09020['B09020:In households:'].rename('adult_65o_in_household')) \
                .join(B09020['B09020:In group quarters'].rename('adult_65o_in_group_quarters'))

    df_gq['adult_in_group_quarters'] = df_gq['adult_total'] - df_gq['adult_total_in_hh']
    df_gq['group quarters:'] = df_gq['n_group_quarters']
    df_gq['group quarters:under 18'] = df_gq['n_group_quarters'] - df_gq['adult_in_group_quarters']
    df_gq['group quarters:18 to 64'] = df_gq['adult_in_group_quarters'] - df_gq['adult_65o_in_group_quarters']
    df_gq['group quarters:65 and over'] = df_gq['adult_65o_in_group_quarters']

    ## ignore cbgs with less than 20 in gq's
    df_gq = df_gq.loc[df_gq['group quarters:'] > 19, 
            ['group quarters:', 'group quarters:under 18', 'group quarters:18 to 64', 'group quarters:65 and over']].copy(deep=True)

    ## from previous decennial census
    ##   institutionalized vs non-inst gq by age (not available in ACS)
    P43 = read_decennial('P43', geos)

    P43 = P43.loc[P43['P43']>0 ,['P43','P43:Male:Under 18 years',
    'P43:Male:Under 18 years:Institutionalized population (101-106, 201-203, 301, 401-405)',
    'P43:Male:Under 18 years:Noninstitutionalized population (501, 601-602, 701-702, 704, 706, 801-802, 900-901, 903-904)',
    'P43:Male:18 to 64 years',
    'P43:Male:18 to 64 years:Institutionalized population (101-106, 201-203, 301, 401-405)',
    'P43:Male:18 to 64 years:Noninstitutionalized population (501, 601-602, 701-702, 704, 706, 801-802, 900-901, 903-904)',
    'P43:Male:18 to 64 years:Noninstitutionalized population (501, 601-602, 701-702, 704, 706, 801-802, 900-901, 903-904):Military quarters (601-602)',
    'P43:Male:65 years and over',
    'P43:Male:65 years and over:Institutionalized population (101-106, 201-203, 301, 401-405)',
    'P43:Male:65 years and over:Noninstitutionalized population (501, 601-602, 701-702, 704, 706, 801-802, 900-901, 903-904)',
    'P43:Female', 'P43:Female:Under 18 years',
    'P43:Female:Under 18 years:Institutionalized population (101-106, 201-203, 301, 401-405)',
    'P43:Female:Under 18 years:Noninstitutionalized population (501, 601-602, 701-702, 704, 706, 801-802, 900-901, 903-904)',
    'P43:Female:18 to 64 years:Noninstitutionalized population (501, 601-602, 701-702, 704, 706, 801-802, 900-901, 903-904):Military quarters (601-602)',
    'P43:Female:18 to 64 years',
    'P43:Female:18 to 64 years:Institutionalized population (101-106, 201-203, 301, 401-405)',
    'P43:Female:18 to 64 years:Noninstitutionalized population (501, 601-602, 701-702, 704, 706, 801-802, 900-901, 903-904)',
    'P43:Female:65 years and over',
    'P43:Female:65 years and over:Institutionalized population (101-106, 201-203, 301, 401-405)',
    'P43:Female:65 years and over:Noninstitutionalized population (501, 601-602, 701-702, 704, 706, 801-802, 900-901, 903-904)']].copy(deep=True)

    P43['u18_gq'] = P43['P43:Male:Under 18 years'] + P43['P43:Female:Under 18 years']
    P43['18_64_gq'] = P43['P43:Male:18 to 64 years'] + P43['P43:Female:18 to 64 years']
    P43['65o_gq'] = P43['P43:Male:65 years and over'] + P43['P43:Female:65 years and over']

    P43['u18_inst'] = (P43['P43:Male:Under 18 years:Institutionalized population (101-106, 201-203, 301, 401-405)'] +
        P43['P43:Female:Under 18 years:Institutionalized population (101-106, 201-203, 301, 401-405)'])
    P43['u18_noninst'] = (P43['P43:Male:Under 18 years:Noninstitutionalized population (501, 601-602, 701-702, 704, 706, 801-802, 900-901, 903-904)'] +
        P43['P43:Female:Under 18 years:Noninstitutionalized population (501, 601-602, 701-702, 704, 706, 801-802, 900-901, 903-904)'])

    P43['18_64_inst'] = (P43['P43:Male:18 to 64 years:Institutionalized population (101-106, 201-203, 301, 401-405)'] +
        P43['P43:Female:18 to 64 years:Institutionalized population (101-106, 201-203, 301, 401-405)'])
    P43['18_64_noninst'] = (P43['P43:Male:18 to 64 years:Noninstitutionalized population (501, 601-602, 701-702, 704, 706, 801-802, 900-901, 903-904)'] +
        P43['P43:Female:18 to 64 years:Noninstitutionalized population (501, 601-602, 701-702, 704, 706, 801-802, 900-901, 903-904)'])
    P43['18_64_noninst_mil'] = (P43['P43:Male:18 to 64 years:Noninstitutionalized population (501, 601-602, 701-702, 704, 706, 801-802, 900-901, 903-904):Military quarters (601-602)'] +
        P43['P43:Female:18 to 64 years:Noninstitutionalized population (501, 601-602, 701-702, 704, 706, 801-802, 900-901, 903-904):Military quarters (601-602)'])
    P43['18_64_noninst_civil'] = (P43['18_64_noninst'] - P43['18_64_noninst_mil'])

    P43['65o_inst'] = (P43['P43:Male:65 years and over:Institutionalized population (101-106, 201-203, 301, 401-405)'] +
        P43['P43:Female:65 years and over:Institutionalized population (101-106, 201-203, 301, 401-405)'])
    P43['65o_noninst'] = (P43['P43:Male:65 years and over:Noninstitutionalized population (501, 601-602, 701-702, 704, 706, 801-802, 900-901, 903-904)'] +
        P43['P43:Female:65 years and over:Noninstitutionalized population (501, 601-602, 701-702, 704, 706, 801-802, 900-901, 903-904)'])

    ## will use previous census proportions with ACS counts
    census_gq_u18 = P43.loc[P43['u18_gq']>0, ['u18_gq','u18_inst','u18_noninst']].copy(deep=True)
    census_gq_18_64 = P43.loc[P43['18_64_gq']>0, ['18_64_gq','18_64_inst','18_64_noninst','18_64_noninst_civil','18_64_noninst_mil']].copy(deep=True)
    census_gq_65o = P43.loc[P43['65o_gq']>0, ['65o_gq','65o_inst','65o_noninst']].copy(deep=True)
    census_gq_u18['p_u18_inst'] = census_gq_u18['u18_inst'] / census_gq_u18['u18_gq']
    census_gq_18_64['p_18_64_inst'] = census_gq_18_64['18_64_inst'] / census_gq_18_64['18_64_gq']
    census_gq_18_64['p_18_64_noninst_civil'] = census_gq_18_64['18_64_noninst_civil'] / census_gq_18_64['18_64_gq']
    census_gq_18_64['p_18_64_noninst_mil'] = census_gq_18_64['18_64_noninst_mil'] / census_gq_18_64['18_64_gq']
    census_gq_65o['p_65o_inst'] = census_gq_65o['65o_inst'] / census_gq_65o['65o_gq']

    df_gq = df_gq.join([census_gq_u18,census_gq_18_64,census_gq_65o])

    ## if census is missing data for CBGs where ACS reports GQ residents,
    ##   assume -18 and 65+ are inst, assume 18-64 are non-inst civilian
    for c in ["p_u18_inst","p_18_64_noninst_civil","p_65o_inst"]:
        df_gq[c] = df_gq[c].fillna(1.0)
    for c in ["p_18_64_inst","p_18_64_noninst_mil"]:
        df_gq[c] = df_gq[c].fillna(0.0)

    ## from puma samples, employment data on people in non-inst group quarters
    ##  (not avail in census data)
    ##
    ## assume only 18-64 are working (not true, but close enough)
    ##
    ## PUMS doesn't distinguish btw civilian and military group quarters
    ## use employment status code (ESR):
    ##  assume that everyone in military GQ is employed in armed forces
    ##  and anyone in GQ not employed by armed forces is in civilian GQ
    s_noninst_civ = p_summary.loc[p_summary['RELSHIPP'].isin(['38']) 
                & (p_summary['armed_forces'] != 1)
                & (p_summary['AGEP'] > 17) 
                & (p_summary['AGEP'] < 65)].copy(deep=True)

    ## note, employment stats generated here still need to be reconciled with census data in next step
    cols = ['commuter','work_from_home','com_LODES_low','com_LODES_high',
            *['ind_'+k for k in ind_codes.keys()], *['occ_'+k for k in occ_codes.keys()]]

    pcols = [c+'_p|ninst1864civ' for c in cols]

    ## PWGTP = individual "weight" according to the pums sample data
    for c,pc in zip(cols,pcols):
        s_noninst_civ[pc] = s_noninst_civ[c] * s_noninst_civ['PWGTP']

    s_noninst_mil = p_summary.loc[p_summary['RELSHIPP'].isin(['38']) 
                & (p_summary['armed_forces'] == 1)
                & (p_summary['AGEP'] > 17) 
                & (p_summary['AGEP'] < 65)].copy(deep=True)

    cols_mil = [c+'_p|milGQ' for c in cols]

    for c,pc in zip(cols,cols_mil):
        s_noninst_mil[pc] = s_noninst_mil[c] * s_noninst_mil['PWGTP']

    ## not many GQ residents represented in PUMS samples
    ## not enough to capture geographic variation in GQ employment stats
    ## assume that all GQs of the same type (civ, mil) have the same emp stats
    agg_civ = s_noninst_civ[['PWGTP',*pcols]].sum()
    p_agg_civ = agg_civ.apply(lambda x: x/agg_civ["PWGTP"]).drop("PWGTP").fillna(0.0)

    ## a hack in case PUMS doesn't have enough samples
    ##
    ## TODO: get national-level data to fall back on?
    ##
    if len(s_noninst_mil) < 20:
        agg_mil = pd.Series({k:0 for k in ['PWGTP',*cols_mil]})
        agg_mil["PWGTP"] = 1
        agg_mil["commuter_p|milGQ"] = 1
        agg_mil["com_LODES_low_p|milGQ"] = 1
        agg_mil["ind_ADM_MIL_p|milGQ"] = 1
        agg_mil["occ_MIL_p|milGQ"] = 1
    else:
        agg_mil = s_noninst_mil[['PWGTP',*cols_mil]].sum()

    p_agg_mil = agg_mil.apply(lambda x: x/agg_mil['PWGTP']).drop("PWGTP").fillna(0.0)
    
    df_gq[p_agg_civ.index] = p_agg_civ.values
    df_gq[p_agg_mil.index] = p_agg_mil.values

    keep_cols = ['group quarters:', 'group quarters:under 18', 'group quarters:18 to 64', 'group quarters:65 and over', 
        'p_u18_inst', 'p_18_64_inst', 'p_65o_inst', 'p_18_64_noninst_civil', 'p_18_64_noninst_mil',
        *pcols, *cols_mil]

    ##
    ## a couple of cbgs in the ACS aren't in the tract-to-puma xwalk
    ## this shouldn't happen, as ACS -2019 uses the 2010 cbg boundaries
    ## drop them from the synth pop, I guess?
    ## (they're also missing from the OD commute data and from geocorr2018)
    ##
    missing_puma = geo_xwalk.index[geo_xwalk['st_puma'].isna()]
    keep_rows = ~df_gq.index.isin(missing_puma)

    df_gq = df_gq.loc[keep_rows, keep_cols].copy(deep=True)
    df_gq.round(6).to_csv(os.path.join('processed','group_quarters.csv'))
    return df_gq


def read_ind_df(geos, ind_codes):
    ## industries
    C24030 = read_acs("C24030",geos)
    ## civilian vs military workforce
    B23025 = read_acs("B23025",geos)

    ## join male and female counts
    ind_names = C24030.columns[1:].map(lambda x: ':'.join(x.split(":")[2:])).unique()
    for x in ind_names:
        C24030["C24030:All:"+x] = C24030["C24030:Male:"+x] + C24030["C24030:Female:"+x]

    C24030 = C24030.join(B23025['B23025:In labor force:Armed Forces'])
    C24030["C24030:All:MIL"] = C24030['B23025:In labor force:Armed Forces']
    C24030["C24030:All:Civilian"] = C24030["C24030:All:"]
    C24030["C24030:All:"] = C24030["C24030:All:"] + C24030["C24030:All:MIL"]
    C24030["C24030:"] = C24030["C24030:All:"]

    ## industry code 92 = public admin and mil
    C24030["C24030:All:ADM_MIL"] = C24030["C24030:All:Public administration"] + C24030["C24030:All:MIL"]

    ind_df = C24030[["C24030:"]].copy(deep=True)
    for k in ind_codes:
        ind_df = ind_df.join(C24030['C24030:All:'+ind_codes[k][1]].rename('C24030:'+k))

    return ind_df


def read_occ_df(geos, occ_codes):
    ## occupations
    C24010 = read_acs("C24010",geos)
    ## civilian vs military workforce
    B23025 = read_acs("B23025",geos)

    ## join male and female counts
    occ_names = C24010.columns[1:].map(lambda x: ':'.join(x.split(":")[2:])).unique()
    for x in occ_names:
        C24010["C24010:All:"+x] = C24010["C24010:Male:"+x] + C24010["C24010:Female:"+x]

    ## add military to industries and occupations
    C24010 = C24010.join(B23025['B23025:In labor force:Armed Forces'])
    C24010["C24010:All:MIL"] = C24010['B23025:In labor force:Armed Forces']
    C24010["C24010:All:Civilian"] = C24010["C24010:All:"]
    C24010["C24010:All:"] = C24010["C24010:All:"] + C24010["C24010:All:MIL"]
    C24010["C24010:"] = C24010["C24010:All:"]

    ## pull education and health out of the mbsa category
    C24010['C24010:All:EDU'] = C24010['C24010:All:Management, business, science, and arts occupations:Education, legal, community service, arts, and media occupations:Educational instruction, and library occupations']
    C24010['C24010:All:MED'] = C24010['C24010:All:Management, business, science, and arts occupations:Healthcare practitioners and technical occupations:']
    C24010['C24010:All:MBSA'] = C24010['C24010:All:Management, business, science, and arts occupations:'] - (C24010['C24010:All:EDU'] + C24010['C24010:All:MED'])

    occ_df = C24010[["C24010:"]].copy(deep=True)
    for k in occ_codes:
        occ_df = occ_df.join(C24010['C24010:All:'+occ_codes[k][1]].rename('C24010:'+k))

    return occ_df


##
## ACS, census block group, from data.census.gov
## put each state in its own folder inside folder named "census"
##
def generate_targets(target_columns, geos, geo_xwalk, gq_stats, inc_cats, inc_cols, ind_codes, occ_codes):
    print("reading census data")

    ##
    ## a couple of cbgs in the ACS aren't in the tract-to-puma xwalk
    ## this shouldn't happen, as ACS -2019 uses the 2010 cbg boundaries
    ## drop them from the synth pop, I guess?
    ## (they're also missing from the OD commute data and from geocorr2018)
    ##
    has_puma_idx = geo_xwalk.index[~geo_xwalk['st_puma'].isna()]

    ## family / nonfamily households by size
    B11016 = read_acs('B11016',geos)
    ## household types married/cohab/single/alone/own_ch_u18/other_rels/only_nonrels
    B11012 = read_acs('B11012',geos)
    ## family households, by # workers _in family_ (not other workers in hh), presence of own_ch_u18, and marriage status
    B23009 = read_acs('B23009',geos)
    ## family households, by marriage status and presence of _related_ children in age groups
    ##  difference between this and "own" children is usually grandchildren of householder
    ## match to HUPARC
    B11004 = read_acs('B11004',geos)
    ## household income
    B19001 = read_acs('B19001',geos)
    ## household received food stamps
    B22010 = read_acs('B22010',geos)
    # 18+ in households
    #   If a householder has no spouse or unmarried partner present, they will be shown in 'other relatives' if they have at least one relative present, or in 'other nonrelatives' if no relatives are present.
    B09021 = read_acs('B09021',geos)
    ## children under 18 in households by relationship
    ## note, this is the only cbg-level table that counts all u18 children in households
    B09018 = read_acs('B09018',geos)
    ## civilian vs military workforce
    B23025 = read_acs("B23025",geos)
    ## everyone, incl group quarters
    B09019 = read_acs('B09019',geos)
    ## race and ethnicity
    B11001H = read_acs("B11001H",geos) ## white non-hispanic householders
    B11001I = read_acs("B11001I",geos) ## hispanic householders
    B25006 = read_acs("B25006",geos) ## householders by race
    race_eth = B25006[["B25006:Householder who is Black or African American alone"]] \
        .join(B11001H["B11001H:"].rename("B11001H:Householder white non-hispanic")) \
        .join(B11001I["B11001I:"].rename("B11001I:Householder hispanic any race"))

    ## combine m and f single householders
    for x in ['With own children of the householder under 18 years','No own children of the householder under 18 years']:
        for y in ['No workers','1 worker','2 workers','3 or more workers']:
            B23009[':'.join(['B23009',x,'Other family','Unmarried householder',y])] = \
                B23009[':'.join(['B23009',x,'Other family','Male householder, no spouse present',y])] + \
                B23009[':'.join(['B23009',x,'Other family','Female householder, no spouse present',y])]

    B11004['B11004:Other family:Unmarried householder:No related children of the householder under 18 years'] = \
        B11004['B11004:Other family:Male householder, no spouse present:No related children of the householder under 18 years'] + \
        B11004['B11004:Other family:Female householder, no spouse present:No related children of the householder under 18 years']

    for x in ['Under 6 years only','Under 6 years and 6 to 17 years','6 to 17 years only']:
        B11004[':'.join(['B11004','Other family:Unmarried householder:With related children of the householder under 18 years',x])] = \
            B11004[':'.join(['B11004','Other family:Male householder, no spouse present:With related children of the householder under 18 years',x])] + \
            B11004[':'.join(['B11004','Other family:Female householder, no spouse present:With related children of the householder under 18 years',x])]

    for x in ['Living alone','With own children under 18 years','With relatives, no own children under 18 years','With only nonrelatives present']:
        B11012[':'.join(['B11012','Single householder',x])] = \
            B11012[':'.join(['B11012','Female householder, no spouse or partner present',x])] + \
            B11012[':'.join(['B11012','Male householder, no spouse or partner present',x])]

    ## combine married and cohab households
    B11012['B11012:Two-partner household:With own children under 18 years'] = \
        B11012['B11012:Married-couple household:With own children under 18 years'] + \
        B11012['B11012:Cohabiting couple household:With own children of the householder under 18 years']

    B11012['B11012:Two-partner household:With no own children under 18 years'] = \
        B11012['B11012:Married-couple household:With no own children under 18 years'] + \
        B11012['B11012:Cohabiting couple household:With no own children of the householder under 18 years']

    ## combine married and unmarried partners
    B09021['B09021:Householder living with partner or partner of householder'] = \
        B09021['B09021:Householder living with spouse or spouse of householder'] + B09021['B09021:Householder living with unmarried partner or unmarried partner of householder']

    for x in ['18 to 34 years','35 to 64 years','65 years and over']:
        B09021[':'.join(['B09021',x,'Householder living with partner or partner of householder'])] = \
            B09021[':'.join(['B09021',x,'Householder living with spouse or spouse of householder'])] + \
            B09021[':'.join(['B09021',x,'Householder living with unmarried partner or unmarried partner of householder'])]

    ## make occupation and industry dataframe with our custom categories
    ind_df = read_ind_df(geos,ind_codes)
    occ_df = read_occ_df(geos,occ_codes)

    ## adjust ACS industry and occ counts using estimated GQ counts
    ## (industry and occupation expected to be different for gq vs households)
    print("estimating household employment counts")

    ind_civ_gq = gq_stats[['ind_'+k+'_p|ninst1864civ' for k in ind_codes.keys()]].copy(deep=True) \
        .apply(lambda s: s * gq_stats['p_18_64_noninst_civil'] * gq_stats['group quarters:18 to 64']) \
        .rename(columns={
            'ind_'+k+'_p|ninst1864civ':'C24030:'+k for k in ind_codes.keys()})

    ind_mil_gq = gq_stats[['ind_'+k+'_p|milGQ' for k in ind_codes.keys()]].copy(deep=True) \
        .apply(lambda s: s * gq_stats['p_18_64_noninst_mil'] * gq_stats['group quarters:18 to 64']) \
        .rename(columns={
            'ind_'+k+'_p|milGQ':'C24030:'+k for k in ind_codes.keys()})

    occ_civ_gq = gq_stats[['occ_'+k+'_p|ninst1864civ' for k in occ_codes.keys()]].copy(deep=True) \
        .apply(lambda s: s * gq_stats['p_18_64_noninst_civil'] * gq_stats['group quarters:18 to 64']) \
        .rename(columns={
            'occ_'+k+'_p|ninst1864civ':'C24010:'+k for k in occ_codes.keys()})

    occ_mil_gq = gq_stats[['occ_'+k+'_p|milGQ' for k in occ_codes.keys()]].copy(deep=True) \
        .apply(lambda s: s * gq_stats['p_18_64_noninst_mil'] * gq_stats['group quarters:18 to 64']) \
        .rename(columns={
            'occ_'+k+'_p|milGQ':'C24010:'+k for k in occ_codes.keys()})

    ## use IPF to get hh and gq workers by industry
    ## assume total (hh+gq) workers in ACS are correct, including industry/occ totals
    ## assume GQ workers total are correct, unless remaining hh workers would be outside min max bounds
    ind_col_totals = ind_df[['C24030:'+k for k in ind_codes.keys()]].copy(deep=True)
    occ_col_totals = occ_df[['C24010:'+k for k in occ_codes.keys()]].copy(deep=True)

    ## calculate minimum # workers in hh
    min_hh_B23009 = (B23009['B23009:With own children of the householder under 18 years:Married-couple family:1 worker']
        + 2*B23009['B23009:With own children of the householder under 18 years:Married-couple family:2 workers:']
        + 3*B23009['B23009:With own children of the householder under 18 years:Married-couple family:3 or more workers:']
        + B23009['B23009:No own children of the householder under 18 years:Married-couple family:1 worker']
        + 2*B23009['B23009:No own children of the householder under 18 years:Married-couple family:2 workers:']
        + 3*B23009['B23009:No own children of the householder under 18 years:Married-couple family:3 or more workers:']
        + B23009['B23009:With own children of the householder under 18 years:Other family:Unmarried householder:1 worker']
        + 2*B23009['B23009:With own children of the householder under 18 years:Other family:Unmarried householder:2 workers']
        + 3*B23009['B23009:With own children of the householder under 18 years:Other family:Unmarried householder:3 or more workers']
        + B23009['B23009:No own children of the householder under 18 years:Other family:Unmarried householder:1 worker']
        + 2*B23009['B23009:No own children of the householder under 18 years:Other family:Unmarried householder:2 workers']
        + 3*B23009['B23009:No own children of the householder under 18 years:Other family:Unmarried householder:3 or more workers']
        ).rename("min_hh")

    df_bounds = pd.DataFrame({'total_workers':ind_df["C24030:"]})
    df_bounds = df_bounds.join(min_hh_B23009) 
    ## in some places minimum # "workers" per table B23009 = more than total people in labor force??
    df_bounds["min_hh"] = df_bounds[["total_workers","min_hh"]].min(axis=1)
    ## using total people in hh as max (no other reliable max; B23009 only counts families)
    df_bounds = df_bounds.join(B09019["B09019:In households:"].rename("max_hh"))

    df_bounds = df_bounds \
                .join(ind_civ_gq.sum(axis=1).rename("civ_gq_workers")) \
                .join(ind_mil_gq.sum(axis=1).rename("mil_gq_workers")) \
                .fillna(0.0)

    ## assuming everyone living in miliary gq works in armed forces:
    ## can't have more mil gq workers than # armed forces in ACS table B23025
    df_bounds = df_bounds.join(B23025["B23025:In labor force:Armed Forces"].rename("max_mil"))
    df_bounds["mil_gq_workers"] = df_bounds[["mil_gq_workers","max_mil"]].min(axis=1)

    ## total hh workers = total workers - gq workers
    df_bounds["_gq_workers"] = df_bounds["civ_gq_workers"] + df_bounds["mil_gq_workers"]
    df_bounds["_p_civ"] = (df_bounds["civ_gq_workers"] / df_bounds["_gq_workers"]).fillna(0.0)
    df_bounds["_p_mil"] = (df_bounds["mil_gq_workers"] / df_bounds["_gq_workers"]).fillna(0.0)
    df_bounds["hh_workers"] = df_bounds["total_workers"] - df_bounds["_gq_workers"]
    df_bounds["hh_workers"] = df_bounds[["hh_workers","min_hh"]].max(axis=1)
    df_bounds["hh_workers"] = df_bounds[["hh_workers","max_hh"]].min(axis=1)

    ## adjust estimated gq totals where hh workers were outside min max bounds
    df_bounds["_gq_workers"] = df_bounds["total_workers"] - df_bounds["hh_workers"]
    df_bounds["civ_gq_workers"] = (df_bounds["_p_civ"] * df_bounds["_gq_workers"])#.round(0)
    df_bounds["mil_gq_workers"] = (df_bounds["_p_mil"] * df_bounds["_gq_workers"])#.round(0)

    ## initial starting point for hh workers by ind/occ = ACS sums * hh workers / total workers
    df_bounds["_p_hh"] = (df_bounds["hh_workers"] / df_bounds["total_workers"]).fillna(0.0)
    ind_df = ind_df.join(df_bounds[["_p_hh"]])
    occ_df = occ_df.join(df_bounds[["_p_hh"]])

    ind_hh = ind_df[['C24030:'+k for k in ind_codes.keys()]].copy(deep=True) \
        .apply(lambda s: s * ind_df["_p_hh"])

    occ_hh = occ_df[['C24010:'+k for k in occ_codes.keys()]].copy(deep=True) \
        .apply(lambda s: s * occ_df["_p_hh"])

    ## IPF for each location that has gq workers
    idxs = gq_stats.index.intersection(has_puma_idx).intersection(df_bounds.index[df_bounds["_gq_workers"]>0])
    df_bounds = df_bounds.loc[idxs,:].copy(deep=True)

    if not np.all([np.isclose(df_bounds["civ_gq_workers"] + df_bounds["mil_gq_workers"] , df_bounds["_gq_workers"]).all(),
    np.isclose(df_bounds["_gq_workers"] + df_bounds["hh_workers"] , df_bounds["total_workers"]).all(),
    (df_bounds["hh_workers"] <= df_bounds["max_hh"]).all(),
    (df_bounds["hh_workers"] >= df_bounds["min_hh"]).all()]):
        print("warning: could not reconcile household and GQ employment sums for some locations")

    ## industries
    for i in idxs:
        ## rows are: hh, civ gq, mil gq
        rowsums = df_bounds.loc[i,["hh_workers","civ_gq_workers","mil_gq_workers"]].values

        ## columns are industries/occupations
        colsums = ind_col_totals.loc[i,:].values

        m = np.stack([ind_hh.loc[i,:].astype(float).values,
                    ind_civ_gq.loc[i,:].astype(float).values, 
                    ind_mil_gq.loc[i,:].astype(float).values])

        ## ipf handles 0's poorly; replace with 0.5
        IPF = ipfn.ipfn(np.maximum(m,0.5), [rowsums,colsums], [[0], [1]], convergence_rate=1e-6, max_iteration=1000, rate_tolerance=0.0)
        new_m = IPF.iteration()

        ## update initial dfs
        ind_hh.loc[i,:] = new_m[0,:]
        ind_civ_gq.loc[i,:] = new_m[1,:]
        ind_mil_gq.loc[i,:] = new_m[2,:]
        
    ## occupations
    for i in idxs:
        ## rows are: hh, civ gq, mil gq
        rowsums = df_bounds.loc[i,["hh_workers","civ_gq_workers","mil_gq_workers"]].values

        ## columns are industries/occupations
        colsums = occ_col_totals.loc[i,:].values

        m = np.stack([occ_hh.loc[i,:].astype(float).values,
                    occ_civ_gq.loc[i,:].astype(float).values, 
                    occ_mil_gq.loc[i,:].astype(float).values])

        ## ipf handles 0's poorly; replace with 0.5
        IPF = ipfn.ipfn(np.maximum(m,0.5), [rowsums,colsums], [[0], [1]], convergence_rate=1e-6, max_iteration=1000, rate_tolerance=0.0)
        new_m = IPF.iteration()

        ## update initial dfs
        occ_hh.loc[i,:] = new_m[0,:]
        occ_civ_gq.loc[i,:] = new_m[1,:]
        occ_mil_gq.loc[i,:] = new_m[2,:]
        
    ## round to integer (preserve row sums)
    ind_hh = ind_hh.apply(lrRound,axis=1)
    ind_civ_gq = ind_civ_gq.apply(lrRound,axis=1)
    ind_mil_gq = ind_mil_gq.apply(lrRound,axis=1)
    occ_hh = occ_hh.apply(lrRound,axis=1)
    occ_civ_gq = occ_civ_gq.apply(lrRound,axis=1)
    occ_mil_gq = occ_mil_gq.apply(lrRound,axis=1)

    ## save gq employment counts for synth pop construction
    ind_civ_gq.join(occ_civ_gq).to_csv(os.path.join('processed','gq_civilian_workers.csv'))
    ind_mil_gq.join(occ_mil_gq).to_csv(os.path.join('processed','gq_military_workers.csv'))

    test_r = df_bounds \
        .join(ind_civ_gq.sum(axis=1).rename("ipf_civ")) \
        .join(ind_mil_gq.sum(axis=1).rename("ipf_mil")) \
        .join(ind_hh.sum(axis=1).rename("ipf_hh")) \
        .fillna(0.0) \
        .loc[idxs,:]

    test_c = ind_hh.add(ind_civ_gq,fill_value=0).add(ind_mil_gq,fill_value=0).loc[idxs,:]
    ref_c = ind_df.loc[idxs,ind_df.columns[1:-1]]
    x_r = (test_r["hh_workers"] - test_r["ipf_hh"]).apply(abs) > 5.0
    x_c = test_c.sub(ref_c).apply(abs).apply(lambda s: (s > 5.0).any(),axis=1)

    if x_r.sum() + x_c.sum() > 0:
        print("warning: IPF could not reconcile household and GQ employment counts for some locations")

    ## join all census tables together (this is household data, for CO targets)
    acs_tables = B11016.join([B11012,B23009,B11004,B19001,B22010,B09018,B09021,race_eth,ind_hh,occ_hh])
    acs_tables['state'] = acs_tables.index.map(lambda x: x[0:2])
    acs_tables['county'] = acs_tables.index.map(lambda x: x[0:5])

    ## income data
    for k,v in zip(inc_cats,inc_cols):
        acs_tables['B19001:'+k] = acs_tables[['B19001:'+x for x in v]].sum(axis=1)

    print("writing census targets")
    ## csv of household counts by geography
    B11012.loc[has_puma_idx,'B11012:'].to_csv(os.path.join('processed','hh_counts.csv'))
    ## drop cbgs with less than 20 hh
    keep_cbg_idx = B11012.index[B11012['B11012:'] > 19].intersection(has_puma_idx)

    ## csv of target columns from census data
    acs_tables.loc[keep_cbg_idx,target_columns].to_csv(os.path.join('processed','acs_targets.csv'))
    ## csv of cbg geo data
    geo_xwalk.loc[keep_cbg_idx,:].to_csv(os.path.join('processed','cbg_geo.csv'))

    return None



##
## workplaces
##
## origin-destination work commute data
## https://lehd.ces.census.gov/data/
## Origin-Destination (OD) File Structure
## 1 w_geocode Char15 Workplace Census Block Code
## 2 h_geocode Char15 Residence Census Block Code
## 3 S000 Num Total number of jobs
##
## use JT01, "primary" jobs (because JT00 counts 2+ jobs for the same individual)
## "main" = work and live in state
## "aux" = work in state, live outside state
## 
## need:
##  main file for every state in the synth area, named *od_main_JT01*
##  aux file for every state in the synth area, named *od_aux_JT01*
##  optionally, aux files for other states to capture commute patterns outside the above state(s)
##
## also need:
##   WAC (work area chars) file for each state in synth area, *wac_S000_JT01*
##
## put into folder "work"

## this fn converts series to proportions
##    unless the series is all zeros, then p = 1 / length
def p_or_frac(ser):
    tot = ser.sum()
    if tot==0:
        return np.ones(len(ser)) / len(ser)
    else:
        return ser / tot

## calculates origin-destination commute matrix and related info for geos in synth pop
def read_origin_destination(geos, commute_states):

    od = read_work_commute()

    if geos is not None:
        ## work in the synth area, live anywhere
        work_in_area = np.any([od['w_geocode'].str.startswith(g) for g in geos],axis=0)
        ## live in the synth area, work anywhere in the state(s) with od files provided
        live_in_area = np.any([od['h_geocode'].str.startswith(g) for g in geos],axis=0)
    else: 
        ## if no synth area geos specified, assume work area is synth area
        main_states = od.loc[od["main"]==True, "w_state"].unique()
        work_in_area = od["w_state"].isin(main_states)
        live_in_area = od["h_state"].isin(main_states)

    ## anyone living outside commute distance doesn't really work in the synth area
    if commute_states is not None:
        within_commute_dist = (live_in_area | od["h_state"].isin(commute_states))
        work_in_area = work_in_area & within_commute_dist

    od["work_dest"] = od["w_cbg"]
    od["origin"] = od["h_cbg"]
    od.loc[~work_in_area, "work_dest"] = "outside"
    od.loc[~live_in_area, "origin"] = "outside"

    ## drop anyone who neither works nor lives in the synth area
    od = od[work_in_area | live_in_area]

    n_commuting_in = od.loc[od.origin=="outside","S000"].sum()

    n_commuting_out = od.loc[od.work_dest=="outside", "S000"].sum()

    ## will need employer sizes for these counties, save for later
    work_counties = od["w_county"].unique()
    pd.DataFrame({"county":work_counties}).to_csv(os.path.join('processed','work_sizes.csv'))

    ## calculate proportions (not separated by income)
    od_matrix = od[['origin','work_dest','S000']] \
        .groupby(['origin','work_dest'],group_keys=True).agg({"S000":sum}) \
        .groupby(level=0,group_keys=False).apply(lambda x: x.apply(p_or_frac)) \
        .loc[:,'S000'] \
        .unstack().fillna(0)

    return (od_matrix, n_commuting_in, n_commuting_out)

## estimate workers by industry x destination from WAC data
def read_industry_by_dest(geos, ind_codes, ind_keys):

    ## map industry keys to WAC column names
    numcols = ['C000', 'CNS01', 'CNS02', 'CNS03', 'CNS04', 'CNS05', 'CNS06', 'CNS07', 'CNS08',
        'CNS09', 'CNS10', 'CNS11', 'CNS12', 'CNS13', 'CNS14', 'CNS15', 'CNS16',
        'CNS17', 'CNS18', 'CNS19', 'CNS20']

    ## from wac documentation
    wac_col_ind = {"CNS01": ["11"],
    "CNS02": ["21"],
    "CNS03": ["22"],
    "CNS04": ["23"],
    "CNS05": ["31","32","33","3M"],
    "CNS06": ["42"],
    "CNS07": ["44","45","4M"],
    "CNS08": ["48","49"],
    "CNS09": ["51"],
    "CNS10": ["52"],
    "CNS11": ["53"],
    "CNS12": ["54"],
    "CNS13": ["55"],
    "CNS14": ["56"],
    "CNS15": ["61"],
    "CNS16": ["62"],
    "CNS17": ["71"],
    "CNS18": ["72"],
    "CNS19": ["81"],
    "CNS20": ["92"],
    }

    wac_ind_col = {x:k for k in wac_col_ind for x in wac_col_ind[k]}

    ind_to_wac = {k:list(set([wac_ind_col[i] for i in ind_codes[k][0]])) for k in ind_keys}

    wac_files = glob(os.path.join("work","*wac_S000_JT01*.[c|z]*"))
    dfs = []
    for f in wac_files:
        df = pd.read_csv(f,usecols=["w_geocode",*numcols],dtype=({"w_geocode":str} | {x:"Int64" for x in numcols}))
        dfs.append(df)
    wac = pd.concat(dfs,ignore_index=True)
    wac['w_cbg'] = wac['w_geocode'].apply(lambda x: x[0:12])
    wac['w_county'] = wac['w_geocode'].apply(lambda x: x[0:5])
    wac['w_state'] = wac['w_geocode'].apply(lambda x: x[0:2])

    if geos is not None:
        in_area = np.any([wac['w_geocode'].str.startswith(g) for g in geos],axis=0)
        wac = wac.loc[in_area]

    ## sum the WAC columns corresponding to each industry
    for k in ind_keys:
        wac[k] = wac[ind_to_wac[k]].sum(axis=1).astype("Int64")

    wac["work_dest"] = wac["w_cbg"]

    ## aggregate by work destination
    wac_by_dest = wac[["work_dest", *ind_keys]].groupby("work_dest").agg(sum)
    return wac_by_dest

## write origin-destination, origin-industry, and industry-destination commute data
## to use in generating commutes by industry
def calc_commute_marginals(geos,ind_codes,ind_keys,commute_states):
    print("reading commute data")
    od_matrix, n_commuting_in, n_commuting_out = read_origin_destination(geos, commute_states)

    ind_df = read_ind_df(geos, ind_codes)
    ind_df = ind_df.astype(float).sort_index()

    ## proportion of workers by industry, same order as ind_keys
    p_by_ind = ind_df[["C24030:"+k for k in ind_keys]].values.sum(axis=0) / ind_df["C24030:"].values.sum()

    ## workers living outside synth area by industry not known
    ## estimate as n_commuting_in * p_by_ind
    ind_df.loc["outside","C24030:"] = n_commuting_in
    ind_df.loc["outside", ["C24030:"+k for k in ind_keys]] = n_commuting_in * p_by_ind
    ## save those for later; they are not generated anywhere else
    pd.DataFrame(ind_df.loc["outside", ["C24030:"+k for k in ind_keys]]).round(2).to_csv(
        os.path.join('processed','work_cats_live_outside.csv'))

    ## keep just the necessary columns, same order as ind_keys
    ind_df = ind_df.loc[:,["C24030:"+k for k in ind_keys]]

    ## this index should have all the locations in the synth pop
    origin_index = ind_df.index

    ## make the commute matrix have the same row index as the census data
    ## some census locations have no commute data; these _should_ be ones with no workers
    ##   but make those rows sum to 1 anyway
    n_destinations = od_matrix.shape[1]
    od_matrix = pd.DataFrame(index=origin_index).join(od_matrix).fillna(1.0 / n_destinations)

    ## this index has all the work locations in the synth pop
    dest_index = od_matrix.columns

    ## estimate workers by industry x destination from WAC data
    wac_by_dest = read_industry_by_dest(geos, ind_codes, ind_keys)

    ## make it have the same index as work_dest in od_matrix
    wac_by_dest = pd.DataFrame(index=dest_index).join(wac_by_dest).astype(float)

    ## n people working outside the synth pop by industry is unknown
    ## estimate it as n_commuting_out * p_by_industry
    wac_by_dest.loc["outside",ind_keys] = n_commuting_out * p_by_ind

    ## save for next step
    print("writing commute-by-industry sums")
    od_matrix.to_csv(os.path.join("processed","work_od_prop.csv"),float_format="%.8g")
    ind_df.round(2).to_csv(os.path.join("processed","work_io_sums.csv"))
    wac_by_dest.round(2).to_csv(os.path.join("processed","work_id_est_sums.csv"))

    return None

## employer size data from:
## https://www.census.gov/programs-surveys/cbp/data/datasets.html
##
## code expects 2016 complete county file (more complete than 2019 data)
## filename: cbp16co.zip
##
## put into folder "work"

## https://www2.census.gov/programs-surveys/rhfs/cbp/technical%20documentation/2015_record_layouts/county_layout_2015.txt
#N1000           N       Number of Establishments: 1,000 or More Employee Size Class
#N1000_1         N       Number of Establishments: Employment Size Class:
#                                1,000-1,499 Employees
#N1000_2         N       Number of Establishments: Employment Size Class:
#                                1,500-2,499 Employees
#N1000_3         N       Number of Establishments: Employment Size Class:
#                                2,500-4,999 Employees
#N1000_4         N       Number of Establishments: Employment Size Class:
#                                5,000 or More Employees
def generate_work_sizes():
    print("generating employer sizes")

    work_counties = pd.read_csv(os.path.join('processed','work_sizes.csv'),dtype=str)["county"].values
    cbp = read_cbp(work_counties)
    ##
    ## attempt to infer size distribution
    ##
    ## last cat is "5000 or more", just guessing what the range is here:
    a = [1,5,10,20,50,100,250,500,1000,1500,2500,5000]
    b =   [5,10,20,50,100,250,500,1000,1500,2500,5000,30000]
    ## don't believe 0 cells?
    adj_z = 0.5
    sim_dist = [np.concatenate([np.random.randint(l,h,np.int64(s)) for (l,h,s) in zip(a,b, 1000*(adj_z+cbp.iloc[r,2:]))]) for r in range(cbp.shape[0])]

    mu_l = [np.mean(np.log(s)) for s in sim_dist]
    var_l = [np.var(np.log(s)) for s in sim_dist]
    mu_sz = [np.mean(s) for s in sim_dist]
    imp_v = [2.0*(np.log(a) - b) for (a,b) in zip(mu_sz, mu_l)]

    cbp['mu_ln'] = mu_l
    cbp['sigma_ln'] = np.sqrt(imp_v)

    cbp.to_csv(os.path.join('processed','work_sizes.csv'))
    return None


##
## schools
##

## using 2018-2019 data

## from https://nces.ed.gov/programs/edge/Geographic/SchoolLocations
## school locations: EDGE_GEOCODE_PUBLICSCH_*.xlsx (put into folder "school")
## GIS data: folder Shapefile_SCH (put inside folder "school")

## from https://nces.ed.gov/ccd/files.asp
## info about grades offered: "Directory" file ccd_sch_029*.csv or .zip
## enrollment data: "Membership" file ccd_sch_052*.csv or .zip
## number of teachers: "Staff" file ccd_sch_059*.csv or .zip
## put info folder "school"
## fields explained: https://nces.ed.gov/ccd/data/txt/sc132alay.txt

## TODO: add private schools https://nces.ed.gov/surveys/pss/index.asp
def generate_schools(geos):
    print("reading school data")

    schools = read_sch_data(geos)
    schools.to_csv(os.path.join('processed','schools.csv'))

    ## school geo data
    fname = glob(os.path.join("school","Shapefile_SCH","EDGE_GEOCODE*.shp"))[0]
    sch_lon_lat_pts = gpd.read_file(fname).set_index('NCESSCH')['geometry']
    schools = pd.merge(sch_lon_lat_pts, schools, left_index=True, right_index=True)
    ## location in projection UTM 18N coords
    schools['coords'] = schools['geometry'].to_crs(epsg=32618)

    ## cbg lat-long coords, for finding schools
    ## from https://www2.census.gov/geo/tiger/TIGER####/BG/ where #### is year
    ##
    ## need tl####_??_bg.zip where ?? is the FIPS code for each state in the synth area
    ##
    ## put in directory "geo"
    print("calculating school distances")

    cbg_geo = read_cbg_geo(geos)

    sch_coords = schools['coords']
    cbg_coords = cbg_geo['cent']
    cbg_sch_cross = pd.merge(cbg_coords,sch_coords,how='cross').set_index(pd.MultiIndex.from_product([cbg_coords.index,sch_coords.index]))
    cbg_sch_cross['X_diff'] = cbg_sch_cross['cent'].x - cbg_sch_cross['coords'].x
    cbg_sch_cross['Y_diff'] = cbg_sch_cross['cent'].y - cbg_sch_cross['coords'].y
    cbg_sch_cross['dist'] = np.round(np.linalg.norm(cbg_sch_cross[['X_diff', 'Y_diff']], axis=1), 2)
    cbg_sch_distmat = cbg_sch_cross['dist'].unstack()

    cbg_sch_distmat.to_csv(os.path.join('processed','cbg_sch_distmat.csv'))

    #print("finding closest schools by grade")
    ## -- julia does this faster
    ## 5 closest schools for each grade
    #for i in ['KG', *range(1,13)]:
    #    print(i, end=" ", flush=True)
    #    mask = schools['_'.join(['G',str(i),'OFFERED'])]
    #    df = cbg_sch_distmat.loc[:,mask].stack().groupby(level=0,group_keys=False).nsmallest(5).droplevel(0)
    #    df.to_csv(os.path.join('processed', ''.join(['cbg_sch_',str(i),'.csv']) ))

    ## assume kindergartens also offer preschool
    ## (we haven't generated private preschools and we have to send the preschoolers somewhere)
    #shutil.copy(os.path.join('processed','cbg_sch_KG.csv'), os.path.join('processed','cbg_sch_PK.csv'))

    return None


def main():

    ## industries and occupations to use in the synth pop
    ## each associated with codes from PUMS and column names from processed census data
    ind_codes = {'AGR_EXT':(["11", "21"],'Agriculture, forestry, fishing and hunting, and mining:'),
    'CON':(["23"],'Construction'),
    'MFG':(["31", "32", "33", "3M"],'Manufacturing'),
    'WHL':(["42"],'Wholesale trade'),
    'RET':(["44", "45", "4M"],'Retail trade'),
    'TRN_UTL':(["48", "49", "22"],'Transportation and warehousing, and utilities:'),
    'INF':(["51"],'Information'),
    'FIN':(["52", "53"],'Finance and insurance, and real estate, and rental and leasing:'),
    'PRF':(["54", "55", "56"],'Professional, scientific, and management, and administrative, and waste management services:'),
    'EDU':(["61"],'Educational services, and health care and social assistance:Educational services'),
    'MED':(["62"],'Educational services, and health care and social assistance:Health care and social assistance'),
    'ENT_art':(["71"],'Arts, entertainment, and recreation, and accommodation and food services:Arts, entertainment, and recreation'),
    'ENT_food':(["72"],'Arts, entertainment, and recreation, and accommodation and food services:Accommodation and food services'),
    'SRV':(["81"],'Other services, except public administration'),
    'ADM_MIL':(["92"],'ADM_MIL')}

    occ_codes = {'MBSA':(["11","13","15","17","19","21","23","27"],'MBSA'),
    'EDU':(["25"],'EDU'),
    'MED':(["29"],'MED'),
    'HLS':(["31"],'Service occupations:Healthcare support occupations'),
    'PRT':(["33"],'Service occupations:Protective service occupations:'),
    'EAT':(["35"],'Service occupations:Food preparation and serving related occupations'),
    'CLN':(["37"],'Service occupations:Building and grounds cleaning and maintenance occupations'),
    'PRS':(["39"],'Service occupations:Personal care and service occupations'),
    'SAL':(["41"],'Sales and office occupations:Sales and related occupations'),
    'OFF':(["43"],'Sales and office occupations:Office and administrative support occupations'),
    'FFF_RPR':(["45","47","49"],'Natural resources, construction, and maintenance occupations:'),
    'PRD_TRN':(["51","53"],'Production, transportation, and material moving occupations:'),
    'MIL':(["55"],'MIL')}

    ## default income categories
    inc_cats_def = ['q1_1', 'q1_2', 'q1_3', 'q2', 'q3', 'q4', 'q5']
    inc_cols_def = [['Less than $10,000'],
                    ['$10,000 to $14,999', '$15,000 to $19,999', '$20,000 to $24,999'],
                    ['$25,000 to $29,999', '$30,000 to $34,999', '$35,000 to $39,999'],
                    ['$40,000 to $44,999', '$45,000 to $49,999', '$50,000 to $59,999', '$60,000 to $74,999'],
                    ['$75,000 to $99,999', '$100,000 to $124,999'],
                    ['$125,000 to $149,999', '$150,000 to $199,999'],
                    ['$200,000 or more']]

    ## parameters from json files
    dgeo = tryJSON("geo.json")
    d = tryJSON("config.json")

    ## read states/counties to include
    geos = dgeo.get("geos", d.get("geos",None))
    commute_states = dgeo.get("commute_states", d.get("commute_states",None))

    ## read ADJINC
    ADJINC = d.get("inc_adj",1.010145)
    ## read income categories
    inc_cats = d.get("inc_cats",inc_cats_def)
    inc_cols = d.get("inc_cols",inc_cols_def)
    LODES_cutoff = d.get("LODES_annual_income_boundary",40000)

    ## additional individual (boolean) traits generated in read_psamp() to include in p_samples.csv summary
    more_summary_cols = d.get("additional_traits",
                              ['sch_public','sch_private','female','race_black_alone','white_non_hispanic','hispanic'])

    ## which census columns to match in synth pop
    ##  read from census or generated in generate_targets()
    target_columns = ['B11016:Family households:2-person household',
        'B11016:Family households:3-person household',
        'B11016:Family households:4-person household',
        'B11016:Family households:5-person household',
        'B11016:Family households:6-person household',
        'B11016:Family households:7-or-more person household',
        'B11016:Nonfamily households:1-person household',
        'B11016:Nonfamily households:2-person household',
        'B11016:Nonfamily households:3-person household',
        'B11016:Nonfamily households:4-person household',
        'B11016:Nonfamily households:5-person household',
        'B11016:Nonfamily households:6-person household',
        'B11016:Nonfamily households:7-or-more person household',
        ## covers distribution of workers per household reasonably well:
        'B23009:With own children of the householder under 18 years:Married-couple family:No workers',
        'B23009:With own children of the householder under 18 years:Married-couple family:1 worker',
        'B23009:With own children of the householder under 18 years:Married-couple family:2 workers:',
        'B23009:With own children of the householder under 18 years:Married-couple family:3 or more workers:',
        'B23009:With own children of the householder under 18 years:Other family:Unmarried householder:No workers',
        'B23009:With own children of the householder under 18 years:Other family:Unmarried householder:1 worker',
        'B23009:With own children of the householder under 18 years:Other family:Unmarried householder:2 workers',
        'B23009:With own children of the householder under 18 years:Other family:Unmarried householder:3 or more workers',
        'B23009:No own children of the householder under 18 years:Married-couple family:No workers',
        'B23009:No own children of the householder under 18 years:Married-couple family:1 worker',
        'B23009:No own children of the householder under 18 years:Married-couple family:2 workers:',
        'B23009:No own children of the householder under 18 years:Married-couple family:3 or more workers:',
        'B23009:No own children of the householder under 18 years:Other family:Unmarried householder:No workers',
        'B23009:No own children of the householder under 18 years:Other family:Unmarried householder:1 worker',
        'B23009:No own children of the householder under 18 years:Other family:Unmarried householder:2 workers',
        'B23009:No own children of the householder under 18 years:Other family:Unmarried householder:3 or more workers',
        # "related" covers almost all children in households
        'B11004:Married-couple family:With related children of the householder under 18 years:Under 6 years only',
        'B11004:Married-couple family:With related children of the householder under 18 years:Under 6 years and 6 to 17 years',
        'B11004:Married-couple family:With related children of the householder under 18 years:6 to 17 years only',
        'B11004:Married-couple family:No related children of the householder under 18 years',
        'B11004:Other family:Unmarried householder:With related children of the householder under 18 years:Under 6 years only',
        'B11004:Other family:Unmarried householder:With related children of the householder under 18 years:Under 6 years and 6 to 17 years',
        'B11004:Other family:Unmarried householder:With related children of the householder under 18 years:6 to 17 years only',
        'B11004:Other family:Unmarried householder:No related children of the householder under 18 years',
        ## covers unmarried partners and married-couple families:
        'B11012:Two-partner household:With own children under 18 years',
        'B11012:Two-partner household:With no own children under 18 years',
        ## various types of non-partner households:
        'B11012:Single householder:Living alone',
        'B11012:Single householder:With own children under 18 years',
        'B11012:Single householder:With relatives, no own children under 18 years',
        'B11012:Single householder:With only nonrelatives present',
        'B09018:', ## total children u18 in households, any relationship
        'B09018:Grandchild', ## under 18 (need this to sample multi-gen households)
        ## covers all adults in households:
        'B09021:18 to 34 years:Lives alone',
        'B09021:18 to 34 years:Householder living with partner or partner of householder',
        'B09021:18 to 34 years:Child of householder',
        'B09021:18 to 34 years:Other relatives',
        'B09021:18 to 34 years:Other nonrelatives',
        'B09021:35 to 64 years:Lives alone',
        'B09021:35 to 64 years:Householder living with partner or partner of householder',
        'B09021:35 to 64 years:Child of householder',
        'B09021:35 to 64 years:Other relatives',
        'B09021:35 to 64 years:Other nonrelatives',
        'B09021:65 years and over:Lives alone',
        'B09021:65 years and over:Householder living with partner or partner of householder',
        # sometimes the ACS has weirdly high estimates for this, seems unlikely
        #'B09021:65 years and over:Child of householder',
        'B09021:65 years and over:Other relatives',
        'B09021:65 years and over:Other nonrelatives',
            *['B19001:'+k for k in inc_cats],
        'B22010:Household received Food Stamps/SNAP in the past 12 months:',
        *['C24030:'+k for k in ind_codes.keys()],
        ## leave out occupations for now
        #*['C24010:'+k for k in occ_codes.keys()],
        "B25006:Householder who is Black or African American alone",
        "B11001H:Householder white non-hispanic",
        "B11001I:Householder hispanic any race"
        ]

    ## generated sample columns that match each of target_columns, in the same order:
    ## (generated in read_psamp and read_hsamp)
    sample_columns = ['fam_hh_2', 'fam_hh_3', 'fam_hh_4', 'fam_hh_5', 'fam_hh_6', 'fam_hh_7o', 
        'non_fam_hh_1', 'non_fam_hh_2', 'non_fam_hh_3', 'non_fam_hh_4', 'non_fam_hh_5', 'non_fam_hh_6', 'non_fam_hh_7o', 
        'w_own_ch_u18_married_fam_work0', 
        'w_own_ch_u18_married_fam_work1',
        'w_own_ch_u18_married_fam_work2',
        'w_own_ch_u18_married_fam_work3o',
        'w_own_ch_u18_unmar_fam_work0', 
        'w_own_ch_u18_unmar_fam_work1',
        'w_own_ch_u18_unmar_fam_work2',
        'w_own_ch_u18_unmar_fam_work3o',
        'no_own_ch_u18_married_fam_work0',
        'no_own_ch_u18_married_fam_work1',
        'no_own_ch_u18_married_fam_work2',
        'no_own_ch_u18_married_fam_work3o',
        'no_own_ch_u18_unmar_fam_work0',
        'no_own_ch_u18_unmar_fam_work1',
        'no_own_ch_u18_unmar_fam_work2',
        'no_own_ch_u18_unmar_fam_work3o', 
        'fam_married_w_rel_ch_u6_only',
        'fam_married_w_rel_ch_u6_and_6_17', 
        'fam_married_w_rel_ch_6_17_only',
        'fam_married_no_rel_ch_u18',
        'fam_unmar_w_rel_ch_u6_only', 
        'fam_unmar_w_rel_ch_u6_and_6_17', 
        'fam_unmar_w_rel_ch_6_17_only',
        'fam_unmar_no_rel_ch_u18',
        'partner_hh_ch_u18', 
        'partner_hh_no_ch_u18', 
        'hh_alone',
        'hh_single_ch_u18', 
        'hh_single_other_rel', 
        'hh_nonrel_only',
        'ch_u18_in_hh',
        'grandch_u18',
        'age_18_34_alone',
        'age_18_34_partner', 
        'age_18_34_child_of_hh',
        'age_18_34_other_rel', 
        'age_18_34_non_rel', 	   
        'age_35_64_alone', 
        'age_35_64_partner', 
        'age_35_64_child_of_hh',
        'age_35_64_other_rel', 
        'age_35_64_non_rel', 
        'age_65o_alone',
        'age_65o_partner', 
        # sometimes the ACS has weirdly high estimates for this, seems unlikely
        #'age_65o_child_of_hh', 
        'age_65o_other_rel',
        'age_65o_non_rel',
        *inc_cats,
        'snap',
        *['ind_'+k for k in ind_codes.keys()],
        ## leave out occupations for now
        #*['occ_'+k for k in occ_codes.keys()],
        'hh_race_black_alone',
        'hh_white_non_hispanic',
        'hh_hispanic'
        ]

    ## output directory
    os.makedirs("processed",exist_ok=True)
    
    ## save codes for workplace gen
    ind_keys = list(ind_codes.keys())
    with open(os.path.join('processed','codes.json'), 'w') as f:
        json.dump({'ind_codes':ind_keys, 'occ_codes':list(occ_codes.keys())}, f)

    p_summary = generate_samples(sample_columns, ADJINC, inc_cats, inc_cols, LODES_cutoff, ind_codes, occ_codes, more_summary_cols)
    adults_hh_cbg = read_acs('B09021',geos)[['B09021:']]
    cbg_index = adults_hh_cbg.index
    geo_xwalk = read_geo_xwalk(cbg_index)
    gq_stats = generate_gq(geos, adults_hh_cbg, geo_xwalk, p_summary, ind_codes, occ_codes)
    generate_targets(target_columns, geos, geo_xwalk, gq_stats, inc_cats, inc_cols, ind_codes, occ_codes)
    calc_commute_marginals(geos, ind_codes, ind_keys, commute_states)
    generate_work_sizes()
    generate_schools(geos)

    print("")
    print("done")







def generate_test_targets(geos,geo_xwalk):

    ## B09002 own ch u 18 by age and family type
    B09002 = read_acs('B09002',geos)
    ## B19123 Families; by size and public assistance/snap
    B19123 = read_acs('B19123',geos)
    ## B22010 households; person with disability
    B22010 = read_acs('B22010',geos)
    ## B23008 Own children under 18 years in families and subfamilies
    B23008 = read_acs('B23008',geos)
    ## households internet access
    B28002 = read_acs('B28002',geos)
    ## B28006 Household population 25 years and over; educational achievement
    B28006 = read_acs('B28006',geos)

    for x in ["Under 6 years","6 to 17 years"]:
        B23008[":".join(["B23008",x,"Living with two parents:One parent in labor force"])] = \
            B23008[":".join(["B23008",x,"Living with two parents:Father only in labor force"])] + \
            B23008[":".join(["B23008",x,"Living with two parents:Mother only in labor force"])]
        for y in ["In labor force","Not in labor force"]:
            B23008[":".join(["B23008",x,"Living with one parent",y])] = \
                B23008[":".join(["B23008",x,"Living with one parent:Living with father",y])] + \
                B23008[":".join(["B23008",x,"Living with one parent:Living with mother",y])]

    for x in ["Under 3 years","3 and 4 years","5 years","6 to 11 years","12 to 17 years"]:
        B09002[":".join(["B09002:In other families:Single householder, no spouse present",x])] = \
            B09002[":".join(["B09002:In other families:Male householder, no spouse present",x])] + \
            B09002[":".join(["B09002:In other families:Female householder, no spouse present",x])]
        
    for x in ["In married-couple families","In other families:Single householder, no spouse present"]:
        B09002[":".join(["B09002",x,"3 to 5 years"])] = \
            B09002[":".join(["B09002",x,"3 and 4 years"])] + \
            B09002[":".join(["B09002",x,"5 years"])]

    for x in ["Households with 1 or more persons with a disability","Households with no persons with a disability"]:
        B22010[":".join(["B22010",x])] = \
            B22010[":".join(["B22010:Household received Food Stamps/SNAP in the past 12 months",x])] + \
            B22010[":".join(["B22010:Household did not receive Food Stamps/SNAP in the past 12 months",x])]
        
    ## join all census tables together
    acs_tables = B09002.join([B19123,B22010,B23008,B28002,B28006])
    acs_tables['state'] = acs_tables.index.map(lambda x: x[0:2])
    acs_tables['county'] = acs_tables.index.map(lambda x: x[0:5])

    ## which census columns to match:
    target_columns = [
        'B09002:In married-couple families:Under 3 years',
        'B09002:In married-couple families:3 to 5 years',
        'B09002:In married-couple families:6 to 11 years',
        'B09002:In married-couple families:12 to 17 years',
        'B09002:In other families:Single householder, no spouse present:Under 3 years',
        'B09002:In other families:Single householder, no spouse present:3 to 5 years',
        'B09002:In other families:Single householder, no spouse present:6 to 11 years',
        'B09002:In other families:Single householder, no spouse present:12 to 17 years',
        'B19123:2-person families:With cash public assistance income or  households receiving Food Stamps/SNAP benefits in the past 12 months',
        'B19123:3-person families:With cash public assistance income or  households receiving Food Stamps/SNAP benefits in the past 12 months',
        'B19123:4-person families:With cash public assistance income or  households receiving Food Stamps/SNAP benefits in the past 12 months',
        'B19123:5-person families:With cash public assistance income or  households receiving Food Stamps/SNAP benefits in the past 12 months',
        'B19123:6-person families:With cash public assistance income or  households receiving Food Stamps/SNAP benefits in the past 12 months',
        'B19123:7-or-more-person families:With cash public assistance income or  households receiving Food Stamps/SNAP benefits in the past 12 months',
        'B22010:Households with 1 or more persons with a disability',
        'B23008:Under 6 years:Living with two parents:Both parents in labor force',
        'B23008:Under 6 years:Living with two parents:One parent in labor force',    
        'B23008:Under 6 years:Living with two parents:Neither parent in labor force',
        'B23008:Under 6 years:Living with one parent:In labor force',
        'B23008:Under 6 years:Living with one parent:Not in labor force',
        'B23008:6 to 17 years:Living with two parents:Both parents in labor force',
        'B23008:6 to 17 years:Living with two parents:One parent in labor force',    
        'B23008:6 to 17 years:Living with two parents:Neither parent in labor force',
        'B23008:6 to 17 years:Living with one parent:In labor force',
        'B23008:6 to 17 years:Living with one parent:Not in labor force',
        "B28002:With an Internet subscription",
        "B28002:Internet access without a subscription",
        "B28002:No Internet access",
        "B28006:Less than high school graduate or equivalency:",
        "B28006:High school graduate (includes equivalency) , some college or associate's degree :",
        "B28006:Bachelor's degree or higher:"]

    missing_puma = geo_xwalk.index[geo_xwalk['st_puma'].isna()]
    acs_tables = acs_tables[~acs_tables.index.isin(missing_puma)].copy(deep=True)

    ## drop cbgs with less than 20 hh
    acs20 = acs_tables[acs_tables['B22010:'] > 19]
    acs20[target_columns].to_csv(os.path.join('processed','test_targets.csv'))

    return None


def gen_samp_test_cols(ADJINC, inc_cats, inc_cols, LODES_cutoff):

    psamp, ptotals = read_psamp(LODES_cutoff, [], [])
    hsamp = read_hsamp(ptotals, ADJINC, inc_cats, inc_cols)
    ## generated sample columns that match each of target_columns, in the same order:
    sample_columns = ['own_ch_u3_in_fam_married',
        'own_ch_3_5_in_fam_married',
        'own_ch_6_11_in_fam_married',
        'own_ch_12_17_in_fam_married',
        'own_ch_u3_in_fam_unmar',
        'own_ch_3_5_in_fam_unmar',
        'own_ch_6_11_in_fam_unmar',
        'own_ch_12_17_in_fam_unmar',
        'fam_hh_2_snap_pap',
        'fam_hh_3_snap_pap',
        'fam_hh_4_snap_pap',
        'fam_hh_5_snap_pap',
        'fam_hh_6_snap_pap',
        'fam_hh_7o_snap_pap',
        'h_1_or_more_with_disab',
        'esp_2p_2w_age_u6',
        'esp_2p_1w_age_u6',
        'esp_2p_nw_age_u6',
        'esp_1p_1w_age_u6',
        'esp_1p_nw_age_u6',
        'esp_2p_2w_age_6_17',
        'esp_2p_1w_age_6_17',
        'esp_2p_nw_age_6_17',
        'esp_1p_1w_age_6_17',
        'esp_1p_nw_age_6_17',
        'h_internet_sub',
        'h_internet_nosub',
        'h_no_internet',
        'edu_not_hsgrad_age_25o',
        'edu_hs_or_somecoll_age_25o',
        'edu_bach_or_higher_age_25o']

    ## csv of household samples
    hsamp[sample_columns].to_csv(os.path.join('processed','test_samples.csv'))

    return None


def test_cols():
    ## default income categories
    inc_cats_def = ['q1_1', 'q1_2', 'q1_3', 'q2', 'q3', 'q4', 'q5']
    inc_cols_def = [['Less than $10,000'],
                    ['$10,000 to $14,999', '$15,000 to $19,999', '$20,000 to $24,999'],
                    ['$25,000 to $29,999', '$30,000 to $34,999', '$35,000 to $39,999'],
                    ['$40,000 to $44,999', '$45,000 to $49,999', '$50,000 to $59,999', '$60,000 to $74,999'],
                    ['$75,000 to $99,999', '$100,000 to $124,999'],
                    ['$125,000 to $149,999', '$150,000 to $199,999'],
                    ['$200,000 or more']]

    ## parameters from json files
    dgeo = tryJSON("geo.json")
    d = tryJSON("config.json")

    ## read states/counties to include
    geos = dgeo.get("geos", d.get("geos",None))

    ## read ADJINC
    ADJINC = d.get("inc_adj",1.010145)
    ## read income categories
    inc_cats = d.get("inc_cats",inc_cats_def)
    inc_cols = d.get("inc_cols",inc_cols_def)
    LODES_cutoff = d.get("LODES_annual_income_boundary",40000)
    adults_hh_cbg = read_acs('B09021',geos)[['B09021:']]
    cbg_index = adults_hh_cbg.index
    geo_xwalk = read_geo_xwalk(cbg_index)

    generate_test_targets(geos,geo_xwalk)
    gen_samp_test_cols(ADJINC, inc_cats, inc_cols, LODES_cutoff)










if __name__ == "__main__":
    main()

