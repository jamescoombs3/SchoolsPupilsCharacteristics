"""
  _____ _____   _____
 / ____|  __ \ / ____|
| (___ | |__) | |
 \___ \|  ___/| |
 ____) | |    | |____
|_____/|_|     \_____|

"Schools Pupils & Characteristics" (SPC) is a dataset released by DfE based on January census. Latest is available:
https://explore-education-statistics.service.gov.uk/find-statistics/school-pupils-and-their-characteristics
The exact location of these datasets changes so instructions will be added on where to find the data.
The source CSV datafiles will be uploaded to
https://github.com/jamescoombs3/XXXXXXX (in due course)

Consistency of the data across the years is quite good, although column names change frequently needing lots of
coding to cater for. The SPC datasets date back to 2010 but the DfE made some major changes in 2014 so this
(currently) only goes that far back. The approach used is was to create a csv file containing field mappings
so these are consistent within the script.

Script is made up of four main components
Function: parse_spc - reads and cleans the data
Function: pivot_tab - collates the data either by Local Authority or Parliamentary Constituency.
Main body: Format -  A section which reformats the data for plotting
Main body: Plots - A section which plots the data (and writes some tables)

"""

# import glob
# from pyproj import Transformer
# import xlrd
# import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

def pivot_tab(collate_by, spc):
    """
    :param collate_by:  String set to either 'LA_name' or 'Constituency'
    :param spc:         pd.DataFrame containing the pre-processed SPC dataset
    :return:            pd.DataFrame containing summarised data
    This function collates the data in SPC either by LA or constituency and creates a few
    derived values, eg the proportion of grammar places in an LA is grammar spaces divided by total spaces.

      _____ _      ______          _   _
     / ____| |    |  ____|   /\   | \ | |
    | |    | |    | |__     /  \  |  \| |
    | |    | |    |  __|   / /\ \ | . ` |
    | |____| |____| |____ / ____ \| |\  |
     \_____|______|______/_/    \_\_| \_|
    This section filters and cleans the SPC data so it is consistent and usable.

    """

    # Create pivot table to work out numbers of pupils at all schools by FSM/non-FSM for each constituency
    df = pd.pivot_table(spc, values=['roll', 'numFSM'], index=collate_by, columns=['AdmPolicy'], aggfunc=np.sum)
    df.columns = ['Non_Sel_FSM', 'Sel_FSM', 'Non_Sel_Roll', 'Sel_Roll']

    # Pandas pivot creates floats and NANs. Convert to integers and zeros.
    for c in df.columns:
        df[c] = df[c].fillna(0.0).astype(int)
        df[c] = df[c].astype(int)

    # work out proportion of selective places for each LA/PC
    df['pct_Sel'] = df['Sel_Roll'] / (df['Sel_Roll'] + df['Non_Sel_Roll']) * 100

    # sort with most 'highly selective' LAs/PCs at the top
    df = df.sort_values(by=['pct_Sel'], ascending=False)

    # work out proportion of FSM in grammars for each LA/PC
    df['pct_FSM_Sel'] = df['Sel_FSM'] / df['Sel_Roll'] * 100

    # work out proportion of FSM in 'local' for each LA/PC
    df['PCT_FSM_area'] = (df['Non_Sel_FSM'] + df['Sel_FSM']) / \
                               (df['Non_Sel_Roll'] + df['Sel_Roll']) * 100

    # calculate 'inclusivity' as %FSM_grammar / %FSM_local for each LA/PC
    df['inclusivity'] = df['PCT_FSM_area'] / df['pct_FSM_Sel']
    return df

def parse_spc(csv, map, year):
    """
    :param csv: String - full path to the CSV data file as created by DfE
    :param map: Dictionary - maps DfE field names which vary to consistent column names
    :param year: integer - *MIGHT BE* used to determine whether it's possible to process constituency
    (Currently only going back as far as 2014 when DfE made major change to the SPC
    :return: df1, df2 containing LA and PC collations for the relative year.

     _____ _______      ______ _______
    |  __ \_   _\ \    / / __ \__   __|
    | |__) || |  \ \  / / |  | | | |
    |  ___/ | |   \ \/ /| |  | | | |
    | |    _| |_   \  / | |__| | | |
    |_|   |_____|   \/   \____/  |_|

    This function mainly aggregates the SPC data into pivot tables. Values such as percentages, based on
    absolute values from SPC are also calculated here.
    """

    # read just the wanted columns into dataframe
    df = pd.read_csv(csv, usecols=list(map.keys()), encoding='unicode_escape', engine='python')
    # rename the columns
    df = df.rename(columns=map)

    # Until 2013 DfE encoded secondary schools as 'Secondary' in the PhaseGroup field.
    # In 2014, they added 2,411 Independent schools and changed the encoded value for
    # secondary schools to the more precise 'State-funded secondary'

    # Pandas inbuilt .replace only matches the full cell so safe to use this to reset to < 2014 encoding
    df = df.replace({'Secondary': 'State-funded secondary'})

    # select just secondary schools (row count drops from ~24k to ~3k. Lots of primary schools!)
    df = df[df['PhaseGroup'] == 'State-funded secondary']

    # force numeric values to integers
    for col in df.columns[-12:]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.int64)

    # add KS3 + KS4 to get total secondary school pupils on roll
    df['roll'] = df['girls_11'] + df['girls_12'] + df['girls_13'] + df['girls_14'] + df['girls_15'] + \
                  df['boys_11'] + df['boys_12'] + df['boys_13'] + df['boys_14'] + df['boys_15']
    # WARNING! Will fail here if the number of columns selected from spc CSV files changes!
    # Modify the slice in "for c in spc.columns ..." two lines up.

    # Make all 'Selective' values 'Selective'
    df = df.replace({'Selective (grammar)': 'Selective'})
    # Use pd.apply in lambda function to set this value to Sel or Comp
    df['AdmPolicy'] = df['AdmPolicy'].apply(lambda x: 'Sel' if x == 'Selective' else 'Comp')

    """
    Adjust spc.numFSM because this potentially includes pupils outside KS3/KS4.
    .roll is number in KS3/KS4 derived by summing individual year groups. 
    .numFSMcalcs is number used to calculate FSM for whole school, which can include primary or sixth form children
    .numFSM needs adjusting to adjust for non KS3/4 children
    """

    df.numFSM = round(df.numFSM * (df.roll / df.numFSMcalcs))
    # some 'secondary' schools have 0 KS3/4 children on roll. These don't matter as we're just summing number
    # of children but it does result in NaN values needs fixing
    df['numFSM'] = df['numFSM'].fillna(0.0).astype(int)
    # numFSM is stored as a float even though all fractional values are zero due to round() above
    df['numFSM'] = df['numFSM'].astype(int)

    """
    Develop this in small chunks. First ensure we have a callable function which can return the spc dataframe.
    When that is working, modify this function to call pivot_tab to produce the final data frame(s) needed
    by the main program. ... then write the main program.  
    """
    return df

"""

 _____  ______ _____  _______      __  _______ _____ ____  _   _ 
|  __ \|  ____|  __ \|_   _\ \    / /\|__   __|_   _/ __ \| \ | |
| |  | | |__  | |__) | | |  \ \  / /  \  | |    | || |  | |  \| |
| |  | |  __| |  _  /  | |   \ \/ / /\ \ | |    | || |  | | . ` |
| |__| | |____| | \ \ _| |_   \  / ____ \| |   _| || |__| | |\  |
|_____/|______|_|  \_\_____|   \/_/    \_\_|  |_____\____/|_| \_|
                                                                 
This section iterates over the years calling the functions to return dataframes of nicely cleaned data from 
the SPC CSV files. After looking into three dimensional data structures in Pandas, the decision was made to 
just add a year column to the dataframes returned by pivot_tab(). This section also does some reformatting
to make plotting easier.                                                                   

"""

# test parse_spc function
# define global values relating to input data.
workdir = 'C:/_python/data/SPC/clean'
fields_map = workdir + '/SPC_field_mappings.csv'
fm = pd.read_csv(fields_map)

# per iteration/year
# datafile = workdir + '/SPC_2018.csv'
# maps = dict(zip(fm.y2018, fm.Python_name))
# spc = parse_spc(datafile, maps, 2018)

# First two columns of fields map contain the consistent name to use in python and a description.
# All remaining columns contain the names used by DfE for each year (which we want to iterate over)
years = fm.columns[2:]
# iterate "for year in years"

# Create empty dataframes LA and PC. NOTE additional 'year' field
cols = ['year', 'Non_Sel_FSM', 'Sel_FSM', 'Non_Sel_Roll', 'Sel_Roll', 'pct_Sel',
        'pct_FSM_Sel', 'PCT_FSM_area', 'inclusivity']
LA = pd.DataFrame(columns=cols)
PC = pd.DataFrame(columns=cols)

for year in years:
    datafile = workdir + '/SPC_' + year + '.csv'
    maps = dict(zip(fm[year], fm['Python_name']))
    print('Loop variables year, datafile and maps are', year, datafile, maps)

    # extract and clean SPC data
    spc = parse_spc(datafile, maps, year)

    # Call pivot_tab twice (once to collate by LA the other by constituency) and
    la = pivot_tab('LA_name', spc)
    pc = pivot_tab('Constituency', spc)

    # add a column with the current year
    la['year'] = year
    pc['year'] = year

    # append the current year to the overall dataframe
    LA = LA.append(la)
    PC = PC.append(la)

    print('End of', year,  'loop. DataFrame shapes are', LA.shape, PC.shape)

# run manually to set parameters to debug functions
# collate_by = 'Constituency'
# csv = datafile
# map = maps

# Export interim data for manual validation and checks
# LA.to_csv('c:/tmp/la.csv')
# PC.to_csv('c:/tmp/pc.csv')

"""
Basis for defining "fully selective LEAs"
========================================= 
The DfE defines nine LEAs, listed in Schedule 1 of the Education (Grammar School Ballots) Regulations 1998, 
as selective areas. (Bexley, Buckinghamshire, Kent, Lincolnshire, Medway, Slough, Southend, Torbay and Trafford). 
Since then, grammar schools in Wirral and Sutton and now account for over 25% of the places. These two LEAs
should be added to this list of "selective areas" making a total 11 selective authorities. 
"""

Sel_LAs = ['Buckinghamshire', 'Trafford', 'Kent', 'Medway', 'Southend-on-Sea', 'Torbay',
           'Wirral', 'Sutton', 'Slough', 'Lincolnshire', 'Bexley']

# Create a dataframe with row per Sel_LA and columns per year 2013/14 to 2021/22 containing the pct_Sel value from LA
# We could probably plot direct from LA dataframe but doing it this way helps retain my sanity!
# These next couple of lines took some figuring out!
LAx = LA.loc[Sel_LAs, ['year', 'pct_Sel']]
LAx = LAx.pivot(columns='year', values='pct_Sel')
LAx.sort_values(by='2022', ascending=False, inplace=True)

pop_cols = ['school_type', 'pupil_age', '2014_actual', '2015_actual', '2016_actual', '2017_actual',
            '2018_actual', '2019_actual', '2020_actual', '2021_actual', '2022_actual']

# Open national population figures
# https://explore-education-statistics.service.gov.uk/find-statistics/national-pupil-projections
# pop = pd.read_csv(workdir + '/pupilprojprincipal_2022.csv', usecols=pop_cols, sep=',', thousands=',')
# The above doesn't work! Munged around the problem below.
pop = pd.read_csv(workdir + '/pupilprojprincipal_2022.csv', usecols=pop_cols)
# Filter out just school_type == 'All schools'
pop = pop[pop.school_type == 'All schools']
# Filter out just pupil_age == '11 to 15'
pop = pop[pop.pupil_age == '11 to 15']

pop.reset_index(inplace=True)
pop.drop(columns=['index', 'school_type', 'pupil_age'], inplace=True)
for c in pop.columns:
    pop[c] = pop[c].str.replace(',', '')
    pop[c] = pd.to_numeric(pop[c], errors='coerce').fillna(0).astype(np.int64)

LAxx = LA.loc[Sel_LAs, ['year', 'Sel_Roll']]
LAxx = pd.pivot_table(LAxx, columns='year', aggfunc=np.sum)
# At this point there are two dataframes:
#    pop contains the population of 11 - 15 year olds in all schools
#    LAxx contains the number of 11 - 15 year olds in grammars in fully selective areas
# These two dataframes have columns 2014 - 2022 inclusive but actual column names are different
# internally "2014" relates to Jan 2014 census but to be clearer what this means to others, remap cols to:
year_names = ['2013/14', '2014/15', '2015/16', '2016/17', '2017/18', '2018/19', '2019/20', '2020/21', '2021/22']
pop.columns = year_names
LAxx.columns = year_names

pop = pop.append(LAxx)
pop = pop.T
pop.columns = ['Population (11-15)', 'Grammar Places', '%increase']

# Work out ratio of ALL population compared to grammars in selective LEAs in 2014 (roughly 40)
ratio = pop.at['2013/14', 'Population in all schools'] / pop.at['2013/14', 'Population in grammar schools']
# Add another column to give % increase/decrease of grammars since 2014
pop['increase'] = (pop['Population in grammar schools'] / pop['Population in all schools'] * ratio - 1) * 100

# Finally, create a plot showing %FSM in grammars compared to FSM in selective authorities (Cribb et al 2009)
# as well as nationally (because Cribb may have underestimated the issue by comparing 'locally')
# Starting point is LA dataframe and the fields
# 'year', 'Non_Sel_FSM', 'Sel_FSM', 'Non_Sel_Roll', 'Sel_Roll',
# Extract data for selective and non-selective in a "Sel"ective LA dataframe
LEA = LA.loc[Sel_LAs, ['year', 'Non_Sel_FSM', 'Sel_FSM', 'Non_Sel_Roll', 'Sel_Roll']]
LEA = pd.pivot_table(LEA, columns='year', aggfunc=np.sum)
# Transpose LEA so we can work with Pandas columns
LEA = LEA.T
# %FSM at grammars in LEAective authorities is simply 'LEA_FSM' / 'LEA_Roll' * 100
# %FSM overall in LEAective authorities is 'Non_LEA_FSM' / ('Non_LEA_Roll' + 'LEA_Roll')  * 100
LEA['gram_FSM'] = LEA['Sel_FSM'] / LEA['Sel_Roll'] * 100
LEA['lea_FSM'] = (LEA['Non_Sel_FSM'] + LEA['Sel_FSM']) / (LEA['Non_Sel_Roll'] + LEA['Sel_Roll']) * 100
# LEA.to_csv('c:/tmp/LEA.csv')

All = pd.pivot_table(LA, values=['Non_Sel_FSM', 'Sel_FSM', 'Non_Sel_Roll', 'Sel_Roll'], columns='year', aggfunc=np.sum)
All = All.T
All['gram_FSM'] = All['Sel_FSM'] / All['Sel_Roll'] * 100
All['lea_FSM'] = (All['Non_Sel_FSM'] + All['Sel_FSM']) / (All['Non_Sel_Roll'] + All['Sel_Roll']) * 100

LEA['national'] = All['lea_FSM']
LEA = LEA.T
# plot LEA ['gram_FSM', 'lea_FSM', 'national'] × years

"""
 _____  _      ____ _______ _____ 
|  __ \| |    / __ \__   __/ ____|
| |__) | |   | |  | | | | | (___  
|  ___/| |   | |  | | | |  \___ \ 
| |    | |___| |__| | | |  ____) |
|_|    |______\____/  |_| |_____/ 

This section will save the data to CSV files. (Might contain more rows than needed.) The main purpose of this
section is to plot graphs. At this point there are three dataframes as follows
LAx: 
    11 fully selective authorities × 9 years containing %grammarplaces in schools in each authority 
LEA:
    9 Rows, although only want 3 containing ['gram_FSM', 'lea_FSM', 'national'] which are precentages × 9 years
pop:
    9 rows containing years × [population of all children, population of grammars (in selective areas)] 
    
Confession! This was written last with time pressure. The code would benefit from some tidying up to ensure things
like having meaningful variable names and removing any experimental code which is just commented out. 
It's a bit messy! 
"""

# Print out the three tables used.
LAx.to_csv(workdir + '/table1.csv')
pop.to_csv(workdir + '/table2.csv')
LEA[-3:].to_csv(workdir + '/table3.csv')  # only want last three rows.

# plot_pop = pop[['Population in all schools', 'Population in grammar schools']]
ppp = pop.reset_index()
matplotlib.rc_file_defaults()
matplotlib.rcParams['figure.dpi'] = 200
popcolour='blue'
gramcolour='red'
ax1 = sns.set_style(style=None, rc=None)
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_ylabel('Population 11-15 (millions)', color=popcolour)
ax1.set_xlabel('Year')
sns.lineplot(data=ppp, y='Population (11-15)', x='index', sort=False, color=popcolour, ax=ax1)
ax2 = ax1.twinx()
ax2.set_ylabel('Grammar places', color=gramcolour)
sns.lineplot(data=ppp, y='Grammar Places', x='index', sort=False, color=gramcolour, ax=ax2)

ax1.tick_params(axis='y', labelcolor=popcolour)
ax2.tick_params(axis='y', labelcolor=gramcolour)

# The relative growth in grammar schools (14.91%) is ~11% more than non-grammars (13.43%).
# Work out the y-axis limits for grammars then scale the population to 0.9 of this.
ax2.set(ylim=(73000, 85600))
ax1.set(ylim=(2940000, 3460000))

fig.savefig(workdir + '/figure2.png')

# fig.legend(labels=['Population (millions)', 'Grammar places'], loc='upper left')
# fig.legend(loc='upper left')
# fig.update_layout(legend=dict(yanchor='top', y=0.99, xanchor='left', x=1))

fsm = LEA.iloc[-3:].T
fsm.columns = ['Grammars', 'Selective LEAs', 'National']

fig2, ax3 = plt.subplots(figsize=(12, 6))
# sns.lineplot(data=fsm, linestyle='-')  # doesn't work!
sns.lineplot(data=fsm)
ax3.set_ylabel('Percent Free School Meals')
ticks = [0, 5, 10, 15, 20, 25]
ax3.set_yticks(ticks)
ax3.set_xticks(fsm.reset_index().year)

fig2.savefig(workdir + '/figure3.png')
# ax1.set_ylabel('Percent Free School Meals')


LAx = LAx.T
# Rename Buckinghamshire -> Bucks, Southen-on-Sea -> Lincolnshire -> Lincs
LAx.columns = ['Bucks', 'Trafford', 'Kent', 'Medway', 'Southend', 'Torbay',
               'Wirral', 'Sutton', 'Slough', 'Lincs', 'Bexley']

fig4, ax4 = plt.subplots(figsize=(12, 6))
# sns.heatmap(data=LAx)  # heatmap didn't tell the story well.
sns.lineplot(data=LAx)
ax4.set_ylabel('Percent children per Local Authority in grammar schools')

fig4.savefig(workdir + '/figure1.png')

"""
# OSSIA - Matplotlib
f2 = fsm.reset_index()
fig4, ax4 = plt.subplots()
ax4.plot(f2.year, f2['Grammars'])
ax4.plot(f2.year, f2['Selective LEAs'])
ax4.plot(f2.year, f2['National'])
plt.show()
"""

