# Schools Pupils and Characteristics (SPC)
Python scripts and datafiles for longitudinal analysis of English pupil data.

The "Schools Pupils & Characteristics" (SPC) is a dataset released by DfE each year based on January census. 
Latest copies of the data can be obtained from:
https://explore-education-statistics.service.gov.uk/find-statistics/school-pupils-and-their-characteristics
It takes a bit of effort to find where they squirrel away past years' data

Each release consists ~24k rows, 1 per school with ~260 attributes. Data consistency across the years is quite good, although DfE keep changing the column names!
SPC datasets date back to 2010, however the DfE made some major changes in 2014 so this script only goes that far back. This repository will contain a script for
made up of four main components
  Function: parse_spc - reads and cleans the data
  Function: pivot_tab - collates the data either by Local Authority or Parliamentary Constituency.
  Main body: Format -  A section which reformats the data for plotting
  Main body: Plots - A section which plots the data (and writes some tables)

The repositor will consist of
  The script
  The raw CSV files
  Details of where the originals can be found. 
  The output (tables and graphs) 
