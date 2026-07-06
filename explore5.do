capture log close
log using "E:\TFP_weather\output5.txt", replace text

* Load climate panel into Stata to check county_type
import delimited "E:\TFP_weather\data\climate_panel_1982_2016.csv", clear varnames(1)

* Check county_type values
tab county_type

* Check zone37_main values
tab zone37_main

* How many unique counties
codebook pac, compact

* Check a sample
list pac year name province county_type zone37_main in 1/5

log close
