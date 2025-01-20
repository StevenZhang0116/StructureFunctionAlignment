import numpy as np

from microns_activity_search_from_activity import *
from microns_across_scans import *
from microns_parameter_search import *

if __name__ == "__main__":
    # R_max_lst = ["1", "8.000000e-01", "1.200000e+00", "1.400000e+00", "1.600000e+00", "1.800000e+00"]
    R_max_lst = ["1"]
    D = 2
    # R_max_lst = ["6.000000e-01"]
    # D = 3
    ww = "normal"
    cc = "count"
    ss = "all"
    downsample_from_connectome = 0
    # for connectome data, coming from scan_specific is True
    # for activity data, coming from scan_specific is False
    scan_specific = False
    
    pendindex = f"noise_{ww}_cc_{cc}_ss_{ss}" + ("_forall" if downsample_from_connectome else "")

    for R_max in R_max_lst:
        all_run(R_max, D, False, ww, cc, ss, scan_specific, downsample_from_connectome)
        microns_across_scans(R_max, D, 5, pendindex, scan_specific)
    
    # microns_parameter_search(D, 0, ww, cc)
 