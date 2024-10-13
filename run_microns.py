import numpy as np

from microns_activity_search_from_activity import *
from microns_across_scans import *
from microns_parameter_search import *

if __name__ == "__main__":
    R_max_lst = ["1.000000e-02", "5.000000e-02", "1.000000e-01", "2.000000e-01", "5.000000e-01", "7.000000e-01", "1", "1.250000e+00", "1.500000e+00"]
    D = 2

    for R_max in R_max_lst:
        all_run(R_max, D, False)
        microns_across_scans(R_max, D)
    
    microns_parameter_search(D)