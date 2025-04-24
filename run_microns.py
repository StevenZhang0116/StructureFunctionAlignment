import numpy as np

from microns_activity_search_from_activity import *
from microns_across_scans import *
from microns_parameter_search import *

if __name__ == "__main__":
    for [ww,cc] in [["normal", "psd"]]:
        for D_ in [2]:
            R_max_lst = ["1"]
            # R_max_lst = ["1"]
            D = D_
            # ww = "normal"
            # cc = "binary"
            ss = "all"
            downsample_from_connectome = False
            # for connectome data, coming from scan_specific is True
            # for activity data, coming from scan_specific is False
            scan_specific = False
            
            pendindex = f"noise_{ww}_cc_{cc}_ss_{ss}_dirpsd" + ("_forall" if downsample_from_connectome else "")

            for R_max in R_max_lst:
                # all_run(R_max, D, False, ww, cc, ss, scan_specific, downsample_from_connectome, pendindex)
                microns_across_scans(R_max, D, 5, pendindex, scan_specific)
            
            # microns_parameter_search(D, 0, ww, cc)