# This is a sample Python script.
import numpy as np
import os
from DFF_OASIS import traces, correlations
import glob
#import build_data_from_suite2p.py


def apply_oasis(condition_folder):
    print(condition_folder)

    data_sets = [os.path.basename(x) for x in glob.glob(condition_folder +"/*_im_*")]
    print(len(data_sets))
    for d in data_sets:
            print("processing .... " + d )
            t = traces(condition_folder=condition_folder + "/", dataset_name=d)
            np.save(file=condition_folder + d + "/" + d + "_DFF.npy", X=t.DFF)
            print("saving ....")
            np.save(file=condition_folder + d + "/preprocessed/" + d + "_oasis_s.npy", X = t.s)
            np.save(file=condition_folder + d + "/preprocessed/" + d + "_oasis_c.npy", X=t.c)
            np.save(file=condition_folder + d + "/preprocessed/" + d + "_oasisAR1_s.npy", X = t.s_AR1)
            np.save(file=condition_folder + d + "/preprocessed/" + d + "_oasisAR1_c.npy", X=t.c_AR1)
            np.save(file=condition_folder + d + "/preprocessed/" + d + "_cell_centers.npy", X=t.C)


if __name__ == '__main__':
    main_folder = "/media/thomas_sainsbury/Samsung_T51/Decoding_Multiple_stim/"
    decon = True



    if decon == True:
        apply_oasis(condition_folder = main_folder + "DC_WT_DR_7_dpf")






