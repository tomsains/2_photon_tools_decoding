# This is a sample Python script.
import numpy as np
import os
from DFF_OASIS import *
import glob
from Build_data_for_decoder import *


if __name__ == '__main__':
    main_folder = "/media/thomas_sainsbury/Samsung_T51/Decoding_Multiple_stim/"
    decon = False
    build_decoder_data_set = True

    if decon == True:
        apply_oasis(condition_folder = main_folder + "DC_WT_DR_7_dpf")

    if build_decoder_data_set == True:
        data_set(folder=main_folder + "DC_WT_DR_7_dpf/", dataset_name= "201016_H2BGC6s_DR_DC_F1_im_00002")








