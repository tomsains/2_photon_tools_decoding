# This is a sample Python script.
import numpy as np
import os
from DFF_OASIS import *
import glob
from Build_data_for_decoder import *
from decoder_class import *


if __name__ == '__main__':
    main_folder = "/media/thomas_sainsbury/Samsung_T5/Decoding_Multiple_stim/"
    decon = True
    build_decoder_data_set = True
    decode_stim = True

    if decon == True:
        apply_oasis(condition_folder = main_folder + "DC_WT_GR_7_dpf")
        apply_oasis(condition_folder=main_folder + "DC_WT_NR_7_dpf")
        apply_oasis(condition_folder=main_folder + "DC_WT_DR_7_dpf")

    if build_decoder_data_set == True:
        apply_build_data_for_decoder(condition_folder=main_folder + "DC_WT_GR_7_dpf")
        apply_build_data_for_decoder(condition_folder=main_folder + "DC_WT_NR_7_dpf")

    if decode_stim == True:
        apply_decoders(condition_folder=main_folder + "DC_WT_GR_7_dpf/")
        apply_decoders(condition_folder=main_folder + "DC_WT_NR_7_dpf")







