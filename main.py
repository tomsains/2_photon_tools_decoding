# This is a sample Python script.
import numpy as np
import os
from DFF_OASIS import *
import glob
from Build_data_for_decoder import *
from decoder_class import *
from Plot_decoder_results import *


if __name__ == '__main__':
    main_folder = "/media/thomas_sainsbury/Samsung_T5/Decoding_Multiple_stim/"
    decon = False
    build_decoder_data_set = False
    decode_stim = True
    plot_results = True


    if decon == True:
        apply_oasis(condition_folder = main_folder + "DC_WT_GR_7_dpf", start_from=3)
        apply_oasis(condition_folder=main_folder + "DC_WT_NR_7_dpf")
        apply_oasis(condition_folder=main_folder + "DC_WT_DR_7_dpf")

    if build_decoder_data_set == True:
        apply_build_data_for_decoder(condition_folder=main_folder + "DC_WT_GR_7_dpf")
        apply_build_data_for_decoder(condition_folder=main_folder + "DC_WT_NR_7_dpf")
        apply_build_data_for_decoder(condition_folder=main_folder + "DC_WT_DR_7_dpf")

    if decode_stim == True:
        apply_decoders(condition_folder=main_folder + "DC_WT_GR_7_dpf")
        apply_decoders(condition_folder=main_folder + "DC_WT_NR_7_dpf")
        apply_decoders(condition_folder=main_folder + "DC_WT_DR_7_dpf")


    if plot_results == True:
        combined_results(main_folder= main_folder)





