import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os


class combined_results():
    def __init__(self, main_folder, data_type = "Cal", deconvolution_method = "estimated", method = "max", decoder = "logreg"):
        self.main_folder = main_folder
        self.data_type  = data_type
        self.deconvolution_method =deconvolution_method

        self.load_all_results(main_folder=self.main_folder, data_type=data_type, decoder=decoder, method=method)
        self.plot_mean_difference()


    def open_data_set(self, condition_folder, data_type, dataset_name, decoder, method):

        decoder_result = np.load(self.main_folder + condition_folder + "/" + dataset_name + "/decoder_results/" +
                                 dataset_name + "_" + self.deconvolution_method + "_" +
                                 data_type + "_" + decoder + "_" + method + "_decoder_score.npy")
        return(pd.DataFrame(decoder_result))

    def open_folder_of_datasets(self, condition_folder, data_type, decoder, method):
        data_sets = [os.path.basename(x) for x in glob.glob(self.main_folder + condition_folder + "/*_im_*")]
        data_sets = data_sets
        print(len(data_sets))
        list_of_data_frames =  [0]*len(data_sets)
        for i, d in enumerate(data_sets):
            print(d)
            data_frame = self.open_data_set(condition_folder=condition_folder, data_type = data_type, dataset_name = d, decoder = decoder, method = method)
            data_frame.columns = ['Real_labels', 'Texture', 'predicted_label', 'score']
            data_frame["Rearing_condition"] = np.repeat(condition_folder [5:7], data_frame.shape[0])
            data_frame["data_set"] = np.repeat(d, data_frame.shape[0])
            means = data_frame.groupby(['Texture', 'Real_labels']).mean()
            print(means.groupby(['Texture']).mean())
            list_of_data_frames [i] = data_frame
        return(pd.concat(list_of_data_frames, ignore_index=True))

    def load_all_results(self, main_folder, decoder, data_type, method):
            condition_folders = [os.path.basename(x) for x in glob.glob(main_folder + "/*WT_*")]
            #print(condition_folders)

            list_folder_df =  [0]*len(condition_folders)
            for i, f in enumerate(condition_folders):
                print(condition_folders)
                list_folder_df [i] = self.open_folder_of_datasets(condition_folder = f, data_type =data_type, method = method, decoder = decoder)
            self.decoder_data_frame = pd.concat(list_folder_df)
            del list_folder_df



    def plot_mean_difference(self):
        meaned_data = self.decoder_data_frame.groupby(["Rearing_condition", "data_set", "Texture", "Real_labels"]).mean()
        meaned_data = meaned_data.groupby(["Rearing_condition", "Texture"]).mean()
        print(meaned_data)












