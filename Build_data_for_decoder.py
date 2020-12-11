import numpy as np

import matplotlib.pyplot as plt
import glob as glob
import pandas as pd
import os
import tqdm
import sys
import pickle

from matplotlib import animation

from matplotlib import animation, rc
from IPython.display import HTML
from scipy.signal import find_peaks

from sklearn.preprocessing import LabelEncoder
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.manifold import Isomap, TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from skimage.io import imread, imsave, concatenate_images
from sklearn.metrics import pairwise_distances
from scipy import stats
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind_from_stats

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from skimage import io
from skimage.external.tifffile import imread, TiffFile, imsave
from skimage.filters import threshold_otsu
from scipy import signal


class data_set:
    def __init__(self, folder, dataset_name, deconvolution_method="estimated"):
        self.folder = folder
        self.dataset_name = dataset_name
        self.epoch_time_file = self.folder + self.dataset_name + "/Imaging_logs/time_epoches.log"
        self.DFF = np.load(self.folder + self.dataset_name + "/preprocessed/" + dataset_name + "_DFF.npy")
        self.sample_rate = 9.7
        self.Centers = np.load(self.folder + self.dataset_name + "/preprocessed/" + dataset_name + "_cell_centers.npy")
        self.deconvolution_method = deconvolution_method
            # load deconvolution output, based on chosen method
        print("loading_deconvolved_data")
        if deconvolution_method == "estimated":
            self.Calcium = np.load(self.folder + self.dataset_name + "/preprocessed/" + dataset_name + "_oasis_c.npy")
            self.Spikes = np.load(self.folder + self.dataset_name + "/preprocessed/" + dataset_name + "_oasis_s.npy")

        if deconvolution_method == "AR1":
            self.Calcium = np.load(self.folder + self.dataset_name + "/preprocessed/" + dataset_name + "_oasisAR1_c.npy")
            self.Spikes = np.load(self.folder + self.dataset_name + "/preprocessed/" + dataset_name + "_oasisAR1_s.npy")

            # Sort cells based on their ant-post axis within the tectum
        print("AP sorting")
        self.get_AP()

            # Genterate epoch labels data frame
        self.make_epoch_table()
        self.create_epoch_vector()

        # binarise traces and calculate the NCC for each neuron
        self.binary = (self.Calcium > 0.05) * 1
        print("generate epoch_table")

        self.NCC_val = np.apply_along_axis(arr=self.binary, axis=1, func1d=self.NCC_both_epochs,
                                            epoch_times=self.epoch_vectors_list[0], texture = self.epoch_vectors_list[2])

        print("saving")
        self.save()

        #print("making_plots")
        #if os.path.isdir(self.folder + self.dataset_name + "/preprocessed/plots") == False:
        #os.mkdir(self.folder + self.dataset_name + "/preprocessed/plots")

    # methods
    def NCC(self, trace, epoch_times):
        trace = np.where(trace > 0, 1, -1)
        epoch_times = np.where(epoch_times > 0, 1, -1)
        Px = np.sum(trace == 1) / len(trace)
        NCC_value = np.sum(trace * epoch_times) / len(trace)
        # s = (np.sum(epoch_times == 1) - np.sum(epoch_times == -1)) / len(epoch_times)

        # if Px > 0:
        #   u = (2 * (Px)) - 1
        #  std_u = np.sqrt(4*((-Px**2)+Px))/ len(epoch_times)
        # else:
        #   u = 0
        #  std_u = 0
        threshMean = (np.mean(trace)) * np.mean(epoch_times)
        threshSD = np.std(trace)
        return(list([NCC_value, threshMean, threshSD]))

    def NCC_both_epochs(self, trace, epoch_times, texture):
        # Tom Shall uses 0.025
        epoch_times = epoch_times[300:]
        trace = trace[300:]
        first_half = self.NCC(trace [:19473], epoch_times [:19473])
        second_half = self.NCC(trace [19473:], epoch_times [19473:])
        if texture [0] == 0:
            return (first_half + second_half)
        elif texture [0] == 1:
            return (second_half + first_half)

    def rotate_via_numpy(self, xy, radians):
        """Use numpy to build a rotation matrix and take the dot product."""
        x, y = xy
        c, s = np.cos(radians), np.sin(radians)
        j = np.array([[c, s], [-s, c]])
        m = np.dot(j, [x, y])

        return float(m.T[0]), float(m.T[1])

    def get_AP(self):
        AP = [0] * len(np.unique(self.Centers[:, 2]))
        for i, p in enumerate(np.unique(self.Centers[:, 2])):
            temp_c = self.Centers[self.Centers[:, 2] == p, 0:2]
            temp_c[:, 0] = temp_c[:, 0] - np.min(temp_c[:, 0])
            temp_c[:, 1] = temp_c[:, 1] - np.min(temp_c[:, 1])
            AP[i] = np.apply_along_axis(arr=temp_c, func1d=self.rotate_via_numpy, axis=1, radians=50 * (np.pi / 180))
        AP = np.concatenate(AP)[:, 1].ravel().argsort()
        self.Calcium = self.Calcium[AP, :]
        self.Spikes = self.Spikes[AP, :]
        self.DFF = self.DFF[AP, :]
        self.Centers = self.Centers[AP, :]

    def plot_DFF_w_bin(self, cell_number):
        plt.figure(figsize=(20, 10))
        plt.plot(self.DFF[cell_number, :])
        plt.plot(self.binary[cell_number, :])

    def make_epoch_table(self, offset=0):
        print("creating epoch_table")
        epoch = pd.read_csv(self.epoch_time_file, delimiter=" ", header=None)
        epoch.columns = ["sec", "stim_name", "textured"]
        epoch["fr"] = ((epoch["sec"] + offset) * (self.sample_rate)).astype(int)
        self.epoch = epoch

    def create_epoch_vector(self, extend_epoch=(2 * 9.7)):
        #extend_epoch_vector_usually 10*9.7
        print("creating epoch_vectors")
        start = self.epoch["fr"].iloc[list(range(0, self.epoch.shape[0], 2))].values
        end = self.epoch["fr"].iloc[list(range(1, self.epoch.shape[0], 2))].values
        if extend_epoch > 0:
            end = (end + extend_epoch).astype(int)

        epoch_locations = np.repeat(0, self.DFF.shape[1])
        epoch_names = self.epoch["stim_name"].iloc[list(range(0, self.epoch.shape[0], 2))]
        texture = self.epoch["textured"].iloc[list(range(0, self.epoch.shape[0], 2))]
        counter = 1
        for i in range(len(start)):
            epoch_locations[start[i]:end[i]] = counter
            counter += 1
        self.epoch_vectors_list = list([epoch_locations, epoch_names, texture])

    def plot_trace_w_epochs(self, cell_number):
        epoch_start = self.epoch["fr"][::2]
        epoch_end = self.epoch["fr"][1::2]
        plt.figure(figsize=(20, 5))
        for index, (start, stop) in enumerate(zip(epoch_start, epoch_end)):
            plt.axvspan(start, stop + (9.7 * 4), alpha=0.2, color='gray')
            # plt.annotate(e ["stim_name"] [::2].iloc [index][-1], (-2, np.mean(start + stop)), size =0.1)
            # print(e ["stim_name"] [::2].iloc [index][-1])
            plt.annotate(str(self.epoch["stim_name"][::2].iloc[index][-1]), (start, -0.002), size=15)
        plt.plot(self.DFF[cell_number, :])
        plt.plot(self.Calcium[cell_number, :])

    def plot_rasters(self, ratio=4):
        fig, ax = plt.subplots(3, figsize=(60, 20), sharex='col')
        ncc_filt = self.NCC_val[:, 0] > (self.NCC_val[:, 1] + (0.025*self.NCC_val [:,2]))
        ax[0].imshow(self.DFF[ncc_filt, :], aspect=ratio)
        ax[1].imshow(self.Calcium[ncc_filt, :], aspect=ratio)
        ax[2].imshow(self.Spikes[ncc_filt, :], aspect=ratio, vmax=0.01)
        plt.savefig(self.folder + self.dataset_name + "/preprocessed/plots/" + self.dataset_name + self.deconvolution_method + "_raster_plots.png")

    def save(self):
        object_to_save = dict({"folder": self.folder,
                               "epoch_folder": self.epoch_time_file,
                               "centers": self.Centers,
                               "DFF": self.DFF,
                               "Spikes": self.Spikes,
                               "Calcium": self.Calcium,
                               "NCC": self.NCC_val,
                               "epoch_table": self.epoch,
                               "epoch_vectors": self.epoch_vectors_list})
        np.save(self.folder + self.dataset_name + '/preprocessed/' + "data_set_object_" + self.deconvolution_method + '.npy', object_to_save, allow_pickle=True)


def apply_build_data_for_decoder(condition_folder):
    print(condition_folder)

    data_sets = [os.path.basename(x) for x in glob.glob(condition_folder +"/*_im_*")]
    print(len(data_sets))
    for d in data_sets:
        if os.path.isdir(condition_folder + "/" + d + "/suite2p") == True:
            print("processing .... " + d )
            data_set(folder=condition_folder + "/", dataset_name=d)


