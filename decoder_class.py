import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob as glob
import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, cross_val_predict
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


class decode:
    def __init__(self, condition_folder, dataset_name, deconvolution_method="estimated", NCC_scaler = 0.025):
        self.folder = condition_folder
        self.dataset_name = dataset_name
        self.deconvolution_method = deconvolution_method
        self.NCC_scaler = NCC_scaler
        if os.path.isdir(self.folder + self.dataset_name + "/decoder_results") == False:
            os.mkdir(self.folder + self.dataset_name + "/decoder_results")

        object = np.load(
            self.folder + self.dataset_name + '/preprocessed/' + "data_set_object_" + self.deconvolution_method + '.npy',
            allow_pickle=True)
        self.Spikes = object.item()['Spikes']
        self.Calcium = object.item()['Calcium']
        self.DFF = object.item()['DFF']
        self.NCC_val = object.item()['NCC']
        #print(self.NCC_val)

        self.epoch_table = object.item()['epoch_table']
        self.epoch_vectors = object.item()['epoch_vectors']

        self.make_epoch_labels()

        #self.apply_S_DFF()

        self.decoder_spikes_and_cal()
        self.NCC_curve()
        #self.decode_all_data_types()
        #self.decode_all_data_types(decoder="LDA")


    def smoothed_trace(self, t):
        S = pd.Series(t)
        smooth = S.rolling(window=10, win_type='gaussian', center=True).mean(std=2)
        return (smooth)

    def apply_S_DFF(self):
        self.S_DFF = np.zeros((self.DFF.shape))
        for i in range(self.DFF.shape[0]):
            if i % 500 == 0:
                print(i)
            self.S_DFF[i, :] = np.array(self.smoothed_trace(self.DFF[i, :]))

    def NCC_filter(self, traces, NCC_scaler, texture = 0):
        if texture == 0:
            NCC = (self.NCC_val[:, 0] > (self.NCC_val[:, 1] + (self.NCC_val[:, 2] * NCC_scaler)))
        elif texture == 1:
            NCC = (self.NCC_val[:, 3] > (self.NCC_val[:, 4] + (self.NCC_val[:, 5] * NCC_scaler)))
        return(traces [NCC,:])

    def make_epoch_labels(self):
        le = LabelEncoder().fit(y=self.epoch_vectors[1][1:])
        labels = le.transform(self.epoch_vectors[1][1:])
        text_lab = self.epoch_vectors[2][1:]
        text_lab[text_lab == 0] = "gray"
        text_lab[text_lab == 1] = "Textured"
        self.epoch_labels = pd.DataFrame(
            {"epoch_name": self.epoch_vectors[1][1:], "epoch_label": labels, "texture": self.epoch_vectors[2][1:],
             "texture_label": text_lab})
        #print(self.epoch_labels["epoch_name"])
        #print(self.epoch_labels["epoch_label"])

    def epoch_summary(self, traces, method="mean", extend_epoch= (0)):
        #print(method)
        epoch_start = np.array(self.epoch_table["fr"][0::2])
        epoch_end = np.array(self.epoch_table["fr"][1::2]) + extend_epoch
        # method can either be mean or max
        mean_per_epoch = np.zeros((traces.shape[0], self.epoch_vectors[1].shape[0]))
        for i in range(len(epoch_start)):
            # print(i)
            start = epoch_start[i]
            end = epoch_end[i]
            # print(start)
            if method == "mean":
                mean_per_epoch[:, i] = np.mean(traces[:, start:end], axis=1)
            if method == "max":
                mean_per_epoch[:, i] = np.max(traces[:, start:end], axis=1)
            if method == "sum":
                mean_per_epoch[:, i] = np.sum(traces[:, start:end], axis=1)
        self.epoch_summ = mean_per_epoch[:, 1:]

    def textured_filter(self, traces, texture=1, method="max", NCC_scaler = 0.14, exclude_epochs=True):
        traces = self.NCC_filter(traces=traces, NCC_scaler=NCC_scaler, texture=texture)
        print(traces.shape)
        self.epoch_summary(traces=traces, method=method, extend_epoch=0)
        if texture == 2:
            labels = np.array(self.epoch_labels["epoch_label"])
            data = self.epoch_summ
            texture = self.epoch_labels["texture"]
        else:
            labels = np.array(self.epoch_labels["epoch_label"])[self.epoch_labels["texture"] == texture]
            data = self.epoch_summ[:, self.epoch_labels["texture"] == texture]
            texture = self.epoch_labels["texture"][self.epoch_labels["texture"] == texture]
        if exclude_epochs == True:
            label_exclusions = (labels != 7) & (labels != 6)
            self.decoder_input = (data[:, label_exclusions].T, labels[label_exclusions], texture[label_exclusions])
        elif exclude_epochs == False:
            self.decoder_input = (data.T, labels, texture)

    def decode_score(self, data, labels, texture, decoder="logreg"):
        if decoder == "logreg":
            pipe = Pipeline([('scaler', StandardScaler()), (
            'log_reg', LogisticRegression(C=1, solver='lbfgs', verbose=0, max_iter=10000, multi_class='multinomial'))])
        if decoder == "LDA":
            pipe = Pipeline([('scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis(n_components=2))])
        cv = LeaveOneOut()
        scores = cross_val_score(pipe, data, labels, cv=cv)
        predicts = cross_val_predict(pipe, data, labels, cv=cv)
        return (pd.DataFrame({"labels": labels, "texture": texture, "predicts": predicts, "scores": scores}))


    def NCC_curve(self):
        NCC_scalers = np.linspace(0.01, 0.3, 20)
        df_list = [0] * len(NCC_scalers)
        for i, N in enumerate(NCC_scalers):
            df_list[i] = self.decoder_score_table(data_type=self.Spikes, dt_name="Spikes", NCC_scaler=N, method="sum", save=False).T.loc["scores"]
            print(N)
            print(self.Spikes.shape)
        df_list = pd.concat(df_list, axis=1).T
        self.NCC_scores = df_list.set_index(NCC_scalers)
        self.NCC_scores.plot()
        plt.savefig(fname = self.folder + self.dataset_name + "/decoder_results/" + self.dataset_name + "_NCC_curve.png")


    def decoder_score_table(self, data_type, dt_name, method="max", decoder="logreg", NCC_scaler=0.14, save = True):
        self.textured_filter(traces=data_type, texture=0, method=method, NCC_scaler = NCC_scaler)
        non_textured_decoder_output = self.decode_score(data=self.decoder_input[0], labels=self.decoder_input[1],
                                                        texture=self.decoder_input[2], decoder=decoder)
        self.textured_filter(traces=data_type, texture=1, method=method, NCC_scaler = NCC_scaler)
        textured_decoder_output = self.decode_score(data=self.decoder_input[0], labels=self.decoder_input[1],
                                                    texture=self.decoder_input[2], decoder=decoder)
        self.decoder_scores = pd.concat([non_textured_decoder_output, textured_decoder_output])
        self.decoder_scores = self.decoder_scores.groupby(['texture', 'labels']).mean()
        self.decoder_scores = self.decoder_scores.groupby(['texture']).mean()

        print(self.decoder_scores)

        if save == False:
            return(self.decoder_scores)

        if save == True:

            self.decoder_scores.to_csv(self.folder + self.dataset_name + "/decoder_results/" + self.dataset_name + "_" + self.deconvolution_method + "_" + dt_name + "_" + decoder + "_" + method + "_decoder_score.npy")


    def decode_all_data_types(self, decoder = "logreg"):
        print("Calcium:")
        self.decoder_score_table(data_type=self.Calcium, dt_name="Cal", method="max", decoder = decoder)
        self.decoder_score_table(data_type=self.Calcium, dt_name="Cal", method="mean", decoder = decoder)

        print("Spikes: ")
        self.decoder_score_table(data_type=self.Spikes, dt_name="Spikes", method="mean", decoder = decoder)
        self.decoder_score_table(data_type=self.Spikes, dt_name="Spikes", method="sum", decoder = decoder)

        print("DFF_S:")
        self.decoder_score_table(data_type=self.S_DFF, dt_name="DFF_S", method="max", decoder = decoder)
        self.decoder_score_table(data_type=self.S_DFF, dt_name="DFF_S", method="mean", decoder = decoder)

        print("DFF : ")
        self.decoder_score_table(data_type=self.DFF, dt_name="DFF", method="max", decoder = decoder)
        self.decoder_score_table(data_type=self.DFF, dt_name="DFF", method="mean", decoder = decoder)

    def decoder_spikes_and_cal(self, decoder = "logreg"):
        print("Calcium:")
        self.decoder_score_table(data_type=self.Calcium, dt_name="Cal", method="max", decoder=decoder)
        self.decoder_score_table(data_type=self.Calcium, dt_name="Cal", method="mean", decoder=decoder)

        print("Spikes: ")
        self.decoder_score_table(data_type=self.Spikes, dt_name="Spikes", method="mean", decoder=decoder)
        self.decoder_score_table(data_type=self.Spikes, dt_name="Spikes", method="sum", decoder=decoder)



def apply_decoders(condition_folder):
    data_sets = [os.path.basename(x) for x in glob.glob(condition_folder +"/*_im_*")]
    print(len(data_sets))
    for d in data_sets:
        if os.path.isdir(condition_folder + "/" + d + "/suite2p") == True:
            print("processing .... " + d )
            decode(condition_folder=condition_folder + "/", dataset_name=d)



