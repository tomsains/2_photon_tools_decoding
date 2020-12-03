import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob as glob
import pandas as pd
from sklearn.model_selection import train_test_split,LeaveOneOut, cross_val_score, cross_val_predict
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
    def __init__(self, folder, dataset_name, deconvolution_method="estimated"):
        self.folder = folder
        self.dataset_name = dataset_name
        self.deconvolution_method = deconvolution_method
        object = np.load(self.folder + self.dataset_name + '/preprocessed/' + "data_set_object_" + self.deconvolution_method + '.npy', allow_pickle=True)
        self.Spikes = object.item()['Spikes']
        self.Calcium = object.item()['Calcium']
        self.DFF = object.item()['DFF']
        self.NCC = object.item()['NCC']
        self.epoch_table = object.item()['epoch_table']
        self.epoch_vectors = object.item()['epoch_vectors']
        self.NCC_filter()
        self.make_epoch_labels()
        self.decoder_score_table()


    def NCC_filter(self):
        self.NCC = (self.NCC [:,0] > self.NCC [:,1])
        self.Spikes= self.Spikes [self.NCC,: ]
        self.DFF = self.DFF [self.NCC,:]
        self.Calcium = self.Calcium [self.NCC,:]

    def make_epoch_labels(self):
        le = LabelEncoder().fit(y=self.epoch_vectors[1][1:])
        labels = le.transform(self.epoch_vectors[1][1:])
        text_lab = self.epoch_vectors[2][1:]
        text_lab[text_lab == 0] = "gray"
        text_lab[text_lab == 1] = "Textured"
        self.epoch_labels = pd.DataFrame({"epoch_name": self.epoch_vectors[1][1:], "epoch_label": labels, "texture": self.epoch_vectors[2][1:],
                                     "texture_label": text_lab})


    def epoch_summary(self, traces, method="mean", extend_epoch=0):
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
                mean_per_epoch[:, i] = np.mean(traces [:, start:end], axis=1)
            elif method == "max":
                mean_per_epoch[:, i] = np.max(traces [:, start:end], axis=1)
        self.epoch_summ = mean_per_epoch [:, 1:]

    def textured_filter(self,  traces, texture=1, method="max"):
        self.epoch_summary(traces=traces, method=method, extend_epoch=1)
        if texture == 2:
            labels = np.array(self.epoch_labels["epoch_label"])
            data = self.epoch_summ
            texture = self.epoch_labels["texture"]
        else:
            labels = np.array(self.epoch_labels["epoch_label"])[self.epoch_labels["texture"] == texture]
            data = self.epoch_summ[:, self.epoch_labels["texture"] == texture]
            texture = self.epoch_labels["texture"][self.epoch_labels["texture"] == texture]
        self.decoder_input = (data.T, labels, texture)

    def decode_score(self, data, labels, texture, decoder="logreg"):
        if decoder == "logreg":
            pipe = Pipeline([('scaler', StandardScaler()), ('log_reg', LogisticRegression(C=1, solver='lbfgs', verbose=0, max_iter=10000, multi_class='multinomial'))])
        if decoder == "LDA":
            pipe = Pipeline([('scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis(n_components=2))])
        cv = LeaveOneOut()
        scores = cross_val_score(pipe, data, labels, cv=cv)
        predicts = cross_val_predict(pipe, data, labels, cv=cv)
        return(pd.DataFrame({"labels": labels, "texture": texture,  "predicts": predicts, "scores": scores}))


    def decoder_score_table(self, method = "max"):
        self.textured_filter(traces = self.Calcium, texture=0, method = method)
        non_textured_decoder_output = self.decode_score(data = self.decoder_input [0], labels = self.decoder_input [1], texture=self.decoder_input [2])
        self.textured_filter(traces=self.Calcium, texture=1, method=method)
        textured_decoder_output = self.decode_score(data=self.decoder_input[0], labels=self.decoder_input[1], texture=self.decoder_input[2])
        self.decoder_scores = pd.concat([non_textured_decoder_output, textured_decoder_output])
        print(self.decoder_scores.groupby(['texture', 'labels']).mean())








def apply_decoders(condition_folder):
    print(condition_folder)

    data_sets = [os.path.basename(x) for x in glob.glob(condition_folder +"/*_im_*")]
    print(len(data_sets))
    for d in data_sets:
        if os.path.isdir(condition_folder + "/" + d + "/suite2p") == True:
            print("processing .... " + d )
            decode(condition_folder=condition_folder + "/", dataset_name=d)



