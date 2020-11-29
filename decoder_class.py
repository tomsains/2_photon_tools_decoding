import numpy as np
import seaborn as sns
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
