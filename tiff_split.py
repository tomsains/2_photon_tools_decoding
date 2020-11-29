 Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from skimage.external.tifffile import imread, TiffFile
import os
import glob
from skimage import io
import argparse

def tif_split(tif, slice_num = 6, mean_bin =2):
    os.mkdir(str(tif [:-4]))
    img = imread(tif)
    for i in range(slice_num -1):
        plane_mean = np.mean(img [list(range(i,img.shape [0],slice_num)),:,:].reshape((-1,mean_bin,img.shape[2],img.shape[1])), axis = 1, dtype=np.int16)
        io.imsave(arr=plane_mean, fname=tif [:-4] +"/slice" + str(i+1) + ".tif")

def batch_tif_split(folder, pattern = "*.tif"):
    files = sorted(glob.glob(pattern))
    for files in files:
        print(files)
        tif_split(files)

# Press the green button in the gutter to run the script.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/



if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('folder')
   args = parser.parse_args()
   batch_tif_split(folder = args.folder)

