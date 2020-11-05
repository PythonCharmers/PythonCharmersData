#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Perform supervised classification on landsat data with a linear SVC, and compare to kmeans clustering
"""
import os.path
from glob import glob

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn import metrics
from osgeo import gdal
from matplotlib import pyplot as plt

DATADIR = "/Users/henry/Dropbox/work/PythonCharmers/Spatial_course/exercise_solutions/Data/lsat7_2002"

##Read in data

X = []

shp = None
geo_transform = None
prj = None

for dpath in glob(os.path.join(DATADIR, "*.tif")):
    dset = gdal.Open(dpath)
    lyr = dset.GetRasterBand(1)
    a = lyr.ReadAsArray()
    a = a.astype(np.int)
    shp = a.shape
    geo_transform = dset.GetGeoTransform()
    prj = dset.GetProjection()
    X.append(a.flatten())

X = np.vstack(X).T

##Supervised classification

samples = np.loadtxt(open(os.path.join(DATADIR, 'samples.csv')), skiprows=1, delimiter=",")

y = samples[:, 0]
samples = samples[:, 1:]

clf = LinearSVC()
clf.fit(samples, y)

supervised = clf.predict(X)
supervised = supervised.reshape(shp)

##Unsupervised classification

kmeans = KMeans(n_clusters=5)
unsupervised = kmeans.fit_predict(X)
unsupervised = unsupervised.reshape(shp)

plt.subplot(121)
plt.title("Supervised classification")
plt.imshow(supervised)
plt.subplot(122)
plt.title("Unsupervised classification")
plt.imshow(unsupervised)
plt.show()

#geotiff = gdal.GetDriverByName("GTiff")

print supervised.shape, shp

# out_dset = geotiff.Create(os.path.join(os.path.dirname(__file__), "supervised.tif"), shp[1], shp[0])
# out_lyr = out_dset.GetRasterBand(1)
# out_lyr.WriteArray(supervised)
# out_dset.SetGeoTransform(geo_transform)
# out_dset.SetProjection(prj)
#
# out_dset = geotiff.Create(os.path.join(os.path.dirname(__file__), "unsupervised.tif"), shp[1], shp[0])
# out_lyr = out_dset.GetRasterBand(1)
# out_lyr.WriteArray(unsupervised)
# out_dset.SetGeoTransform(geo_transform)
# out_dset.SetProjection(prj)
#

print metrics.homogeneity_score(supervised.flatten(), unsupervised.flatten())
print metrics.completeness_score(supervised.flatten(), unsupervised.flatten())
print len(set(supervised.flatten()))