#ref: https://www.kaggle.com/andradaolteanu/pulmonary-fibrosis-competition-eda-dicom-prep

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sci
import pydicom # for DICOM images
import re
import numpy as np

sns.set_style("whitegrid")

input_path = r"C:\Users\Leonor.furtado\OneDrive - Accenture\Uni\2nd Year\Data"

train = pd.read_csv(input_path + r"\train.csv")
test = pd.read_csv(input_path + r"\test.csv")

print("There are {} unique patients in Train Data.".format(len(train["Patient"].unique())), "\n")

# Recordings per Patient
data = train.groupby(by="Patient")["Weeks"].count().reset_index(drop=False)
# Sort by Weeks
data = data.sort_values(['Weeks']).reset_index(drop=True)
print("Minimum number of entries are: {}".format(data["Weeks"].min()), "\n" +
      "Maximum number of entries are: {}".format(data["Weeks"].max()))

# Plot
plt.figure(figsize = (16, 6))
p = sns.barplot(data["Patient"], data["Weeks"])

plt.title("Number of Entries per Patient", fontsize = 17)
plt.xlabel('Patient', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

p.axes.get_xaxis().set_visible(False);

# visualise bio info for the patient group
data = train.groupby(by="Patient")[["Patient", "Age", "Sex", "SmokingStatus"]].first().reset_index(drop=True)

# Figure
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (16, 6))

a = sns.distplot(data["Age"], ax=ax1, hist=False, kde_kws=dict(lw=6, ls="--"))
b = sns.countplot(data["Sex"], ax=ax2)
c = sns.countplot(data["SmokingStatus"], ax=ax3)

a.set_title("Patient Age Distribution", fontsize=16)
b.set_title("Sex Frequency", fontsize=16)
c.set_title("Smoking Status", fontsize=16)


print("Min FVC value: {:,}".format(train["FVC"].min()), "\n" +
      "Max FVC value: {:,}".format(train["FVC"].max()), "\n" +
      "\n" +
      "Min Percent value: {:.4}%".format(train["Percent"].min()), "\n" +
      "Max Percent value: {:.4}%".format(train["Percent"].max()))

# Figure
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))

a = sns.distplot(train["FVC"], ax=ax1,hist=False, kde_kws=dict(lw=6, ls="--"))
b = sns.distplot(train["Percent"], ax=ax2, hist=False, kde_kws=dict(lw=6, ls="-."))

a.set_title("FVC Distribution", fontsize=16)
b.set_title("Percent Distribution", fontsize=16);

print("Minimum no. weeks before CT: {}".format(train['Weeks'].min()), "\n" +
      "Maximum no. weeks after CT: {}".format(train['Weeks'].max()))

plt.figure(figsize = (16, 6))

a = sns.distplot(train['Weeks'], hist=False, kde_kws=dict(lw=8, ls="--"))
plt.title("Number of weeks before/after the CT scan", fontsize = 16)
plt.xlabel("Weeks", fontsize=14);

# Compute Correlation
corr1, _ = sci.pearsonr(train["FVC"], train["Percent"])
corr2, _ = sci.pearsonr(train["FVC"], train["Age"])
corr3, _ = sci.pearsonr(train["Percent"], train["Age"])
print("Pearson Corr FVC x Percent: {:.4}".format(corr1), "\n" +
      "Pearson Corr FVC x Age: {:.0}".format(corr2), "\n" +
      "Pearson Corr Percent x Age: {:.2}".format(corr3))

# Figure
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (16, 6))

a = sns.scatterplot(x = train["FVC"], y = train["Percent"],
                    hue = train["Sex"], style = train["Sex"], s=100, ax=ax1)

b = sns.scatterplot(x = train["FVC"], y = train["Age"],
                    hue = train["Sex"], style = train["Sex"], s=100, ax=ax2)

c = sns.scatterplot(x = train["Percent"], y = train["Age"],
                    hue = train["Sex"], style = train["Sex"], s=100, ax=ax3)

a.set_title("Correlation between FVC and Percent", fontsize = 16)
a.set_xlabel("FVC", fontsize = 14)
a.set_ylabel("Percent", fontsize = 14)

b.set_title("Correlation between FVC and Age", fontsize = 16)
b.set_xlabel("FVC", fontsize = 14)
b.set_ylabel("Age", fontsize = 14)

c.set_title("Correlation between Percent and Age", fontsize = 16)
c.set_xlabel("Percent", fontsize = 14)
c.set_ylabel("Age", fontsize = 14);


# Figure
f, (ax1, ax2) = plt.subplots(1,2, figsize = (16, 6))

a = sns.barplot(x = train["SmokingStatus"], y = train["FVC"], ax=ax1)
b = sns.barplot(x = train["SmokingStatus"], y = train["Percent"], ax=ax2)

a.set_title("Mean FVC per Smoking Status", fontsize=16)
b.set_title("Mean Perc per Smoking Status", fontsize=16);

# Create Time variable to count in ascending order the times the Patient has done a check in FVC
data_time = train.groupby(by="Patient")["Weeks"].count().reset_index()
train["Time"] = 0

for patient, times in zip(data_time["Patient"], data_time["Weeks"]):
    train.loc[train["Patient"] == patient, 'Time'] = range(1, times+1)

# For graph purposes, keep only Patients that had a big difference in FVC between Time 1 and last Time
min_fvc = train[train["Time"] == 1][["Patient", "FVC"]].reset_index(drop=True)

idx = train.groupby(["Patient"])["Weeks"].transform(max) == train["Weeks"]
max_fvc = train[idx][["Patient", "FVC"]].reset_index(drop=True)

# Compute difference and select only top patients with biggest difference
data = pd.merge(min_fvc, max_fvc, how="inner", on="Patient")
data["Dif"] = data["FVC_x"] - data["FVC_y"]

# Select only top n
l = list(data.sort_values("Dif", ascending=False).head(100)["Patient"])
x = train[train["Patient"].isin(l)]

plt.figure(figsize = (16, 6))

a = sns.lineplot(x = x["Time"], y = x["FVC"], hue = x["Patient"], legend=False,
                 palette=sns.color_palette("GnBu_d", 100), size=1)

plt.title("Patient FVC decrease on Weeks", fontsize = 16)
plt.xlabel("Weeks", fontsize=14)
plt.ylabel("FVC", fontsize=14);

train_dir = r"C:\Users\Leonor.furtado\OneDrive - Accenture\Uni\2nd Year\Data\train"
# Create path column with the path to each patient's CT
train["Path"] = train_dir + "\\" + train["Patient"]

# Create variable that shows how many CT scans each patient has
train["CT_number"] = 0

for k, path in enumerate(train["Path"]):
    train["CT_number"][k] = len(os.listdir(path))

#Huge imbalance in the number of CT scans: half of the patients have less that 100 photos registered.

print("Minimum number of CT scans: {}".format(train["CT_number"].min()), "\n" +
      "Maximum number of CT scans: {:,}".format(train["CT_number"].max()))

# Scans per Patient
data = train.groupby(by="Patient")["CT_number"].first().reset_index(drop=False)
# Sort by Weeks
data = data.sort_values(['CT_number']).reset_index(drop=True)

# Plot
plt.figure(figsize = (16, 6))
p = sns.barplot(data["Patient"], data["CT_number"])
plt.axvline(x=85 , linestyle='--', lw=3)

plt.title("Number of CT Scans per Patient", fontsize = 17)
plt.xlabel('Patient', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.text(86, 850, "Median=94", fontsize=13)

p.axes.get_xaxis().set_visible(False);

class bcolors:
    OKBLUE = '\033[96m'
    OKGREEN = '\033[92m'

#test

path = r"\ID00007637202177411956430\19.dcm"
dataset = pydicom.dcmread(train_dir + path)

print(bcolors.OKBLUE + "Patient id.......:", dataset.PatientID, "\n" +
      "Modality.........:", dataset.Modality, "\n" +
      "Rows.............:", dataset.Rows, "\n" +
      "Columns..........:", dataset.Columns)

plt.figure(figsize = (7, 7))
plt.imshow(dataset.pixel_array, cmap="plasma")
plt.axis('off');

#patient test

patient_dir = train_dir + r"/ID00007637202177411956430"
datasets = []

# First Order the files in the dataset
files = []
for dcm in list(os.listdir(patient_dir)):
    files.append(dcm)
files.sort(key=lambda f: int(re.sub('\D', '', f)))

# Read in the Dataset
for dcm in files:
    path = patient_dir + "/" + dcm
    datasets.append(pydicom.dcmread(path))

# Plot the images
fig=plt.figure(figsize=(16, 6))
columns = 10
rows = 3

for i in range(1, columns*rows +1):
    img = datasets[i-1].pixel_array
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap="plasma")
    plt.title(i, fontsize = 9)
    plt.axis('off');


# https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/

# Segmentation
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
# init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings("ignore")

def make_lungmask(img, display=False):
    row_size = img.shape[0]
    col_size = img.shape[1]

    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std

    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)

    # To improve threshold finding, I'm moving the
    # underflow and overflow on the pixel spectrum
    img[img == max] = mean
    img[img == min] = mean

    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)

    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
    dilation = morphology.dilation(eroded, np.ones([8, 8]))

    labels = measure.label(dilation)  # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < row_size / 10 * 9 and B[3] - B[1] < col_size / 10 * 9 and B[0] > row_size / 5 and B[
            2] < col_size / 5 * 4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size, col_size], dtype=np.int8)
    mask[:] = 0

    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask

    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask * img, cmap='gray')
        ax[2, 1].axis('off')

        plt.show()
    return mask * img

#test mask one image
path = train_dir + "//ID00007637202177411956430//19.dcm"
dataset = pydicom.dcmread(path)
img = dataset.pixel_array

# Masked image
mask_img = make_lungmask(img, display=True)

#test mask one patient

patient_dir = train_dir + "//ID00007637202177411956430"
datasets = []

# First Order the files in the dataset
files = []
for dcm in list(os.listdir(patient_dir)):
    files.append(dcm)
files.sort(key=lambda f: int(re.sub('\D', '', f)))

# Read in the Dataset
for dcm in files:
    path = patient_dir + "\\" + dcm
    datasets.append(pydicom.dcmread(path))

imgs = []
for data in datasets:
    img = data.pixel_array
    imgs.append(img)

# Show masks
fig = plt.figure(figsize=(16, 6))
columns = 10
rows = 3

for i in range(1, columns*rows +1):
    img = make_lungmask(datasets[i-1].pixel_array)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap="gray")
    plt.title(i, fontsize = 9)
    plt.axis('off');