# Firstly, we will be reading the dataset in the form of a CSV File
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True)

data_path = "https://drive.google.com/file/d/1cSQl7_USJfiTf8UXAL1n75CdcY_N9riA/view?usp=share_link"
iris = pd.read_csv(data_path)

   # the iris dataset is now a Pandas DataFrame
print(iris.tail(10))
print("\n")
print(iris.shape)

   # Let's see how many examples we have of each species
iris["Species"].value_counts()


# Some Exploratory Data Analysis is required to understand the hidden features of the dataset.
   #Scatter Plot just to visualize the dataset on 2D
   # The first way we can plot things is using the .plot extension from Pandas dataframes
   # We'll use this to make a scatterplot of the Iris features.
iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")
plt.show()

iris.plot(kind="scatter", x="PetalLengthCm", y="PetalWidthCm")
plt.show()


# FacetGrid Plot to understand if any cluster is forming from the dataset or not.
   # One piece of information missing in the plots above is what species each plant is
   # We'll use seaborn's FacetGrid to color the scatterplot by species
sns.FacetGrid(iris, hue="Species", size=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()

   # One piece of information missing in the plots above is what species each plant is
   # We'll use seaborn's FacetGrid to color the scatterplot by species
sns.FacetGrid(iris, hue="Species", size=4).map(plt.scatter, "PetalLengthCm", "PetalWidthCm").add_legend()


# Box-Plot for visualizing any outliers that may be present in the dataset.
   # We can look at an individual feature in Seaborn through a boxplot
sns.boxplot(x="Species", y="PetalWidthCm", data=iris)

   # We can look at an individual feature in Seaborn through a boxplot
sns.boxplot(x="Species", y="SepalLengthCm", data=iris)

   # One way we can extend this plot is adding a layer of individual points on top of it through Seaborn's striplot
   # We'll use jitter=True so that all the points don't fall in single vertical lines above the species
   # Saving the resulting axes as ax each time causes the resulting plot to be shown on top of the previous axes
ax = sns.boxplot(x="Species", y="SepalWidthCm", data=iris)
ax = sns.stripplot(x="Species", y="SepalWidthCm", 
                   data=iris, jitter=True, edgecolor="gray")



# Facet-Grid Kernel Density Enstimation Plot to identify the distribution of the data# A final seaborn plot useful for looking at univariate relations is the kdeplot,
   # which creates and visualizes a kernel density estimate of the underlying feature
sns.FacetGrid(iris, hue="Species", size=6).map(sns.kdeplot, "SepalLengthCm").add_legend()


# Most important of all the pairplot to understand the correlation among the features according to the classes.
   # Another useful seaborn plot is the pairplot, which shows the bivariate relation between each pair of features
   # From the pairplot, we'll see that the Iris-setosa species is separataed from the other two across all feature combinations
sns.pairplot(iris, hue="Species", size=3, diag_kind="hist")


# Data Preprocessing for the Machine Learning Algorithm
features = iris.iloc[:, 1:4+1]
print(features)
targets = iris.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, targets, 
                                                    test_size=0.2, random_state=42)

import numpy as np
X_train, y_train = np.array(X_train, dtype=np.float64), np.array(y_train)

X_test, y_test = np.array(X_test, dtype=np.float64), np.array(y_test)

y_train[:5]

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train.reshape((-1, 1)))
y_test_encoded = encoder.transform(y_test.reshape((-1, 1)))

X_train[:10]

y_train_encoded[:10]

print("The Final Training Data Tensor shape is : {} for the features and {} for the targets.".format(X_train.shape, y_train_encoded.shape))

print("The Final Testing Data Tensor shape is : {} for the features and {} for the targets.".format(X_test.shape, y_test_encoded.shape))

