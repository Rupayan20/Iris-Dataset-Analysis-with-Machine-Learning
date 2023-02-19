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


#Some Exploratory Data Analysis is required to understand the hidden features of the dataset.
   #Scatter Plot just to visualize the dataset on 2D
   # The first way we can plot things is using the .plot extension from Pandas dataframes
   # We'll use this to make a scatterplot of the Iris features.
iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")
plt.show()

iris.plot(kind="scatter", x="PetalLengthCm", y="PetalWidthCm")
plt.show()
