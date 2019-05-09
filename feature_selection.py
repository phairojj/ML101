#to select candidate features
#ex seqallenght vs seqwidht, petallenght vs petalwidgt
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

dataset = pd.read_csv("./Iris.csv")

print(dataset.describe())
#sns.pairplot(dataset)
sns.pairplot(dataset, hue="Species")
plt.show(sns)

sns.countplot(dataset["Species"])
plt.show(sns)

sns.distplot(dataset["SepalLengthCm"])
plt.show(sns)

iris_vericolor = dataset.loc[lambda data: dataset['Species'] == 'Iris-versicolor', : ]
iris_setosa = dataset.loc[lambda data: dataset["Species"] == "Iris-setosa", : ]
iris_virginica = dataset.loc[lambda data: dataset["Species"] == "Iris-virginica", : ]
sns.distplot(iris_vericolor["SepalLengthCm"], hist=False, color = "red")
sns.distplot(iris_setosa["SepalLengthCm"], hist=False, color = "green")
sns.distplot(iris_virginica["SepalLengthCm"], hist=False, color = "blue")
plt.show(sns)

sns.lmplot(x='SepalLengthCm', y='SepalWidthCm', col='Species', order=2, data=dataset)
plt.show()

sns.catplot(x='Species', y='PetalLengthCm', data=dataset)
plt.show()

# Relationship Between Sepal Length and Sepal Width
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', size="PetalLengthCm", hue='Species', data=dataset)
plt.show()

corr = dataset.corr()
sns.heatmap(round(corr,2), annot=True)
plt.show(sns)

