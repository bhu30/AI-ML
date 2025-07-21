# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# set the graps to show  in the jupyter notebook
%matplotlib inline

# COMMAND ----------

# set seaborn graphs to a better styli
sns.set(style="ticks")

# COMMAND ----------

path = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Online_Retail/Online_Retail.csv"

online_rt = pd.read_csv(path, encoding='latin1')

online_rt.head()

# COMMAND ----------

# group by the Country
countries = online_rt.groupby('Country').sum()

# COMMAND ----------

# sort the value and get the first 10 after UK
countries = countries.sort_values('Quantity', ascending=False).head(10)
display(countries.head())

# COMMAND ----------

# create the plot
countries['Quantity'].plot(kind='bar')  

# COMMAND ----------

# set the title and labels
plt.title('Top Countries by Quantity')
plt.xlabel('Country')
plt.ylabel('Quantity')
plt.title('10 countries with most orders')

# COMMAND ----------

# show the plot
plt.show()

# COMMAND ----------

online_rt=online_rt[online_rt.Quantity>0]
online_rt.head()

# COMMAND ----------

# groupby CustomerID
customers = online_rt.groupby(['CustomerID','Country']).sum()
customers['Country']=customers.index.get_level_values(1)
top_countries=['Netherlands','EIRE','Germany']
customers=customers[customers['Country'].isin(top_countries)]
# Graph Section #
# creates the FaceGrid
g=sns.FacetGrid(customers,col="Country")
# maps over a make a scatterplot
g.map(plt.scatter, "Quantity", "UnitPrice", alpha=1)
# adds legend
g.add_legend()

# COMMAND ----------

# adds Legend
g.add_legend()