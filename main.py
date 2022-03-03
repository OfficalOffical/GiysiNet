from compVis import *
from editCsv import getAndCleanCsv

datasetPath = "E:/db"
csvPath = "C:/Users/Sefa/Desktop/outfit_data_cleaned.csv"

temp = pd.read_csv(csvPath)

plt.figure(figsize=(7, 25))
temp.categoryx_id.value_counts().sort_values().plot(kind='barh')
plt.show()




a = getImageFromCSV(csvPath)











