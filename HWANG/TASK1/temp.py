import json
import pandas as pd

df = pd.DataFrame(confusion_matrix(test["Churn"], predictions), 
                  columns=["False", "True"], 
                  index=["False", "True"])
df.index.name= 'Actual'
df.columns.name= 'Predicted'
print (df)
# Predicted  False  True
# Actual                
# False        877   183
# True         100   179

confusionmatrix = df.unstack().rename('value').reset_index()
print (confusionmatrix)
#   Predicted Actual  value
# 0     False  False    877
# 1     False   True    100
# 2      True  False    183
# 3      True   True    179

results = [{'confusionmatrix' : confusionmatrix}]

final = pd.Series(results).to_json(orient='records')
print (final)
# [{"confusionmatrix":[{"Predicted":"False","Actual":"False","value":877},
#                      {"Predicted":"False","Actual":"True","value":100},
#                      {"Predicted":"True","Actual":"False","value":183},
#                      {"Predicted":"True","Actual":"True","value":179}]}]