import pandas as pd
import matplotlib.pyplot as plt

test = pd.read_csv("../lgi-ppgi-db/gtdump.xmp")

#print(test)

print(test.iloc[0])
hr = []
for hang in test.iloc:
    hr.append(hang['68'])
    
    
plt.plot(hr)
plt.ylim(50, 150)
plt.show()