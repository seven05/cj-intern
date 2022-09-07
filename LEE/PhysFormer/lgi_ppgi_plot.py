import matplotlib.pyplot as plt
import xml.etree.ElementTree as elemTree
from lxml import etree 
import numpy as np

path = '../lgi-ppgi-db/id3/cpi/cpi_rotation/cms50_stream_handler.xml'
tree = elemTree.parse(path)
doc = etree.parse(path)
root = doc.getroot()
length = len(root.getchildren())

prs = []
for i in range(length-1):
    str_ = 'value'+str(i)

    frame_cnt =tree.find('./'+str_)

    fc = frame_cnt.find('value0').text
    pr = frame_cnt.find('value1').text
    ap = frame_cnt.find('value2').text

    # print(fc)
    # print(pr)
    # print(ap)
    # print()
    #print(i)
    # if int(pr)>150 or int(pr)<50:
    #     pr = prs[-1]

    prs.append(int(pr))

    # if i>10:
    #     break

print(np.mean(prs))

plt.plot(prs)
plt.ylim([50, 150])
plt.title('gt')
plt.show()