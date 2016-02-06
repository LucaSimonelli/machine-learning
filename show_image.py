#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
img = ndimage.imread('./stinkbug.png')
plt.imshow(img)
plt.show()
