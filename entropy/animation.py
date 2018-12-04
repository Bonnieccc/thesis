import os
import sys

import imageio

base_dir = '/Users/abbyvansoest/thesis/entropy/figs/'

# example: python animation.py 2018_12_02-22-15
directory = base_dir + sys.argv[1] + '/'
print(directory)

images = []
for filename in sorted(os.listdir(directory)):
    if 'heatmap_' not in filename:
        continue
    images.append(imageio.imread(directory+filename))

imageio.mimsave(directory + '/movie.mp4', images)