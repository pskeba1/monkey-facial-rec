from skimage import transform,img_as_ubyte
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import cv2
import os
import pdb

def renderFace(im, landmarks,name=None):
    coords = []
    for p in landmarks:
         cv2.circle(im, (p[0], p[1]), radius=3, color=(10, 250, 0))
         coords.append([p[0],p[1]])

    if name:
        cv2.imshow(name,im); cv2.waitKey(0); cv2.destroyAllWindows()
    return(pd.DataFrame(coords))


def warp_faces():
	landmark_coords = pickle.load(open('../data/feret100_coords.p','rb'))
	avg_face = pickle.load(open('../data/feret100_avg.p','rb'))

	if not os.path.exists('../data/feret100_shapeless'):
		os.mkdir('../data/feret100_shapeless')

	doplot = True

	for p in landmark_coords.keys():
		img = cv2.imread('../data/feret100_straight/' + p)
		src = landmark_coords[p]

		tf = transform.estimate_transform('similarity', src.values, avg_face.values)
		result = img_as_ubyte(transform.warp(img, inverse_map=tf.inverse))#, output_shape=(320, 320, 3)))
		cv2.imwrite('../data/feret100_shapeless/' + p,cv2.resize(result,(120,120)))
	
		if doplot:
			plt.subplot(1, 2, 1), plt.imshow(img,'gray')
			plt.subplot(1, 2, 2), plt.imshow(result,'gray')
			plt.show()
			doplot = False

def shapeless_intensity():
	# Since not all images are in color, we should use gray intensities
	intensities = []
	facedir = '../data/feret100_shapeless'

	for f in os.listdir(facedir):
		img = cv2.imread(os.path.join(facedir,f),cv2.IMREAD_GRAYSCALE)

		# cv2.imshow(f,img); cv2.waitKey(0); cv2.destroyAllWindows()
		newrow = [f,*img.flatten()]
	
	intensities = pd.DataFrame(intensities)
	pickle.dump(intensities,open('../data/shapeless_intensities.p','wb'))

if __name__ == "__main__":
	warp_faces()
	shapeless_intensity()
