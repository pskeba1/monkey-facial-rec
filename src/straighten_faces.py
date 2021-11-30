import cv2
import dlib
import pickle
import re
import os
from detect_landmarks import renderFace

def get_face_patch(img,detector,predictor):
	"""
	Crop face to uniform 320x320 size and straighten. Uses 5-point landmark detection
	and dlib get_face_chips to geometrically transform image.
	"""

	# Find the bounding box of the face and upscale so we don't lose quality
	dets = detector(img, 1)

	if len(dets) == 0:
	    print("Sorry, there were no faces found in '{}'".format(face_file_path))
	    return(1)
	 
	# Find the 5 face landmarks we need to do the alignment.
	faces = dlib.full_object_detections()
	for detection in dets:
	    faces.append(predictor(img, detection))

	images = dlib.get_face_chips(img, faces, size=320,padding=0.3)
	return(images[0])

def main():
	
	# Finds rectangular box around front-facing faces in the data
	detector = dlib.get_frontal_face_detector()

	try:
		predictor = dlib.shape_predictor("../models/shape_predictor_5_face_landmarks.dat")
	except RuntimeError:
		print('5-point landmark model missing: ../models/shape_predictor_5_face_landmarks.dat. See README for download link.')
		exit()

	if not os.path.exists('../data/feret100_straight'):
		os.mkdir('../data/feret100_straight')

	
	try:
		face_dir = '../data/colorferet100'
		people = os.listdir(face_dir)
	except FileNotFoundError:
		print('Image data missing: ..data/colorferet100/. See README for download link')
		exit()

	for p in people:
		p_dir = os.path.join(face_dir,p)

		try:
			p_files = os.listdir(p_dir)
		except NotADirectoryError:
			print('This is not a directory: ' + p_dir)
			continue

		i = 0
		for f in p_files:
			if re.search('.*fa.*ppm',f):
				img_path = os.path.join(p_dir,f)
				img = cv2.imread(img_path)
				

				img2 = get_face_patch(img,detector,predictor)
				cv2.imwrite('../data/feret100_straight/' + p + '_' + str(i) + '.ppm',img2)
				i += 1
				# winname = 'Straightened Face'
				# cv2.imshow(winname,img2); cv2.waitKey(0); cv2.destroyAllWindows()

if __name__ == "__main__":main()


