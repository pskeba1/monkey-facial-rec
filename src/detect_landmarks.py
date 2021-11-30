import cv2
import dlib
import sys
import os
import pandas as pd
import numpy as np
import pickle


def renderFace(im, landmarks,name=None):
    coords = []
    for p in landmarks.parts():
         cv2.circle(im, (p.x, p.y), radius=3, color=(10, 250, 0))
         coords.append([p.x,p.y])

    if name:
        cv2.imshow(name,im); cv2.waitKey(0); cv2.destroyAllWindows()
    return(pd.DataFrame(coords))


def get_landmarks():

    # 68 point face model
    try:
        PREDICTOR_PATH = "../models/shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
    except:
        print('68-point landmark model missing: ../models/shape_predictor_68_face_landmarks.dat. See README for download link.')
        exit()

    detector = dlib.get_frontal_face_detector()

    flat_loc = []
    landmark_coords = {}

    for p in os.listdir('../data/feret100_straight'):

        if p == '.DS_Store': continue

        im = cv2.imread('../data/feret100_straight/' + p)

        face = detector(im,0)[0]

        newRect = dlib.rectangle(int(face.left()), int(face.top()),int(face.right()), int(face.bottom()))

        # Find face landmarks by providing reactangle for each face
        landmarks = predictor(im, newRect)
        
        # Draw facial landmarks
        if p == '00005_0.ppm':
            renderFace(im,landmarks,p)
        coords = renderFace(im,landmarks,None)

        # Flatten coordinates
        flat_loc.append([p,*list(coords[1] * (360) + coords[0])])

        landmark_coords[p] = coords


    all_face = pd.DataFrame(flat_loc)
    pickle.dump(all_face,open('../data/feret100_flat.p','wb'))
    pickle.dump(landmark_coords,open('../data/feret100_coords.p','wb'))
    return(0)

def find_average_face():
    coords = pickle.load(open('../data/feret100_coords.p','rb'))

    keys = list(coords.keys())

    avg_coords = coords[keys[0]]

    for i in range(1,len(keys)):
        avg_coords += coords[keys[i]]

    avg_coords = (avg_coords/len(keys)).astype('int')
    pickle.dump(avg_coords,open('../data/feret100_avg.p','wb'))

if __name__ == "__main__":
    get_landmarks()
    find_average_face()










