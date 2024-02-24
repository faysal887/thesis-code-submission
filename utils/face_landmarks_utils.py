# !wget -nd https://github.com/JeffTrain/selfie/raw/master/shape_predictor_68_face_landmarks.dat
import sys
import utils.helper_functions as helper_functions
from collections import Counter
import cv2, os, sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import dlib,cv2
from PIL import Image
import pdb


# This below mehtod will draw all those points which are from 0 to 67 on face one by one.
def drawPoints(image, faceLandmarks, startpoint, endpoint, isClosed=False):
	points = []
	for i in range(startpoint, endpoint+1):
		point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
		points.append(point)

	points = np.array(points, dtype=np.int32)
	cv2.polylines(image, [points], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)
	return points

def drawPointsCustom(image, faceLandmarks, custom_points, isClosed=False):
	points = []
	for i in custom_points:
		point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
		points.append(point)

	points = np.array(points, dtype=np.int32)
	cv2.polylines(image, [points], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)
	return points

def facePoints(image, faceLandmarks):
	points_dict={}
	assert(faceLandmarks.num_parts == 68)
	points_dict['jawline'] = drawPoints(image, faceLandmarks, 0, 16,  False)         # Jaw line
	points_dict['left_eyebrow'] = drawPoints(image, faceLandmarks, 17, 21, True)          # Left eyebrow
	points_dict['right_eyebrow'] = drawPoints(image, faceLandmarks, 22, 26, True)          # Right eyebrow
	# points_dict['nose'] = drawPointsNose(image, faceLandmarks, 27, 35, True)    # Nose 
	points_dict['nose'] = drawPointsCustom(image, faceLandmarks, [27, 31,32,33,34,35,], True)    # Nose 
	# drawPoints(image, faceLandmarks, 30, 35, True)    # Lower nose
	points_dict['left_eye'] = drawPoints(image, faceLandmarks, 36, 41, True)    # Left eye
	points_dict['right_eye'] = drawPoints(image, faceLandmarks, 42, 47, True)    # Right Eye
	points_dict['lips'] = drawPoints(image, faceLandmarks, 48, 59, True)    # Outer lip
	# points_dict['inner_lip'] = drawPoints(image, faceLandmarks, 60, 67, True)    # Inner lip
	points_dict['left_cheek'] = drawPointsCustom(image, faceLandmarks, [0,1,2,3,4,5,31], True)    # left cheek
	points_dict['right_cheek'] = drawPointsCustom(image, faceLandmarks, [11,12,13,14,15,16,35,], True)    # right cheek
	points_dict['chin'] = drawPointsCustom(image, faceLandmarks, [5,6,7,8,9,10,11], True)    # right cheek

	return points_dict

def facePoints2(image, faceLandmarks, color=(0, 255, 0), radius=4):
	for p in faceLandmarks.parts():
		cv2.circle(im, (p.x, p.y), radius, color, -1)

def writeFaceLandmarksToLocalFile(faceLandmarks, fileName):
	with open(fileName, 'w') as f:
		for p in faceLandmarks.parts():
			f.write("%s %s\n" %(int(p.x),int(p.y)))
	f.close()

def detect_landmarks_dlib(img_path, frontalFaceDetector, faceLandmarkDetector, display=True, save=True, resize=True, save_path=None):
	# now from the dlib we are extracting the method get_frontal_face_detector()
	# and assign that object result to frontalFaceDetector to detect face from the image with 
	# the help of the 68_face_landmarks.dat model

	# frontalFaceDetector = dlib.get_frontal_face_detector()



	# Now we are reading image using openCV
	img= cv2.imread(img_path)

	if resize: img = cv2.resize(img, (256,256)) 

	imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# Now this line will try to detect all faces in an image either 1 or 2 or more faces
	allFaces = frontalFaceDetector(imageRGB, 0)
	# List to store landmarks of all detected faces
	allFacesLandmark = []

	if len(allFaces)==0:
		print('no face detected in ',  img_path)
		return None, None

	else:
		# Below loop we will use to detect all faces one by one and apply landmarks on them
		for k in range(0, len(allFaces)):
			# dlib rectangle class will detecting face so that landmark can apply inside of that area
			faceRectangleDlib = dlib.rectangle(int(allFaces[k].rect.left()),int(allFaces[k].rect.top()),
				int(allFaces[k].rect.right()),int(allFaces[k].rect.bottom()))

			# Now we are running loop on every detected face and putting landmark on that with the help of faceLandmarkDetector
			detectedLandmarks = faceLandmarkDetector(imageRGB, faceRectangleDlib)

			# count number of landmarks we actually detected on image
			# if k==0:
				# print("Total number of face landmarks detected ",len(detectedLandmarks.parts()))

			# Svaing the landmark one by one to the output folder
			allFacesLandmark.append(detectedLandmarks)

			# Now finally we drawing landmarks on face
			points_dict=facePoints(img, detectedLandmarks)

			# fileName = faceLandmarksOuput +"_"+ str(k)+ ".txt"

			# Write landmarks to disk
			if save: 
				# writeFaceLandmarksToLocalFile(detectedLandmarks, fileName)
				#Name of the output file
				print("Saving output image to", save_path)
				cv2.imwrite(save_path, img)

			if display:
				plt.imshow(img)
				plt.show()

			# Pause screen to wait key from user to see result
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

		return points_dict, detectedLandmarks

# Define a function to check the importance level of a given RGB color
def get_importance_level(rgb):
	# ORIGINAL
	# custom_colormap = np.array([
	# 						[255, 0, 0],  # Red
	# 						[255, 128, 0], # Orange
	# 						[255, 255, 0], # Yellow
	# 						[0, 128, 255], # Light Blue
	# 						[0, 0, 255]    # Blue
	# 						],dtype=np.uint8)  

	# UPDATED FROM BING
	# custom_colormap = np.array([
    #                     [128, 0, 0],  # Red
    #                     [255, 0, 0], # Orange
    #                     [255, 255, 0], # Yellow
    #                     [0, 255, 0], # Light Blue
    #                     [0, 0, 128]    # Blue
    #                     ],dtype=np.uint8) 

	# CUSTOM
	custom_colormap = np.array([
                        [255, 70, 0],  # Red
                        [255, 245, 0], # Orange
                        [96,255,165], # Yellow
                        [0, 185, 255], # Light Blue
                        [0, 25, 255]    # Blue
                        ],dtype=np.uint8) 

	# Calculate the Euclidean distance between the pixel color and the custom colormap
	color_distances = np.linalg.norm(custom_colormap - rgb, axis=1)

	# Determine the index of the nearest color in the custom colormap
	importance_level = np.argmin(color_distances)

	return importance_level+1

def get_dominant_importance_level(mapimage, points, display=True, crop=True):
	try:
		img = cv2.imread(mapimage)
		img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# img=Image.open(mapimage)
		# img=np.array(img)

		points = np.array(points)
		mask = np.zeros_like(img)
		cv2.fillPoly(mask, [points], (255, 255, 255))
		result = cv2.bitwise_and(img, mask)
		points[points < 0] = 0
		if crop:
			x, y, w, h = cv2.boundingRect(points)
			result = result[y:y+h, x:x+w]

		if display: 
			plt.imshow(result)
			plt.show()

		imp=[]
		for i in range (result.shape [0]):
			for j in range (result.shape [1]): 
				r,g,b = result.item (i, j,0),result.item (i, j,1),result.item (i, j,2) 
				if (r,g,b)==(0,0,0): continue
				imp.append(get_importance_level((r,g,b)))


		my_dict=dict(Counter(imp))
		return max(my_dict, key=my_dict.get)
	except Exception as e:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print('Error in get_dominant_importance_level: ',display,e, exc_type, fname, exc_tb.tb_lineno)
