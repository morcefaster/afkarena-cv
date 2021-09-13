import os
import numpy as np
import imutils
import glob
import cv2
import cv2 as cv
from enum import Enum
from plot_helper import plot_helper as plt
import matplotlib.pyplot as plt2


class img_helper:

	@staticmethod 
	def find_scaled_path(template_path, img_path, match_bounds = [[0,0],[0,0]], min_scale = 0.1, max_scale = 2.0, steps = 40, show_progress = False):
		cv_template = cv.imread(template_path)
		cv_image = cv.imread(img_path)
		return img_helper.find_scaled(cv_template, cv_image, match_bounds, min_scale, max_scale, steps, show_progress)
	
	@staticmethod	
	def find_scaled(cv_template, cv_image, match_bounds, min_scale = 0.1, max_scale = 2.0, steps = 40, show_progress = False):
		template = cv_template
		template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
		template = cv.Canny(template, 50, 200)
		plt.show_bgr(template)
		(tH, tW) = template.shape[:2]		
		image = cv_image		
		gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		found = None
		# loop over the scales of the image
		for scale in np.linspace(min_scale, max_scale, steps)[::-1]:
			# resize the image according to the scale, and keep track
			# of the ratio of the resizing
			resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
			r = gray.shape[1] / float(resized.shape[1])
			# if the resized image is smaller than the template, then break
			# from the loop
			if resized.shape[0] < tH or resized.shape[1] < tW:
				break
			edged = cv.Canny(resized, 50, 200)			
			result = cv.matchTemplate(edged, template, cv.TM_CCOEFF)
			(_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
			# check to see if the iteration should be visualized
			if show_progress:
				# draw a bounding box around the detected region
				clone = np.dstack([edged, edged, edged])
				cv.rectangle(clone, (maxLoc[0], maxLoc[1]),
					(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
				cv.imshow("Visualize", clone)
				cv.waitKey(0)
			# if we have found a new maximum correlation value, then update
			# the bookkeeping variable			
			if found is None or maxVal > found[0]:
				print(maxLoc)
				print(r)
				maxMatch = imutils.resize(image, width = int(image.shape[1] * scale))[maxLoc[1]+match_bounds[0][1]:maxLoc[1]+tH+match_bounds[1][1],maxLoc[0]+match_bounds[0][0]:maxLoc[0]+tW+match_bounds[1][0]]				
				found = (maxVal, maxLoc, r, maxMatch)
		# unpack the bookkeeping variable and compute the (x, y) coordinates
		# of the bounding box based on the resized ratio
		(_, maxLoc, r, match) = found
		(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
		(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
		print(found)
		# draw a bounding box around the detected result and display the image
		cv.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)		
		plt.show_bgr(image)
		return startX, startY, endX, endY, r, match
	

	@staticmethod
	def find(cv_template, cv_image):
		template = cv.cvtColor(cv_template, cv.COLOR_BGR2GRAY) 
		template = cv.Canny(cv_template, 50, 200) 		
		edged = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
		edged = cv.Canny(cv_image, 50, 200)		
		result = cv.matchTemplate(edged, template, cv.TM_CCOEFF)
		(_, maxVal, _, maxLoc)= cv.minMaxLoc(result)
		return maxVal, maxLoc

	@staticmethod
	def find_best_match(dir, cv_image):		
		found = None
		for filename in os.listdir(dir):						 
			template_path = os.path.join(dir, filename)
			template = cv.imread(template_path)			
			(maxVal, maxLoc) = img_helper.find(template, cv_image)
			print(f'{filename} : {maxVal}')
			if found == None or maxVal>found[0]:
				found = (maxVal, maxLoc, filename)
		return filename, maxVal

	def find_first_match(dir, cv_image, threshold):
		for filename in os.listdir(dir):						 
			template_path = os.path.join(dir, filename)
			template = cv.imread(template_path)
			(maxVal, maxLoc) = img_helper.find(template, cv_image)
			if (maxVal > threshold):
				return filename
		return None

	def find_bf(cv_template, cv_image):
        # Initiate SIFT detector
		sift = cv.SIFT_create()
		# find the keypoints and descriptors with SIFT
		kp1, des1 = sift.detectAndCompute(cv_template, None)
		kp2, des2 = sift.detectAndCompute(cv_image, None)
		# BFMatcher with default params
		bf = cv.BFMatcher()
		bf = cv.BFMatcher()
		matches = bf.knnMatch(des1,des2,k=2)
		# Apply ratio test
		good = []
		for m,n in matches:
			if m.distance < 0.95*n.distance:
				good.append([m])
		# cv.drawMatchesKnn expects list of lists as matches.
		img3 = cv.drawMatchesKnn(cv_template,kp1,cv_image,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
		plt2.imshow(img3),plt2.show()
		return len(good)


		#matches = bf.knnMatch(des1, des2, k=2)
		## Apply ratio test
		#good = []
		#for m,n in matches:
		#	if m.distance < 0.75*n.distance:
		#		good.append([m])
		## cv.drawMatchesKnn expects list of lists as matches.
		#img3 = cv.drawMatchesKnn(cv_template,kp1,cv_image,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
		#plt2.imshow(img3),plt2.show()
		return None

	def find_best_match_bf(dir, cv_image):		
		found = None
		for filename in os.listdir(dir):						 
			template_path = os.path.join(dir, filename)
			template = cv.imread(template_path)			
			maxVal = 1
			matches = img_helper.find_bf(template, cv_image)
			if found == None or matches > found[0]:
				found = (matches, filename)
			
		return found

	@staticmethod
	def match_with_alpha(cv_template, cv_image, cv_template_alpha):
		img = cv_image		
		templ = cv_template
		(tH, tW) = templ.shape[:2]
		channels = cv.split(cv_template_alpha)
		#extract "transparency" channel from image
		alpha_channel = np.array(channels[3]) 
		#generate mask image, all black dots will be ignored during matching
		mask = cv.merge([alpha_channel,alpha_channel,alpha_channel])
		cv.imshow("Mask", mask)

		result = cv.matchTemplate(img, templ, cv.TM_CCORR_NORMED, None, mask)
		#cv.imshow("Matching with mask", result)
		min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
		print('Highest correlation WITH mask', max_val)
		(startX, startY) = (int(max_loc[0]), int(max_loc[1]))
		(endX, endY) = (int((max_loc[0] + tW)), int((max_loc[1] + tH)))
		cv.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)		
		plt.show_bgr(img)
		result = cv.matchTemplate(img, templ, cv.TM_CCORR_NORMED)
		#cv.imshow("Matching without mask", result)
		min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
		print('Highest correlation without mask', max_val)
		(startX, startY) = (int(max_loc[0]), int(max_loc[1]))
		(endX, endY) = (int((max_loc[0] + tW)), int((max_loc[1] + tH)))
		cv.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)		
		plt.show_bgr(img)

	@staticmethod
	def match_with_mask(cv_template, cv_image):
		img = cv_image		
		templ = cv_template	

		(tH, tW) = templ.shape[:2]
		result = cv.matchTemplate(img, templ, cv.TM_CCORR_NORMED, None, templ)		
		min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
		print('Highest correlation WITH mask', max_val)
		(startX, startY) = (int(max_loc[0] ), int(max_loc[1] ))
		(endX, endY) = (int((max_loc[0] + tW) ), int((max_loc[1] + tH) ))
		#cv.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)		
		#plt.show_bgr(img)

		#result = cv.matchTemplate(img, templ, cv.TM_CCORR_NORMED)		
		#min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
		#(startX, startY) = (int(max_loc[0] ), int(max_loc[1] ))
		#(endX, endY) = (int((max_loc[0] + tW) ), int((max_loc[1] + tH) ))
		#print('Highest correlation without mask', max_val)
		#cv.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)		
		#plt.show_bgr(img)
