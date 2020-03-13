# python correct.py --east frozen_east_text_detection.pb --image summa.jpg
from smartencoding import smart_unicode_with_replace, smart_unicode
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
from yandex_translate import YandexTranslate
from PIL import Image
import os
import pytesseract
from picamera import PiCamera
from picamera.array import PiRGBArray

def predictions(prob_score, geo):
	(numR, numC) = prob_score.shape[2:4]
	boxes = []
	confidence_val = []

	# loop over rows
	for y in range(0, numR):
		scoresData = prob_score[0, 0, y]
		x0 = geo[0, 0, y]
		x1 = geo[0, 1, y]
		x2 = geo[0, 2, y]
		x3 = geo[0, 3, y]
		anglesData = geo[0, 4, y]

		# loop over the number of columns
		for i in range(0, numC):
			if scoresData[i] < args["min_confidence"]:
				continue

			(offX, offY) = (i * 4.0, y * 4.0)

			# extracting the rotation angle for the prediction and computing the sine and cosine
			angle = anglesData[i]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# using the geo volume to get the dimensions of the bounding box
			h = x0[i] + x2[i]
			w = x1[i] + x3[i]

			# compute start and end for the text pred bbox
			endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
			endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
			startX = int(endX - w)
			startY = int(endY - h)

			boxes.append((startX, startY, endX, endY))
			confidence_val.append(scoresData[i])

	# return bounding boxes and associated confidence_val
	return (boxes, confidence_val)
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

camera = PiCamera()
camera.resolution=(640,480)
camera.framerate=30
rawCapture= PiRGBArray(camera,size=(640,480))
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        imaging = frame.array
	cv2.imshow("Frame", imaging)
	key = cv2.waitKey(1) & 0xFF
	
	rawCapture.truncate(0)

	if key == ord("s"):
		camera.capture('/home/pi/Desktop/opencv-text-detection/summa.jpg')
		break

cv2.destroyAllWindows()

image = cv2.imread(args['image'])

#Saving a original image and shape
orig = image.copy()
(origH, origW) = image.shape[:2]

# set the new height and width to default 320 by using args #dictionary.  
(newW, newH) = (args["width"], args["height"])

#Calculate the ratio between original and new image for both height and weight. 
#This ratio will be used to translate bounding box location on the original image. 
rW = origW / float(newW)
rH = origH / float(newH)

# resize the original image to new dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# construct a blob from the image to forward pass it to EAST model

layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]
net = cv2.dnn.readNet(args["east"])
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)

(scores, geometry) = net.forward(layerNames)
(boxes, confidence_val) = predictions(scores, geometry)
boxes = non_max_suppression(np.array(boxes), probs=confidence_val)
results = []

# loop over the bounding boxes to find the coordinate of bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	#extract the region of interest
	r = orig[startY:endY, startX:endX]

	#configuration setting to convert image to string.  
	configuration = ("-l eng --oem 1 --psm 8")
        ##This will recognize the text from the image of bounding box
	text = pytesseract.image_to_string(r, config=configuration)

	# append bbox coordinate and associated text to the list of results 
	results.append(((startX, startY, endX, endY), text))

orig_image = orig.copy()

# Moving over the results and display on the image
for ((start_X, start_Y, end_X, end_Y), text) in results:
	# display the text detected by Tesseract
        text=(smart_unicode(text))
        translate = YandexTranslate('trnsl.1.1.20200222T042110Z.10474b882cdf0dd2.08d7b3579f016b9d507aaa6e6a7ec116a445d121')
        res=translate.translate(text,'fr-en')
        result=str(res)
        first=result.rfind("[")
        last=result.rfind("]")
        text=(result[first+3:last-1])
	# Displaying text
	text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
	cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
		(0, 255,0), 2)
	cv2.putText(orig_image, text, (start_X, start_Y),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0), 2)


cv2.imshow("output",orig_image)
text = pytesseract.image_to_string(image)
translate = YandexTranslate('trnsl.1.1.20200224T013140Z.ed35248d3e84bd29.ed27af3d6b7b4b4e2548774c7c481accbb1b143b')
res=(translate.translate(text, 'en'))
result=str(res)
first=result.rfind("[")
last=result.rfind("]")
pure=(result[first+3:last-1])
print(pure)
os.system('echo "' +pure+'"| festival --tts')

cv2.waitKey(0)

os.remove("summa.jpg")
