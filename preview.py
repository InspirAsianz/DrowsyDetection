import numpy as np
import pandas as pd
import cv2

source = cv2.VideoCapture('01-10.mov')

while True:
	ret, img = source.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (90, 160))

	print(gray.shape)

	# displaying the video
	cv2.imshow("Live", gray)

	# exiting the loop
	key = cv2.waitKey(1)
	if key == ord("q"):
		break
      
# closing the window
cv2.destroyAllWindows()
source.release()