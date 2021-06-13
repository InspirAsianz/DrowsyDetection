import cv2
import joblib
import numpy as np
import pandas as pd
import time

classifier = joblib.load("rf_model.joblib")

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

video_capture = cv2.VideoCapture(0)

cur_state = 0
zeroes = 0
fives = 0
tens = 0
last_update = time.time()

while True:
	ret, frame = video_capture.read()

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	inp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	inp = inp[:, 219:421]
	resized = cv2.resize(inp, (90, 160)).flatten()
	s = scaler.fit_transform(resized.reshape(-1, 1))
	s = s.reshape(1, -1)

	out = classifier.predict(s)
	if out[0] == 0:
		zeroes+=1
	elif out[0] == 5:
		fives+=1
	elif out[0] == 10:
		tens+=1

	if time.time() - last_update > 0.5:
		if zeroes > fives:
			if zeroes > tens:
				cur_state = 0
			else:
				cur_state = 10
		else:
			if fives > tens:
				cur_state = 5
			else:
				cur_state = 10
		zeroes, fives, tens = 0, 0, 0
		last_update = time.time()

	print(int(out[0]))

	cv2.putText(frame, str(cur_state), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
	cv2.rectangle(frame, (421, 360), (219, 0), (0, 255, 0), 2)
	cv2.imshow('Video', frame)

video_capture.release()
cv2.destroyAllWindows()