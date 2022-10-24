import face_recognition as fr
import cv2 as cv
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv.VideoCapture(0)

adarsh_image = fr.load_image_file("photos/adarsh.png")
adarsh_encoding = fr.face_encodings(adarsh_image)[0]

rishabh_image = fr.load_image_file("photos/rishabh.png")
rishabh_encoding = fr.face_encodings(rishabh_image)[0]

tapase_image = fr.load_image_file("photos/tapase.png")
tapase_encoding = fr.face_encodings(tapase_image)[0]

yash_image = fr.load_image_file("photos/yash.png")
yash_encoding = fr.face_encodings(yash_image)[0]

known_face_encoding = [adarsh_encoding,rishabh_encoding,tapase_encoding,yash_encoding]
known_face_name = ["adarsh","rishabh","tapase","yash"]

students = known_face_name.copy()

face_locations = []
face_encoding = []
face_names = []
s = True

now = datetime.now()
current_date = 	now.strftime("%Y-%m-%d")
f = open(current_date + 'csv','w+',newline = '')
Imwriter = csv.writer(f)

while True:
	_,frame = video_capture.read()
	small_frame = cv.resize(frame,(0,0),fx=0.25,fy=0.25)
	rgb_small_frame = small_frame[:,:,::-1]
	if s:
		face_locations = fr.face_locations(rgb_small_frame)
		face_encodings = fr.face_encodings(rgb_small_frame,face_locations)
		face_names	= []
		for face_encoding in face_encodings:
			matches = fr.compare_faces(known_face_encoding,face_encoding)
			name = ""
			face_distance = fr.face_distance(known_face_encoding,face_encoding)
			best_match_index = np.argmin(face_distance)
			if matches[best_match_index]:
				name = known_face_name[best_match_index]

			face_names.append(name)
			if name in known_face_name:
				if name in students:
					students.remove(name)
					print(students)
					current_time = now.strftime("%H-%M-%S")
					Imwriter.writerow([name,current_time])
	cv.imshow("attendance system",frame)
	if cv.waitKey(1) & 0xFF == ord('q'):
		break	

video_capture.release()
cv.destroyAllWindows()
f.close()