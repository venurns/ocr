import cv2 
import pytesseract
from picamera.array import PiRGBArray
from picamera import PiCamera
from yandex_translate import YandexTranslate
import pyttsx3
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30

rawCapture = PiRGBArray(camera, size=(640, 480))

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	image = frame.array
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF
	
	rawCapture.truncate(0)

	if key == ord("s"):
		text = pytesseract.image_to_string(image)
		translate = YandexTranslate('trnsl.1.1.20200222T042110Z.10474b882cdf0dd2.08d7b3579f016b9d507aaa6e6a7ec116a445d121')
                res=(translate.translate(text, 'en'))
                result=str(res)
                first=result.rfind("[")
                last=result.rfind("]")
                pure=(result[first+2:last-1])
                pure=(result[first+2:last-1])
                print(pure)
                engine = pyttsx3.init()
                engine.say(pure)
                engine.setProperty('rate',10)
                engine.setProperty('volume', 0.9)
                engine.runAndWait()
		cv2.imshow("Frame", image)
		cv2.waitKey(0)
		break

cv2.destroyAllWindows()
