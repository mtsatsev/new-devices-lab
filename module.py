import numpy as np
import cv2

class Device():
    def __init__(self):
        self.device = 0
        self.capture = cv2.VideoCapture(self.device)

        if not(self.capture.isOpened()):
            print("I failed")
            self.capture.open(self.device)
        if self.capture.isOpened():
            print("HEYLYA")

        while True:
            ret,frame = self.capture.read()

            cv2.imshow("str", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.capture.release()
        cv2.destroyAllWindows()


def main():
    device = Device()


main()
