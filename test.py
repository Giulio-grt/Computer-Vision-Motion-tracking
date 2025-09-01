import cv2, mediapipe
print("cv2:", cv2.__version__, " | mp:", mediapipe.__version__)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # macOS backend
ok, frame = cap.read()
print("opened:", cap.isOpened(), " read:", ok, " shape:", None if not ok else frame.shape)
cap.release()
