import cv2

animal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

video_path = 'video.mp4'  #
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   
    animals = animal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    for (x, y, w, h) in animals:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Animal Detection', frame)
 # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
     
      break

# Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
