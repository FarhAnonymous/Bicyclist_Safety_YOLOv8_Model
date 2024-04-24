import cv2
import random
from ultralytics import YOLO
from scipy.spatial import distance

#video_path = "C:/Users/dell/OneDrive/Desktop/Khalisha Civil YOLO/bicycle2.mp4"
video_path = "./videos/bicycle2.mp4"
capture = cv2.VideoCapture(video_path)

model = YOLO("./trained_models/best_80.pt")

while True:
    # Read a frame from the video
    ret, frame = capture.read()

    if not ret:
        break

    # Run predictions
    results = model.predict(frame)

    # Extract and collecing bounding box coordinates in array
    boxesCo = []
    colors = []
    for result in results:
        boxes = result.boxes.xyxy  # This gives you [xmin, ymin, xmax, ymax] format
        for box in boxes:
            boxesCo.append(box)
            # Generate a random color for the bounding box
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            colors.append(color)
            # Draw the bounding box
            # x1, y1, x2, y2 = [int(coord) for coord in box]
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for i in range(len(boxesCo)):
        for j in range(i + 1, len(boxesCo)):
            box1 = boxesCo[i]
            box2 = boxesCo[j]
            # Draw the bounding boxes
            x1, y1, x2, y2 = [int(coord) for coord in box1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[i], 2)
            x1, y1, x2, y2 = [int(coord) for coord in box2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[j], 2)

            # Calculate the centers of each box
            center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
            center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)

            # Calculate the Euclidean distance between the two centers
            dist = distance.euclidean(center1, center2)

            cv2.putText(
                frame,
                f"Distance: {dist:.2f}",
                (50, 50 + 20 * (i * len(boxesCo) + j)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 100, 100),
                2,
            )
            # Draw a line between the centers of the bounding boxes
            cv2.line(
                frame,
                (int(center1[0]), int(center1[1])),
                (int(center2[0]), int(center2[1])),
                (0, 0, 255),
                2,
            )

    # Display the processed frame
    cv2.imshow("Video", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close the window
capture.release()
cv2.destroyAllWindows()
