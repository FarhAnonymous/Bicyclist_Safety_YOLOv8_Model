import cv2
from ultralytics import YOLO
from scipy.spatial import distance

image_path = 'C:/Users/dell/OneDrive/Desktop/Khalisha Civil YOLO/helmet1P.jpg'
image = cv2.imread(image_path)

model = YOLO('best_80.pt') 


# model.predict(source='helmet1.jpg', conf=0.2, save=True)

# Run predictions
results = model.predict('helmet1P.jpg')

# Extract and collecing bounding box coordinates in array
boxesCo = []
for result in results:
    boxes = result.boxes.xyxy  # This gives you [xmin, ymin, xmax, ymax] format
    for box in boxes:
        boxesCo.append(box)

# box1 = boxesCo[0]
# box2 = boxesCo[1]
for i in range(len(boxesCo)):
        for j in range(i + 1, len(boxesCo)):
            box1 = boxesCo[i]
            box2 = boxesCo[j]
# Calculate the centers of each box
            center1 = ((box1[0]+box1[2])/2, (box1[1]+box1[3])/2)
            center2 = ((box2[0]+box2[2])/2, (box2[1]+box2[3])/2)

# Calculate the Euclidean distance between the two centers
dist = distance.euclidean(center1, center2)
# print(f"Distance: {dist}")

cv2.putText(image, f"Distance: {dist}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)
cv2.imwrite('C:/Users/dell/OneDrive/Desktop/Khalisha Civil YOLO/annotated_helmet1P.jpg', image)