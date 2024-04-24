from scipy.spatial import distance

# Assuming box1 = [x1, y1, x2, y2] for the helmet and box2 for the vehicle
box1 = [259.2914, 264.0425, 621.9386, 562.8878]
box2 = [663.4927, 269.2556, 880.9656, 501.2211]
# Calculate the centers of each box
center1 = ((box1[0]+box1[2])/2, (box1[1]+box1[3])/2)
center2 = ((box2[0]+box2[2])/2, (box2[1]+box2[3])/2)

# Calculate the Euclidean distance between the two centers
dist = distance.euclidean(center1, center2)
print(f"Distance: {dist}")