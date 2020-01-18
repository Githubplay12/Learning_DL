import numpy as np
import cv2

labels = ['dog', 'cat', 'panda']
# when we give a number to random seed we can the

# Normally learned from the model but here generated randomly
np.random.seed(1)
W = np.random.randn(3, 3072)
b = np.random.randn(3)

orig = cv2.imread('beagle.png')
image = cv2.resize(orig, (32, 32)).flatten()

scores = W.dot(image) + b

# Looping over the scores and labels then displaying them
for label, score in zip(labels, scores):
    print(f'[INFO] {label}: {score:.2f}')


# Draw the label with the highest score on the image
# of our prediction

cv2.putText(orig, f'Label: {labels[np.argmax(scores)]}',
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 255, 0), 2)

# Display the images
cv2.imshow('Image', orig)
cv2.waitKey(0)

