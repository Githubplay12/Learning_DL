import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        # If the preprocessors are None, initiliaze them as an
        # empty list
        if not self.preprocessors:
            self.preprocessors = []

    # Initialize the list of features and labels
    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []

        # The verbose level is to get more informations
        # For example how much images went through the processor
        # Loops over the input images
        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)


            # Used to extract the class label, assuming that it
            # looks like this :
            # /path/to/dataset/{class}/{image}.jpg
            # Dans ce cas ghay yjbed {class}
            label = imagePath.split(os.path.sep)[-2]
            if image is None:
                if label == 'Cat':
                    os.rename(imagePath, os.path.join(r'C:\Users\CARBON\Desktop\Learning\DL For Computer Vision\datasets\Failedpics\Failedcatpics', os.path.basename(imagePath)))
                elif label == 'Dog':
                    os.rename(imagePath, os.path.join(r'C:\Users\CARBON\Desktop\Learning\DL For Computer Vision\datasets\Failedpics\Faileddogpics', os.path.basename(imagePath)))
                else:
                    os.rename(imagePath, os.path.join(r'C:\Users\CARBON\Desktop\Learning\DL For Computer Vision\datasets\Failedpics\Failedpandapics', os.path.basename(imagePath)))
                continue

            if self.preprocessors:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print(f'[INFO] processed {i+1} / {len(imagePaths)}')

        return (np.array(data), np.array(labels))



