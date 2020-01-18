from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledataloader import SimpleDatasetLoader
from imutils import paths
import argparse

# Construct the argument parser and parse the args
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='Path to input dataset')
ap.add_argument('-k', '--neighbors', type=int, default=1,
                help='# of neighbors for classification')
ap.add_argument('-j', '--jobs', type=int, default=-1,
                help='# of jobs for k-NN distance '
                     '-1 uses all available cores')
args = vars(ap.parse_args())

# grab the list of image we'll be describing
print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset']))

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
# flatten the images from 3D representation to single pixel list
# Transformation 32 X 32 X 3 to array with (3000, 3072) shape
# Needed to apply the k-NN algorithm
data = data.reshape(data.shape[0], 3072)

# Show informations about images's memory consumption
print(f'[INFO] features matrix : {(data.nbytes / (1024 * 1000.0)):.1f} MB')

# Building the training and testing split here :
# Encode the labels as integers because many ML models assume so
le = LabelEncoder()
labels = le.fit_transform(labels)

# Partition the data into training and testing split
# (a auteur de 75 / 25 %)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25,
                                                  random_state=42)

# Finally we train then evaluate the classifier here
print('[INFO] evaluating the k_NN classifier...')
model = KNeighborsClassifier(n_neighbors=args['neighbors'],
                             n_jobs=args['jobs'])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),
                            target_names=le.classes_))
















