import keras
import tensorflow
import cv2
import imageai
from imageai.Prediction import ImagePrediction
import os

execution_path=os.getcwd()
prediction = ImagePrediction()
#Choose model type from https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Prediction/README.md
prediction.setModelTypeAsSqueezeNet()
#set a path to directory
prediction.setModelPath(os.path.join(execution_path, "drive/My Drive/Colab Notebooks/imagePredictionProject/squeezenet_weights_tf_dim_ordering_tf_kernels.h5"))
prediction.loadModel()

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "drive/My Drive/Colab Notebooks/imagePredictionProject/cruiseShip.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)

