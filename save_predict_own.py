# import rospy
import argparse
import json
from scipy import misc
from keras.optimizers import SGD
from keras.models import model_from_json, load_model
import utils
import numpy as np
# import thread
import tensorflow as tf
# from geometry_msgs.msg import Twist
import time
# from premodel import ChauffeurModel
from keras import backend as K
from keras import metrics
from collections import deque
import csv
import os
from math import pi
import pandas as pd
import matplotlib.image as mpimg

print("tf version: ", tf.__version__)

def load_data():
    """
    Load training data and split it into training and validation set
    """
    #reads CSV file into a single dataframe variable
    data_df = pd.read_csv('Ch2_001/final_example.csv')

    #yay dataframes, we can select rows and columns by their names
    #we'll store the camera images as our input data

    X = data_df['frame_id'].apply(complete_path) # images 
    # iamge paths will be look like "Ch2_001/center/1412421421421.jpg "
    # print(X)
    #and our steering commands as our output data
    y = data_df["steering_angle"].astype('float64')

    return X, y

def process(model, img):
    if img is not None:
        # print(img)
        # print(img.shape)
        # img = misc.imresize(img[:, :, :], (66, 200, 3))
        # print(img.shape)
        img = utils.rgb2yuv(img)
        # print(img.shape)
        img = np.array([img])
        # print(img.shape)
        # print('\n\n')
        # steering_angle = model.predict(img[None, :, :, :])[0][0]
        steering_angle = float(model.predict(img, batch_size=1))
        print(steering_angle)
        # pub_steering(steering_angle)
        return(steering_angle)

def complete_path(path):
    path = os.path.join(path +".jpg")
    return path

def get_model(model_file):

    with open(model_file, 'r') as jfile:

        model = model_from_json(jfile.read())

    model.compile("adam", "mse")

    weights_file = model_file.replace('json', 'h5')
    model.load_weights(weights_file)
    # model = load_model(weights_file)
    
    # return model
    # graph = tf.get_default_graph()
    return model

# def make_predictor(cnn_json_path, cnn_weights_path,lstm_json_path,lstm_weights_path):
#     K.set_learning_phase(0)
#     model = ChauffeurModel(
#         cnn_json_path,
#         cnn_weights_path,
#         lstm_json_path,
#         lstm_weights_path)
#     return model.make_stateful_predictor()


# def process(predictor, img):
    # steering_angle = predictor(img)
    # print(steering_angle)
    # return(steering_angle)



if __name__ == '__main__':
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Model Runner')
    parser.add_argument('model', type=str, help='Path to model definition json. \
                        Model weights should be on the same path.')
    
    args = parser.parse_args()
    model = get_model(args.model)
    print("Model loaded")
    
    # model = make_predictor(cnn_json_path,cnn_weights_path,lstm_json_path,lstm_weights_path)
    # node = SteeringNode()

    # rospy.Timer(rospy.Duration(1), process(model, node.img))

    # rospy.spin()
    X, true_y = load_data()
    predicted_list = list()
    for image in X:
        img = mpimg.imread(image)
        pre_angle = process(model, img)
        predicted_list.append(pre_angle)
        text_file = open("Ch2_001/predict.csv", "w")
    for row in predicted_list:
        
        text_file.write(str(row) +'\n')
    text_file.close()
    print("predicted angles saved!")
    
