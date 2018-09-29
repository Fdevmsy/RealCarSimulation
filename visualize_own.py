import numpy as np
import pandas as pd
import pygame
import glob
import os
# from config import VisualizeConfig

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

def complete_path(path):
    path = os.path.join(path +".jpg")
    return path

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
    print(X)
    #and our steering commands as our output data
    y = data_df["steering_angle"]
    # print(y)
load_data()
# config = VisualizeConfig()
preds = pd.read_csv('Ch2_001/predict.csv', names=['preds_angles'])
print(preds)
dataset = pd.read_csv('Ch2_001/final_example.csv')

# Draw
pygame.init()
size = (640, 480)
pygame.display.set_caption("Data viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
myfont = pygame.font.SysFont("monospace", 15)

# for i in range(1000):
for i in range(len(dataset)):
    # pygame.event.wait(0.1)
    pygame.event.get()
    screen.fill((RED))
    # angle = dataset["steering_angle"].iloc[i] # radians
    angle = preds["preds_angles"].iloc[i] # radians
    true_angle = dataset["steering_angle"].iloc[i] # radians
    # print(type(true_angle))
    print(angle)
    # add image to screen
    # print(os.path.join('Ch2_001/center/', str(dataset["frame_id"].iloc[i])))
    img = pygame.image.load(os.path.join(str(dataset["frame_id"].iloc[i])+".jpg"))
    screen.blit(img, (0, 0))
    
    # add text
    pred_txt = myfont.render("Prediction:" + str(round(angle* 57.2958, 3)), 1, (255,255,0)) # angle in degrees
    true_txt = myfont.render("True angle:" + str(round(true_angle* 57.2958, 3)), 1, (255,255,0)) # angle in degrees
    screen.blit(pred_txt, (10, 440))
    screen.blit(true_txt, (10, 460))

    # draw steering wheel
    radius = 60
    pygame.draw.circle(screen, WHITE, [480, 400], radius, 2) 

    # draw cricle for true angle
    x = radius * np.cos(np.pi/2 + true_angle)
    y = radius * np.sin(np.pi/2 + true_angle)
    pygame.draw.circle(screen, GREEN, [480 + int(x), 400 - int(y)], 18)
    
    # draw cricle for predicted angle
    x = radius * np.cos(np.pi/2 + angle)
    y = radius * np.sin(np.pi/2 + angle)
    pygame.draw.circle(screen, RED, [480 + int(x), 400 - int(y)], 12) 
    

    pygame.display.update()
    pygame.display.flip()
