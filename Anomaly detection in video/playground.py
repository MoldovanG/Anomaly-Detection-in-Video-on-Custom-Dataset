import os
# ROOT_DIR = os.path.abspath("../")
# # Directory of images to run detection on
# VIDEO_DIR = os.path.join(ROOT_DIR, "Avenue Dataset/training_videos")
from DataSetTrainer import DataSetTrainer

VIDEO_DIR = 'C:\\Users\\georg\\PycharmProjects\\Licenta\\Avenue Dataset'
dataset_trainer = DataSetTrainer(VIDEO_DIR)