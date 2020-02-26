from AutoEncoderModel import AutoEncoderModel
from DataSetTrainer_Stage2 import DataSetTrainer_Stage2
from VideoProcessor_Stage2 import VideoProcessor_Stage2

autoencoder_images = AutoEncoderModel([],"raw_object_autoencoder")
autoencoder_gradients = AutoEncoderModel([],"gradient_object_autoencoder")

dataset_directory_path = "/home/george/Licenta/Anomaly detection in video/Avenue Dataset/training_videos_small"
final_dataset_trainer = DataSetTrainer_Stage2(dataset_directory_path,autoencoder_images,autoencoder_gradients)

