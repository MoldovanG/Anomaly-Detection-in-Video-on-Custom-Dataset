from code.training.stage2_clustering_and_svms.DataSetTrainer_Stage2 import DataSetTrainer_Stage2
from code.training.utils.AutoEncoderModel import AutoEncoderModel

autoencoder_images = AutoEncoderModel([],"raw_object_autoencoder")
autoencoder_gradients = AutoEncoderModel([],"gradient_object_autoencoder")

dataset_directory_path = "/home/george/Licenta/Anomaly detection in video/Avenue Dataset/training_videos"
final_dataset_trainer = DataSetTrainer_Stage2(dataset_directory_path,autoencoder_images,autoencoder_gradients)
