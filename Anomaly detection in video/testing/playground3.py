from testing.ModelEvaluator import ModelEvaluator
from training.AutoEncoderModel import AutoEncoderModel
from training.DataSetTrainer_Stage2 import DataSetTrainer_Stage2

autoencoder_images = AutoEncoderModel([],"raw_object_autoencoder")
autoencoder_gradients = AutoEncoderModel([],"gradient_object_autoencoder")

dataset_directory_path = "/Avenue Dataset/training_videos_small"
final_dataset_trainer = DataSetTrainer_Stage2(dataset_directory_path,autoencoder_images,autoencoder_gradients)

dataset_directory_path = "/home/george/Licenta/Anomaly detection in video/Avenue Dataset/testing_videos"
modelEvaluator = ModelEvaluator(final_dataset_trainer,dataset_directory_path)
modelEvaluator.evaluate_dataset()