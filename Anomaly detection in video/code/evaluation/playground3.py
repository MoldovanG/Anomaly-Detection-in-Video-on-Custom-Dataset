from code.testing.ModelEvaluator import ModelEvaluator
from code.training.AutoEncoderModel import AutoEncoderModel
from code.training.DataSetTrainer_Stage2 import DataSetTrainer_Stage2

autoencoder_images = AutoEncoderModel([],"raw_object_autoencoder")
autoencoder_gradients = AutoEncoderModel([],"gradient_object_autoencoder")

dataset_directory_path = "/Avenue Dataset/training_videos_small"
final_dataset_trainer = DataSetTrainer_Stage2(dataset_directory_path,autoencoder_images,autoencoder_gradients)

dataset_directory_path = "/Avenue Dataset/testing_videos_small"
ground_truth_directory = "/home/george/Licenta/Anomaly detection in video/Avenue Dataset/testing_label_mask"
modelEvaluator = ModelEvaluator(final_dataset_trainer,dataset_directory_path,ground_truth_directory)
modelEvaluator.evaluate_dataset()
modelEvaluator.show_average_precision()