from AutoEncoderModel import AutoEncoderModel
from DataSetTrainer_Stage2 import DataSetTrainer_Stage2
from VideoProcessor_Stage2 import VideoProcessor_Stage2

autoencoder_images = AutoEncoderModel([],"raw_object_autoencoder")
autoencoder_gradients = AutoEncoderModel([],"gradient_object_autoencoder")
# video_path = 'C:\\Users\\georg\\PycharmProjects\\Licenta\\Avenue Dataset\\training_videos\\12.avi'
# videoprocessor = VideoProcessor_Stage2(video_path,autoencoder_images,autoencoder_gradients)
# print(videoprocessor.feature_vectors.shape)

dataset_directory_path = "C:\\Users\\georg\\PycharmProjects\\Licenta\\Avenue Dataset\\training_videos_small"
final_dataset_trainer = DataSetTrainer_Stage2(dataset_directory_path,autoencoder_images,autoencoder_gradients)

print(final_dataset_trainer.feature_vectors.shape)