import os

import numpy as np
from sklearn import svm

from AutoEncoderModel import AutoEncoderModel
from VideoProcessor_Stage2 import VideoProcessor_Stage2
from sklearn.cluster import KMeans

class DataSetTrainer_Stage2:
    """
    Class used for genereting the svm models neede for the inferrence phase.
    It requires the dataset directory path, the pretrained-autoencoder
    and based on that, it will cluster the data, and train the 1 vs all svm models.
    Those models will be later used in the inferrrence phase for predicting the anomalies.
    """
    def __init__(self,dataset_directory_path,autoencoder_images : AutoEncoderModel, autoencoder_gradients : AutoEncoderModel):
        self.num_clusters = 10
        self.dataset_directory_path = dataset_directory_path
        self.autoencoder_images = autoencoder_images
        self.autoencoder_gradients = autoencoder_gradients
        self.feature_vectors = self.__get_dataset_feature_vectors()
        self.cluster_labels = self.__cluster_data(self.feature_vectors,self.num_clusters)
        self.models = self.__generate_models()

    def __cluster_data(self, feature_vectors, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(feature_vectors)
        return kmeans.labels_

    def __get_dataset_feature_vectors(self):
        total_feature_vectors = np.resize([],(0,3072))
        for video_name in os.listdir(self.dataset_directory_path):
            video_path = os.path.join(self.dataset_directory_path,video_name)
            videoprocessor = VideoProcessor_Stage2(video_path, self.autoencoder_images, self.autoencoder_gradients)
            feature_vectors = videoprocessor.get_feature_vectors()
            total_feature_vectors = np.concatenate((total_feature_vectors,feature_vectors),axis = 0)
        return total_feature_vectors

    def __generate_models(self):
        models = []
        for cluster in range(self.num_clusters):
            labels = []
            for i in range(self.feature_vectors.shape[0]):
                if self.cluster_labels[i] == cluster:
                    labels.append(1)
                else:
                    labels.append(0)
            model = svm.SVC(kernel='linear', C = 1.0)
            model.fit(self.feature_vectors,labels)
            models.append(model)
        return models

    def get_inference_score(self,feature_vector):
        scores = [model.predict(feature_vector) for model in self.models]
        return max(scores)