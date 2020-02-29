import os
import pickle
import numpy as np
from sklearn import svm

from training.AutoEncoderModel import AutoEncoderModel
from training.VideoProcessor_Stage2 import VideoProcessor_Stage2
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
        self.checkpoint_models = "/home/george/Licenta/Anomaly detection in video/Avenue Dataset/checkpoints/pretrained_svm"
        self.dataset_directory_path = dataset_directory_path
        self.autoencoder_images = autoencoder_images
        self.autoencoder_gradients = autoencoder_gradients
        if not os.path.exists(self.checkpoint_models):
            self.__feature_vectors = self.__get_dataset_feature_vectors()
            self.__cluster_labels = self.__cluster_data(self.__feature_vectors, self.num_clusters)
            self.models = self.__generate_models()
        else:
            self.models = self.__load_models()

    def __cluster_data(self, feature_vectors, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(feature_vectors)
        return kmeans.labels_

    def __get_dataset_feature_vectors(self):
        total_feature_vectors = np.resize([],(0,3072))
        for video_name in os.listdir(self.dataset_directory_path):
            print(video_name)
            video_path = os.path.join(self.dataset_directory_path,video_name)
            videoprocessor = VideoProcessor_Stage2(video_path, self.autoencoder_images, self.autoencoder_gradients)
            feature_vectors = videoprocessor.get_feature_vectors()
            total_feature_vectors = np.concatenate((total_feature_vectors,feature_vectors),axis = 0)
        return total_feature_vectors

    def __generate_models(self):
        models = []
        os.mkdir(self.checkpoint_models)
        for cluster in range(self.num_clusters):
            labels = []
            for i in range(self.__feature_vectors.shape[0]):
                if self.__cluster_labels[i] == cluster:
                    labels.append(1)
                else:
                    labels.append(0)
            model = svm.SVC(kernel='linear', C = 1.0)
            model.fit(self.__feature_vectors, labels)
            pickle.dump(model,open(os.path.join(self.checkpoint_models,str(cluster)+'.sav'),'wb'))
            models.append(model)
        return models

    def get_inference_score(self,feature_vector):
        scores = [model.predict([feature_vector]) for model in self.models]
        return max(scores)

    def __load_models(self):
        models = []
        for cluster in range(self.num_clusters):
            model = pickle.load(open(os.path.join(self.checkpoint_models,str(cluster)+'.sav'),'rb'))
            models.append(model)
        return models