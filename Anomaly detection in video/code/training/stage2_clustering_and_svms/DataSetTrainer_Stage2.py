import os
import pickle
import numpy as np
from sklearn import svm

from sklearn.cluster import KMeans

from code.training.stage2_clustering_and_svms.VideoProcessor_Stage2 import VideoProcessor_Stage2
from code.training.utils.AutoEncoderModel import AutoEncoderModel


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
        self.saved_feature_vectors_path = "/home/george/Licenta/Anomaly detection in video/Avenue Dataset/saved_feature_vectors"
        self.dataset_directory_path = dataset_directory_path
        self.autoencoder_images = autoencoder_images
        self.autoencoder_gradients = autoencoder_gradients
        self.__feature_vectors = self.__get_dataset_feature_vectors()
        self.__feature_vectors = self.drop_the_one_hot_encoding(self.__feature_vectors)
        print(self.__feature_vectors.shape)
        print(np.max(self.__feature_vectors[0]), " ", np.min(self.__feature_vectors[0]))
        print(self.__feature_vectors.shape)
        self.__cluster_labels = self.__cluster_data(self.__feature_vectors, self.num_clusters)
        self.__generate_models()
        self.models = self.__load_models()

    def __cluster_data(self, feature_vectors, num_clusters):
        clustering_savedir = "/home/george/Licenta/Anomaly detection in video/Avenue Dataset/checkpoints/clustering_labels"
        if not os.path.exists(clustering_savedir):
            os.mkdir(clustering_savedir)
            kmeans = KMeans(n_clusters=num_clusters).fit(feature_vectors)
            best_kmeans = kmeans
            for i in range (4):
                print("Working on clustering number : " + str(i))
                kmeans = KMeans(n_clusters=num_clusters).fit(feature_vectors)
                if kmeans.inertia_ < best_kmeans.inertia_:
                    print("Am gasit o clusterizare mai buna")
                    best_kmeans = kmeans
            np.save(os.path.join(clustering_savedir,"labels.npy"),best_kmeans.labels_)
            return best_kmeans.labels_
        else:
            return np.load(os.path.join(clustering_savedir,"labels.npy"))

    def __get_dataset_feature_vectors(self):
        total_feature_vectors = np.resize([],(0,3163))
        for video_name in os.listdir(self.dataset_directory_path):
            print(video_name)
            name_withouth_extesion = video_name.split(".")[0]
            video_path = os.path.join(self.dataset_directory_path,video_name)
            feature_vectors = np.array([])
            feature_vector_save_point = os.path.join(self.saved_feature_vectors_path, name_withouth_extesion+".npy")
            if not os.path.exists(feature_vector_save_point):
                videoprocessor = VideoProcessor_Stage2(video_path, self.autoencoder_images, self.autoencoder_gradients)
                feature_vectors = videoprocessor.get_feature_vectors()
                np.save(feature_vector_save_point, feature_vectors)
            else:
                print("Loading feature_vectors from file for video :", video_name)
                feature_vectors = np.load(feature_vector_save_point)
                print(feature_vectors.shape)
            total_feature_vectors = np.concatenate((total_feature_vectors,feature_vectors),axis = 0)
        print("Total number of feature vectors : ", total_feature_vectors.shape)
        return total_feature_vectors

    def __generate_models(self):
        models = []
        if not os.path.exists(self.checkpoint_models):
            os.mkdir(self.checkpoint_models)
        for cluster in range(self.num_clusters):
            checkpoint_model_path = os.path.join(self.checkpoint_models,str(cluster)+'.sav')
            if not os.path.exists(checkpoint_model_path):
                print("working on : ", checkpoint_model_path)
                labels = []
                for i in range(self.__feature_vectors.shape[0]):
                    if self.__cluster_labels[i] == cluster:
                        labels.append(1)
                    else:
                        labels.append(0)
                model = svm.LinearSVC(C=1.0)
                model.fit(self.__feature_vectors, labels)
                pickle.dump(model,open(checkpoint_model_path,'wb'))
                models.append(model)
        return models

    def get_inference_score(self,feature_vector):
        scores = [model.decision_function([feature_vector])[0] for model in self.models]
        return max(scores)

    def __load_models(self):
        models = []
        for cluster in range(self.num_clusters):
            model = pickle.load(open(os.path.join(self.checkpoint_models,str(cluster)+'.sav'),'rb'))
            models.append(model)
        return models

    def drop_the_one_hot_encoding(self, feature_vectors):
        return  feature_vectors[:,91:]
