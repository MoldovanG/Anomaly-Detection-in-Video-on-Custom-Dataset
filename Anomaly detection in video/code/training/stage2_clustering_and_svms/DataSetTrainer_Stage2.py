import os
import pickle
import numpy as np
from sklearn import svm

from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize

from stage2_clustering_and_svms.VideoProcessor_Stage2 import VideoProcessor_Stage2
from utils.AutoEncoderModel import AutoEncoderModel

class DataSetTrainer_Stage2:
    """
    Class used for genereting the svm models neede for the inferrence phase.
    It requires the dataset directory path, the pretrained-autoencoder
    and based on that, it will cluster the data, and train the 1 vs all svm models.
    Those models will be later used in the inferrrence phase for predicting the anomalies.
    """
    def __init__(self,dataset_directory_path,autoencoder_images : AutoEncoderModel, autoencoder_gradients : AutoEncoderModel, norm):
        self.num_clusters = 10
        self.l = norm
        self.dataset_directory_path = dataset_directory_path
        self.checkpoint_directory = os.path.join(dataset_directory_path,"checkpoints")
        if not os.path.exists(self.checkpoint_directory):
            os.mkdir(self.checkpoint_directory)
        self.checkpoint_models_path = os.path.join(dataset_directory_path, "checkpoints", "pretrained_svm")
        self.saved_feature_vectors_path = os.path.join(dataset_directory_path,"saved_feature_vectors")
        self.dataset_training_videos_path = os.path.join(dataset_directory_path,"training_videos")
        self.autoencoder_images = autoencoder_images
        self.autoencoder_gradients = autoencoder_gradients
        self.__feature_vectors = self.__get_dataset_feature_vectors()
        self.__feature_vectors = self.normalize_data(self.__feature_vectors)
        print(self.__feature_vectors.shape)
        print(np.max(self.__feature_vectors[0]), " ", np.min(self.__feature_vectors[0]))
        print(self.__feature_vectors.shape)
        self.__cluster_labels = self.__cluster_data(self.__feature_vectors, self.num_clusters)
        self.__generate_models()
        self.model = self.__load_model()


    def __cluster_data(self, feature_vectors, num_clusters):

        clustering_savedir = os.path.join(self.dataset_directory_path,"checkpoints","clustering_labels")
        if not os.path.exists(clustering_savedir):
            os.mkdir(clustering_savedir)
            print('Clustering data ...')
            kmeans = KMeans(n_clusters=num_clusters).fit(feature_vectors)
            np.save(os.path.join(clustering_savedir,"labels.npy"),kmeans.labels_)
            print('Finished clustering !')
            return kmeans.labels_
        else:
            print("Clustering labels found !! Automatically loading them ...")
            return np.load(os.path.join(clustering_savedir,"labels.npy"))

    def __get_dataset_feature_vectors(self):
        total_feature_vectors = np.resize([],(0,3072))
        for video_name in os.listdir(self.dataset_training_videos_path):
            print(video_name)
            name_withouth_extesion = video_name.split(".")[0]
            video_path = os.path.join(self.dataset_training_videos_path, video_name)
            feature_vectors = np.array([])
            feature_vector_save_point = os.path.join(self.saved_feature_vectors_path, name_withouth_extesion+".npy")
            if not os.path.exists(feature_vector_save_point):
                videoprocessor = VideoProcessor_Stage2(video_path, self.autoencoder_images, self.autoencoder_gradients)
                feature_vectors = videoprocessor.get_feature_vectors()
                np.save(feature_vector_save_point, feature_vectors)
            else:
                print("Loading feature_vectors from file for video :", video_name)
                feature_vectors = np.load(feature_vector_save_point, allow_pickle=True)
                print(feature_vectors.shape)
            total_feature_vectors = np.concatenate((total_feature_vectors,feature_vectors),axis = 0)
        print("Total number of feature vectors : ", total_feature_vectors.shape)
        return total_feature_vectors

    def __generate_models(self):
        if not os.path.exists(self.checkpoint_models_path):
            os.mkdir(self.checkpoint_models_path)
            checkpoint_model_path = os.path.join(self.checkpoint_models_path, 'model.sav')
            predictor = svm.LinearSVC(C=1.0,multi_class='ovr',max_iter=len(self.__feature_vectors)*5)
            clf = OneVsRestClassifier(predictor)
            clf.fit(self.__feature_vectors,self.__cluster_labels)
            pickle.dump(clf,open(checkpoint_model_path,'wb'))
            return clf
        else:
            print("Pre-trained OneVsRest classifier found!! Automatically loading it ...")
            return None

    def get_inference_score(self,feature_vector):
        scores = self.model.decision_function([feature_vector])[0]
        return -max(scores)

    def __load_model(self):
        model = pickle.load(open(os.path.join(self.checkpoint_models_path, 'model.sav'), 'rb'))
        return model

    def normalize_data(self, feature_vectors):
        if self.l == 0 :
            return feature_vectors
        else:
            return normalize(feature_vectors, 'l'+str(self.l))
