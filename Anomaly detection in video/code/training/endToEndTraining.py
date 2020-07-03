import argparse
import sys

sys.path.append('../')
from stage1_autoencoders.DataSetTrainer import DataSetTrainer
from stage2_clustering_and_svms.DataSetTrainer_Stage2 import DataSetTrainer_Stage2
from utils.AutoEncoderModel import AutoEncoderModel

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--datasetdirectory", required=True,
	help="path to dataset directory. The dataset directory has to contain the following subfolders : /testing_videos , /training_videos")
ap.add_argument("-n", "--normalization", required=False,
	help="The normalization parameter: Can be 0,1,2. This sets the norm used(raw,l1 or l2) for preprocessing the data before clustering and 1vsRest training")
args = vars(ap.parse_args())

dataset_directory_path = args['datasetdirectory']
dataset_trainer = DataSetTrainer(dataset_directory_path)
autoencoder_images = AutoEncoderModel([],"raw_object_autoencoder",dataset_directory_path)
autoencoder_gradients = AutoEncoderModel([],"gradient_object_autoencoder",dataset_directory_path)
norm = 0 if args['normalization'] is None else args['normalization']
final_dataset_trainer = DataSetTrainer_Stage2(dataset_directory_path,autoencoder_images,autoencoder_gradients,norm)

# input = np.expand_dims(dataset_trainer.total_objects[0],axis=0 )
# prediction = dataset_trainer.autoencoder_images.autoencoder.predict(input)[0]
# plt.figure()
# plt.imshow(prediction[:,:,0]*255, cmap='gray')
# plt.show()
# plt.figure()
# plt.imshow(input[0][:,:,0]*255, cmap='gray')
# plt.show()
