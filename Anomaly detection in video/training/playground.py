import numpy as np

from training.DataSetTrainer import DataSetTrainer

VIDEO_DIR = '/Avenue Dataset'
dataset_trainer = DataSetTrainer(VIDEO_DIR)

input = np.expand_dims(dataset_trainer.total_objects[0],axis=0 )
prediction = dataset_trainer.autoencoder_images.autoencoder.predict(input)[0]
# plt.figure()
# plt.imshow(prediction[:,:,0]*255, cmap='gray')
# plt.show()