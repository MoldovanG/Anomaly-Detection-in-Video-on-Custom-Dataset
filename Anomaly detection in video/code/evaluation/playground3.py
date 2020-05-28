from code.evaluation.ModelEvaluator import ModelEvaluator
from code.training.stage2_clustering_and_svms.DataSetTrainer_Stage2 import DataSetTrainer_Stage2
from code.training.utils.AutoEncoderModel import AutoEncoderModel
from code.training.utils.PrecisionCalculator import PrecisionCalculator
import time
autoencoder_images = AutoEncoderModel([],"raw_object_autoencoder")
autoencoder_gradients = AutoEncoderModel([],"gradient_object_autoencoder")

dataset_directory_path = "/home/george/Downloads/Licenta-refactored/Avenue Dataset/training_videos"
final_dataset_trainer = DataSetTrainer_Stage2(dataset_directory_path,autoencoder_images,autoencoder_gradients)

dataset_directory_path = "/home/george/Downloads/Licenta-refactored/Avenue Dataset/testing_videos"
ground_truth_directory = "/home/george/Downloads/Licenta-refactored/Avenue Dataset/testing_label_mask"
start = time.time()
modelEvaluator = ModelEvaluator(final_dataset_trainer,dataset_directory_path,ground_truth_directory)
modelEvaluator.evaluate_dataset()
end = time.time()
print(end-start)
print("video took %f seconds to process" % (end-start))
# precision_calculator = PrecisionCalculator()
# precision_calculator.show_average_precision(modelEvaluator.true_positives,modelEvaluator.false_positives,modelEvaluator.num_gt_detections)