import os

import numpy as np
from matplotlib import pyplot as plt

class PrecisionCalculator:

    def __init__(self) -> None:
        super().__init__()

    def __compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def show_average_precision(self,true_positives,false_positives,num_gt_detections):
        cum_false_positive = np.cumsum(np.array(false_positives))
        cum_true_positive = np.cumsum(np.array(true_positives))
        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.__compute_average_precision(rec, prec)
        # plt.plot(rec, prec, '-')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Average precision %.3f' % average_precision)
        # plt.savefig(os.path.join("/home/george/Licenta/Anomaly detection in video/pictures", 'precizie_medie.png'))
        # plt.show()
        # print("Accuraccy is :",
        #       str(max(cum_true_positive) * (100 / (max(cum_true_positive) + max(cum_false_positive)))), "%")
        # print("Num of gt_detections:", str(num_gt_detections))
        return average_precision

