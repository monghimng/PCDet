import numpy as np

'''
Keeps track of a confusion_matrix of all correct and incorrect predictions. Given this
matrix, we can calculate

1. mIoU
2. Pixel accuracy

Note that the confusion matrix is structured such that 
1. sum along the rows (result is a column vector) gives the count of gt pixels
2. sum along the cols (result is a row vector) gives the count of predicted pixels

'''

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros([num_class, num_class])

    def Pixel_Accuracy(self):
        '''
        sum of diagonal is is the number of correct classification
        sum of all entries is the number of training samples
        '''
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def class_iou(self):
        gt_classes_count = np.sum(self.confusion_matrix, axis=1)
        predicted_classes_count = np.sum(self.confusion_matrix, axis=0)
        true_positive_classes = np.diag(self.confusion_matrix)
        return true_positive_classes / (gt_classes_count + predicted_classes_count - true_positive_classes)

    def _generate_matrix(self, gt_image, pre_image):
        '''
        |pre_image| and |gt_image| can be any dimension, as long as they have the same
        dimension.

        Each element contains the predicted label. This is a werid way to calculate confusion_matrix,
        and sckilearn offer a simpler way. Essentially, we are viewing the position of each
        cell of the confusion matrix as a two digit number based |self.num_class|.
        '''
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        '''
        Each element contains the predicted label. Should be np array.
        '''
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)