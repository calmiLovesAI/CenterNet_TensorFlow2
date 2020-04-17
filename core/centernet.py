import tensorflow as tf

from configuration import Config
from core.models.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
from data.dataloader import GT
from core.loss import CombinedLoss

backbone_zoo = {"resnet_18": resnet_18(),
                "resnet_34": resnet_34(),
                "resnet_50": resnet_50(),
                "resnet_101": resnet_101(),
                "resnet_152": resnet_152()}


class CenterNet(tf.keras.Model):
    def __init__(self):
        super(CenterNet, self).__init__()
        self.backbone = backbone_zoo[Config.backbone_name]

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs, training=training)
        x = tf.concat(values=x, axis=-1)
        return x


class PostProcessing:
    @staticmethod
    def training_procedure(batch_labels, pred):
        gt = GT(batch_labels)
        gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices = gt.get_gt_values()
        loss_object = CombinedLoss()
        loss = loss_object(y_pred=pred, heatmap_true=gt_heatmap, reg_true=gt_reg, wh_true=gt_wh, reg_mask=gt_reg_mask, indices=gt_indices)
        return loss

    def testing_procedure(self):
        pass