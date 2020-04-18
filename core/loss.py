import tensorflow as tf

from configuration import Config


class FocalLoss:
    def __call__(self, y_true, y_pred, *args, **kwargs):
        return FocalLoss.__neg_loss(y_pred, y_true)

    @staticmethod
    def __neg_loss(pred, gt):
        pos_idx = tf.cast(tf.math.equal(gt, 1), dtype=tf.float32)
        neg_idx = tf.cast(tf.math.less(gt, 1), dtype=tf.float32)
        neg_weights = tf.math.pow(1 - gt, 4)

        loss = 0
        num_pos = tf.math.reduce_sum(pos_idx)
        pos_loss = tf.math.log(pred) * tf.math.pow(1 - pred, 2) * pos_idx
        pos_loss = tf.math.reduce_sum(pos_loss)
        neg_loss = tf.math.log(1 - pred) * tf.math.pow(pred, 2) * neg_weights * neg_idx
        neg_loss = tf.math.reduce_sum(neg_loss)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class RegL1Loss:
    def __call__(self, y_true, y_pred, mask, index, *args, **kwargs):
        y_pred = RegL1Loss.gather_feat(y_pred, index)
        mask = tf.tile(tf.expand_dims(mask, axis=-1), tf.constant([1, 1, 2], dtype=tf.int32))
        loss = tf.math.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
        reg_loss = loss / (tf.math.reduce_sum(mask) + 1e-4)
        return reg_loss

    @staticmethod
    def gather_feat(feat, idx):
        feat = tf.reshape(feat, shape=(feat.shape[0], -1, feat.shape[-1]))
        idx = tf.cast(idx, dtype=tf.int32)
        feat = tf.gather(params=feat, indices=idx, batch_dims=1)
        return feat


class CombinedLoss:
    def __init__(self):
        self.heatmap_loss_object = FocalLoss()
        self.reg_loss_object = RegL1Loss()
        self.wh_loss_object = RegL1Loss()

    def __call__(self, y_pred, heatmap_true, reg_true, wh_true, reg_mask, indices, *args, **kwargs):
        heatmap, reg, wh = tf.split(value=y_pred, num_or_size_splits=[Config.num_classes, 2, 2], axis=-1)
        heatmap = tf.clip_by_value(t=tf.math.sigmoid(heatmap), clip_value_min=1e-4, clip_value_max=1.0 - 1e-4)
        heatmap_loss = self.heatmap_loss_object(y_true=heatmap_true, y_pred=heatmap)
        off_loss = self.reg_loss_object(y_true=reg_true, y_pred=reg, mask=reg_mask, index=indices)
        wh_loss = self.wh_loss_object(y_true=wh_true, y_pred=wh, mask=reg_mask, index=indices)
        return Config.hm_weight * heatmap_loss + Config.off_weight * off_loss + Config.wh_weight * wh_loss
