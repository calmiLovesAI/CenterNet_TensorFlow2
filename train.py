import tensorflow as tf
import time

from core.centernet import PostProcessing, CenterNet
from data.dataloader import DetectionDataset, DataLoader
from configuration import Config


def print_model_summary(network):
    sample_inputs = tf.random.normal(shape=(Config.batch_size, Config.image_size[0], Config.image_size[1], Config.image_channels))
    sample_outputs = network(sample_inputs, training=True)
    network.summary()

if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # dataset
    train_dataset = DetectionDataset()
    train_data, train_size = train_dataset.generate_datatset()
    data_loader = DataLoader()
    steps_per_epoch = tf.math.ceil(train_size / Config.batch_size)


    # model
    centernet = CenterNet()
    print_model_summary(centernet)

    # optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4,
                                                                 decay_steps=steps_per_epoch * Config.learning_rate_decay_epochs,
                                                                 decay_rate=0.96)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # metrics
    loss_metric = tf.metrics.Mean()

    post_process = PostProcessing()

    def train_step(batch_images, batch_labels):
        with tf.GradientTape() as tape:
            pred = centernet(batch_images, training=True)
            loss_value = post_process.training_procedure(batch_labels=batch_labels, pred=pred)
        gradients = tape.gradient(target=loss_value, sources=centernet.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, centernet.trainable_variables))
        loss_metric.update_state(values=loss_value)

    for epoch in range(Config.epochs):
        for step, batch_data in enumerate(train_data):
            step_start_time = time.time()
            images, labels = data_loader.read_batch_data(batch_data)
            train_step(images, labels)
            step_end_time = time.time()
            print("Epoch: {}/{}, step: {}/{}, loss: {}, time_cost: {:.3f}s".format(epoch,
                                                                                  Config.epochs,
                                                                                  step,
                                                                                  steps_per_epoch,
                                                                                  loss_metric.result(),
                                                                                  step_end_time - step_start_time))
        loss_metric.reset_states()

        if epoch % Config.save_frequency == 0:
            centernet.save_weights(filepath=Config.save_model_dir+"epoch-{}".format(epoch), save_format="tf")

    centernet.save_weights(filepath=Config.save_model_dir + "saved_model", save_format="tf")
