import tensorflow as tf
import tensorflow.keras.layers as layers


if __name__ == "__main__":
    import Runner
    Runner.run()

from src.services.modeling.CModels import CModels
from src.utils.Prints import pt
from ..Config import Config

def main(cmodel: CModels, config: Config):
    """

    Args:
        model_class: CCModel class

    """
    tf.debugging.set_log_device_placement(True)
    pt("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    tf.keras.backend.clear_session()
    print("C")
    # Treat inputs
    cmodel.batch_size = 128
    cmodel.update_batch()
    x_test, y_test = cmodel.update_batch(is_test=True)
    """ LAYERS """
    model = network_structure_v1(config)
    model.summary()
    # Training
    model.fit(x=cmodel.input_batch, y=cmodel.label_batch, batch_size=8, verbose=2)


def network_structure_v1(_: Config):
    """

    Args:
        _:

    Returns:

    """
    """ INPUTS """
    # Order of shape: (height, width, dimensions)
    input_shape = shape=(_.height, _.width, _.dimensions)
    inputs = tf.keras.Input(shape=(_.height, _.width, _.dimensions))
    """ LAYERS """
    model = tf.keras.Sequential(name="Retinopathy")
    convolution_1 = layers.Conv2D(filters=_.neurons[0], kernel_size=_.kernel_size, activation="relu",
                                  input_shape=input_shape)
    pool_1 = layers.MaxPooling2D(pool_size=_.pool_size, strides=_.strides, padding="same")
    dropout_1 = layers.Dropout(rate=_.train_dropout)
    convolution_2 = layers.Conv2D(filters=_.neurons[1], kernel_size=_.kernel_size, activation="relu")
    pool_2 = layers.MaxPooling2D(pool_size=_.pool_size, strides=_.strides, padding="same")
    convolution_3 = layers.Conv2D(filters=_.neurons[2], kernel_size=_.kernel_size, activation="relu")
    """ DENSE LAYER """
    flatten = layers.Flatten()
    dense_1 = layers.Dense(units=_.neurons[2], activation='relu')
    outputs = layers.Dense(units=_.number_of_classes, activation="softmax")
    #add_to_model = lambda x: model.add(x)
    layers_ = [convolution_1, pool_1, dropout_1, convolution_2, pool_2, convolution_3, flatten, dense_1, outputs]
    [model.add(layer) for layer in layers_]
    model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.keras.optimizers.Adadelta())
    return model
