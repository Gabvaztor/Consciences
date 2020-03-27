from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.layers as layers
import src.config.GlobalSettings as GS

from src.services.modeling.CModels import CModels
from ..Config import Config
from src.utils.Prints import pt

def set_cmodels(cmodel: CModels) -> CModels:
    return cmodel

def core(cmodel: CModels, config: Config):
    set_global_params(cmodel, config)
    main()

def set_global_params(cmodel: CModels, config: Config):
    """
    Set globals CMODEL and CONFIG(_) objects.
    Args:
        cmodel: Current CMODEL to be used. -From CModel module.
        config: Current CONFIG to be used. From Config module.
    """
    global CONFIG, CMODEL, _
    CMODEL = set_cmodels(cmodel)
    CONFIG = set_config(config)
    _ = CONFIG

def set_config(config: Config) -> Config:
    return config

CMODEL = set_cmodels(CModels())
CONFIG =  Config
_ = Config


if __name__ == "__main__":
    import Runner
    Runner.run()

def main(**kwargs):
    """

    Args:
        model_class: CModel class
        config: Config class
    """
    from tensorflow.python.client import device_lib
    #m = device_lib.list_local_devices()
    pt("CUDA status", tf.test.is_built_with_cuda())
    #tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        pt("Num GPUs Available: ", len(gpus))
    else:
        pt("No GPUs Availables")

    #tf.keras.backend.clear_session()

    """ LAYERS """
    model = network_structure_v1()
    model.summary()
    execute_model(model=model)

def execute_model(model: tf.keras.Sequential, **kwargs):
    # Print actual configuration
    CMODEL.print_current_configuration(config=CONFIG)

    # Batching values and labels from input and labels (with batch size)

    # TODO (@gabvaztor) When restore model and don't change train size, it must to keep the same order of
    #  train set.
    # To restore model

    if CONFIG.restore_model_flag:
        #self.load_and_restore_model_v2()
        # TODO (@gabvaztor) model path must satisfied path logic (/framework/Models/...). Not use SETTINGS for
        #  models paths
        #fullpath_save = CMODEL.settings_object.model_path + CONFIG.model_name_saved
        fullpath_save = GS.MODELS_PATH + "Retinopathy.1\\models\\" + CONFIG.model_name_saved
        pt("Trying to load model...")
        model = tf.keras.models.load_model(fullpath_save)
        pt("Model Loaded successfully!")

    # Besides this, when test/validation set requires check its accuracy but its size is very long to save
    # in memory, it has to update all files during training to get the exact precision.

    #train_current_model(model=model)
    CMODEL.train_current_model(model=model, config=CONFIG)

def train_current_model(model: tf.keras.Sequential):
    pass

def train_model(model: tf.keras.Sequential):
    """
    Args:
        model: current model

    Returns: model trained
    """
    # Training
    model.fit(x=CMODEL.input_batch, y=CMODEL.label_batch, batch_size=CMODEL.batch_size,
              epochs=2)
    return model

def evaluate_model(model: tf.keras.Sequential):
    """

    Args:
        model:

    Returns:

    """
    model.evaluate(CMODEL.x_test,  CMODEL.y_test, verbose=2)
    return model

def network_structure_v1():
    """

    Returns: model generated

    """
    """ INPUTS """
    # Order of shape: (height, width, dimensions)
    input_shape = (_.height, _.width, _.dimensions)
    #inputs = tf.keras.Input(shape=(_.height, _.width, _.dimensions))
    """ LAYERS """
    model = tf.keras.Sequential(name="Retinopathy")
    #tf.keras.backend.set_floatx('float64')
    convolution_1 = layers.Conv2D(filters=_.neurons[0], kernel_size=_.kernel_size, activation="relu",
                                  input_shape=input_shape)
    pool_1 = layers.MaxPooling2D(pool_size=_.pool_size, strides=_.strides, padding="same")
    dropout_1 = layers.Dropout(rate=_.train_dropout)
    convolution_2 = layers.Conv2D(filters=_.neurons[1], kernel_size=_.kernel_size)
    pool_2 = layers.MaxPooling2D(pool_size=_.pool_size, strides=_.strides, padding="same")
    convolution_3 = layers.Conv2D(filters=_.neurons[2], kernel_size=_.kernel_size, activation="relu")
    """ DENSE LAYER """
    flatten = layers.Flatten()
    #dense_1 = layers.Dense(units=_.neurons[3], activation='relu')
    #normalization = layers.BatchNormalization()
    outputs = layers.Dense(units=_.number_of_classes, activation="softmax")
    #add_to_model = lambda x: model.add(x)
    layers_ = [
        convolution_1,
        pool_1,
        dropout_1,
        convolution_2,
        pool_2,
        #convolution_3,
        flatten,
        #dense_1,
        #normalization,
        outputs,
    ]
    [model.add(layer) for layer in layers_]
    loss = tf.keras.losses.categorical_crossentropy(from_logits=True)
    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adagrad(),
                  #optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    return model
