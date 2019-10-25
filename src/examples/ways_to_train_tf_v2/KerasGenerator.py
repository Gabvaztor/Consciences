import tensorflow as tf
import tensorflow.keras.layers as layers

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

def main():
    """

    Args:
        model_class: CCModel class
        config: Config class
    """
    from tensorflow.python.client import device_lib
    #m = device_lib.list_local_devices()
    pt("CUDA status:", tf.test.is_built_with_cuda())
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
    #train_model(model=model)
    #evaluate_model(model=model)
    # Training
    """
    model.fit(x=CMODEL.input_batch, y=CMODEL.label_batch,
              batch_size=64, epochs=CMODEL.epoch_numbers,
              use_multiprocessing=True)
    """
    training_generator = CMODEL.batch_generator_v2(shape=_.shape)
    validation_generator = CMODEL.batch_generator_v2(shape=_.shape, is_test=True)

    filepath_to_save = CMODEL.settings_object.model_path + "model" + "{epoch:04d}" + ".ckpt"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=filepath_to_save,
                                           verbose=1,
                                           period=1),
    ]


    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=CMODEL.epoch_numbers,
                                  use_multiprocessing=True,
                                  verbose=2,
                                  callbacks=callbacks)
    model.evaluate(CMODEL.x_test,  CMODEL.y_test, verbose=2)

def train_model(model):
    """
    Args:
        model: current model

    Returns: model trained
    """
    # Training
    model.fit(x=CMODEL.input_batch, y=CMODEL.label_batch, batch_size=CMODEL.batch_size,
              epochs=2)
    return model

def evaluate_model(model):
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
    convolution_1 = layers.Conv2D(filters=_.neurons[0], kernel_size=_.kernel_size, activation="relu",
                                  input_shape=input_shape)
    pool_1 = layers.MaxPooling2D(pool_size=_.pool_size, strides=_.strides, padding="same")
    dropout_1 = layers.Dropout(rate=_.train_dropout)
    convolution_2 = layers.Conv2D(filters=_.neurons[1], kernel_size=_.kernel_size, activation="relu")
    pool_2 = layers.MaxPooling2D(pool_size=_.pool_size, strides=_.strides, padding="same")
    convolution_3 = layers.Conv2D(filters=_.neurons[2], kernel_size=_.kernel_size, activation="relu")
    """ DENSE LAYER """
    flatten = layers.Flatten()
    #dense_1 = layers.Dense(units=_.neurons[3], activation='relu')
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
        outputs
        ,
    ]
    [model.add(layer) for layer in layers_]
    model.compile(loss=tf.keras.losses.mean_squared_error,
                  optimizer=tf.keras.optimizers.Adadelta())
    return model
