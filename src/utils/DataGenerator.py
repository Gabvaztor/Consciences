from src.services.modeling.CModels import CModels
import tensorflow

class DataGenerator(tensorflow.keras.utils.Sequence):

    'Generates data for Keras'
    def __init__(self, CMODELS: CModels, shape, is_test=False):
        'Initialization'
        self.CMODELS = CMODELS
        self.is_test = is_test
        self.shape = shape
        self.batch_size = CMODELS.batch_size
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.CMODELS.trains

    def __getitem__(self, index):
        'Generate one batch of data'
        return self.CMODELS.update_batch(is_test=self.is_test)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.CMODELS.shuffle_data and not self.is_test:
            self.CMODELS.input, self.CMODELS.input_labels = self.CMODELS.shuffle_dataset(self.input, self.input_labels)