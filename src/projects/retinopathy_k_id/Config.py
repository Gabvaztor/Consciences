from src.config.Projects import Projects

problem_id = Projects.retinopathy_k_problem_id
options = [problem_id, 1, 720, 1280]

class Config():
    problem_id = Projects.retinopathy_k_problem_id
    dimensions = 3
    width = 1280
    height = 720
    shape = [dimensions, width, height]
    # Convolution
    kernel_size = (20, 20)
    # Pooling
    pool_size = [2, 2]
    strides = 2
    # Neurons
    neurons = [8, 16, 16, 32]
    # Dropout
    train_dropout = 0.5
    # Training
    learning_rate = 1e-4
    number_of_classes = 5
    # Model
    model_name_saved = "modelckpt_" + "my_model.h5"
    # Options
    options = [problem_id, 1, height, width]
def _protected():
    pass

def __private():
    pass

def call():
    return Config()
