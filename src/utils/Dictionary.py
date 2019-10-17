DICT = {}

def dict_return(key):
    """
    Function to return the value from a key
    :param key:
    :return: The Value
    """
    # TODO Check valid values
    return DICT[key]

class Dictionary:
    """
    Dictionary:Encoder
    Each attribute will be a dictionary's key
    Attribute = Dictionary Key
    Value's Attribute = Dictionary Value
    Doing an abstract class like a Dictionary we have the references to all keys from another class.
    """
    """
    STRINGS
    """

    # German Prices
    string_german_prices_csv_path = "D:\Machine_Learning\DataSets\German-Prices\GermanyPrices2005-2016.csv"
    # Data.csv from Breast_Cancer_Wisconsin project
    string_path_Breast_Cancer_Wisconsin = '../DataTest/Breast_Cancer_Wisconsin/data.csv'
    # Data.csv Label Column name
    string_label_column_name_Breast_Cancer_Wisconsin = 'diagnosis'
    # Hierarchy option
    string_labels_type_option_hierarchy = 'hierarchy'
    # Format png
    string_extension_png = '.png'
    # Format png
    string_extension_jpeg = '.jpeg'
    # Format ckpt
    string_ckpt_extension = '.ckpt'
    # Format meta
    string_meta_extension = '.meta'
    # Format csv
    string_csv_extension = '.csv'
    # Format npy (numpy)
    string_npy_extension = '.npy'
    # Format zip
    string_zip_extension = '.zip'
    # Format json
    string_json_extension = '.json'
    # Train String
    string_train = 'train'
    # Test String
    string_test = 'test'
    # Separator
    string_separator = '-------------------------------------'
    # Gray scale
    string_gray_scale = 'gray_scale'
    # Want to save model
    string_want_to_save = "Do you want to save the model?"
    #
    string_want_to_continue_without_load = "You select 'restore model' but model doesn't exist," \
                                           " do you want to continue without loading a model?"
    # Answer
    string_get_response = "Press 'Y' to Yes or 'N' to No:"
    # Char Y
    string_char_Y = "Y"
    # Char N
    string_char_N = "N"
    # Char N
    string_char_comma = ","
    # Char _
    string_char_low_stripe = "_"

    """
    STRINGS PROBLEMS
    
    Here will be all problems to load configuration
    """
    # TODO(@gabvaztor) Create new Dict (or class) to contain the different parts of Dictionary
    # Settings path (for save models) (it has the same configuration that kaggle submit best practice)
    string_settings_sberbank_russian_houssing_market_path = "../Settings/Sberbank_Russian_Houssing_Market/SETTINGS.json"

    string_settings_german_signal_path = "../Settings/German_Signal/SETTINGS.json"

    string_settings_zillow_price = "../Settings/Zillow_Price/SETTINGS.json"

    string_settings_web_traffic = "../Settings/Web_Traffic_Time/SETTINGS.json"

    string_settings_retinopathy_k = "../Settings/Retinopathy_k/SETTINGS.json"
    """
    STRING FILENAMES
    """
    """ NUMPY"""
    filename_train_accuracies = "train_accuracies"
    filename_validation_accuracies = "validation_accuracies"
    filename_train_losses = "train_losses"
    filename_validation_losses = "validation_losses"
    filename_numpy_default = "numpy_file_"

    """
    STRINGS OPTIONS
    
    Here you add the id option of your problem
    """
    # Option signals image problem
    string_option_signals_images_problem = 'signals_images_problems'
    # Option german prizes problem
    string_option_german_prizes_problem = 'german_prizes_problem'
    # Option zillow price problem
    string_option_zillow_price_problem = 'zillow_price_problem'
    # Option zillow price problem
    string_option_web_traffic_problem = 'web_traffic_problem'
    # Option Breast Cancer Wisconsin problem
    string_breast_cancer_wisconsin_problem = 'web_traffic_problem'
    # Option Retinopathy_K problem
    string_option_retinopathy_k_problem = "retinopathy_k_problem"
