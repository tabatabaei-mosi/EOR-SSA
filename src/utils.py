from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import normalize, to_categorical


def path_check(file_path, path_return=False):
    """
    check the path, if it doesn't exist, create it.

    Arguments:
        file_path {str} -- the path should be checked.
        
    keyword Arguments:
        path_return {bool} -- if the path doesn't exist, return the path string (default: False)
    
    Returns:
        str -- the path file (if path_return is True)
    """
    
    # check if the path exist
    path_exist = Path(file_path).exists()
    
    # if the path doesn't exist, create it
    if not path_exist:
        Path(file_path).mkdir(parents=True)
        
    # return the created path
    if path_return:
        return file_path
        


def f1_accuracy(recall, precision):
    """
    f1 score for the model.
    
    Arguments:
        recall {array-like} -- recall score calculated by the model
        
        precision {array-like} -- precision score calculated by the model

    Returns:
       array-like -- calculated f1 score for the model
    """
 
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def decode_label(output_codes, re_group=False):
    """
    decode the label code (retruned by model or true outputs) to the true label (EOR method).
    The results of this function is used for plotting the confusion matrix.

    Arguments:
        output_codes {set} -- the label codes returned by the model (true or predicted output)

    Keyword Arguments:
        re_group {bool} -- If the data is re-grouped to contain only 3 EOR category to balance dataset (default: True)

    Returns:
        list -- the true label of the EOR methods (The real name of the EOR methods) 
    """
    
    # The list of true label of the EOR methods
    decode_ouputs = list()
    
    if re_group:
        # If re-grouped strategy is used
        true_label = ['Thermal methods', 'Gas miscible/im.', 'Chemical and others']

    else:
        # The whole list of EOR methods in dataset (sorted in a list according to the code assigned to each method)
        true_label = [
            'Steam', 'HW', 'Combustion', 'CO2 mis', 'CO2 immis', 
            'H mis', 'H immis', 'N immi', 'Polymer',
            'AGM', 'Microbial', 'Cyclic steam', 'HC mis/Water', 'AGA', 'Steam-SAGD'
        ]
        
    # Decode the label code to the true label
    for code in output_codes:
        decode_ouputs.append(true_label[code])
        
    return decode_ouputs



def preprocessor(
    numpy_dataset, 
    inp_feature, 
    test_frac,
    normalize_data=True, seed=None
    ):
    """
    Preprocesses the input dataset by shuffling, splitting into input and output,
    categorizing the output (one-hot encoding), and normalizing the input data (if specified).
    Splits the preprocessed dataset into training, validation, and testing sets.
    
    Arguments:
        numpy_dataset: numpy array containing the input dataset
        inp_feature: number of input features
        test_frac: fraction of the dataset to be used for testing (between 0 and 1)
    
    Keyword Arguments:
        normalize_data: whether to normalize the input data (default is True)
        seed (optinal): random seed for shuffling the dataset (default is None)
    
    Returns:
        A dict containing preprocessed input and output data for training and testing.
        keywords: 'x_train', 'x_test', 'y_train', 'y_test'
    """
    
    # Shuffle the dataset
    shuffled_dataset = shuffle(numpy_dataset, random_state=seed)
    
    # Split the dataset into input and output
    X = shuffled_dataset[:, :inp_feature]
    Y = shuffled_dataset[:, inp_feature]
    
    # Convert the output to categorical (one-hot encoded category)
    y_categorize = to_categorical(Y)
    
    # Normalize the input data
    if normalize_data:
        normalized_x = normalize(X, axis=0)
    else:
        normalized_x = X
    
    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(normalized_x, y_categorize, test_size=test_frac, random_state=seed)
    data_set = {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    # Return the preprocessed input and output data
    return data_set