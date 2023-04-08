from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import normalize, to_categorical


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