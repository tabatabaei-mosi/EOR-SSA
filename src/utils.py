
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
