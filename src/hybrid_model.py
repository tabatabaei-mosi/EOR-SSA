import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from mealpy.swarm_based import SSA
from tensorflow.keras.losses import CategoricalCrossentropy
from utils import path_check


class SSA_ANN:
    
    """
    This class is for creating a neural network model and using SSA for optimizing its weights (parameters).
    """
    
    def __init__(
        self,
        dataset,
        n_hidden_nodes, activation_list,
        epoch, pop_size, ST, Producers, Scroungers
    ):
        """
        This is the constructor of the class.

        Arguments:
            dataset {dict} -- A dict with following keys "x_train", "y_train", "x_test", "y_test" (respectively). 
            
            n_hidden_nodes {list} -- number of hidden nodes in each hidden layer (respectively).
            Note that the length of this list must be equal to the number of hidden layers.
            The number of hidden layers is determined by the length of this list.
            The input and output layer nodes are determined by the dataset.
            
            activation_list {list} -- activation function for each hidden layer (respectively).
            Note that the length of this list must be equal to the number of hidden layers.
            The output activation function is set to "Softmaxt" which is essential for classification problems.
            
            epoch {int} -- Number of epochs for SSA optimizer.
            
            pop_size {float} -- Population size for SSA optimizer.
            
            ST {float} -- Safty treshold for SSA optimizer.
            
            Producers {float} -- The number of producers in population.
            
            Scroungers {float} -- The number of scroungers in population.
        """
        
        # initializing the class attributes.
        self.X_train, self.y_train, self.X_test, self.y_test = dataset["x_train"], dataset["y_train"], dataset["x_test"], dataset["y_test"]
        self.n_hidden_nodes = n_hidden_nodes
        self.activation = activation_list
        self.epoch = epoch
        self.pop_size = pop_size
        self.ST = ST
        self.PD = Producers
        self.SD = Scroungers
        
        # number of input and output nodes according to the dataset.
        self.n_inputs = self.X_train.shape[1]
        self.output_shape = self.y_train.shape[1]
        
        # initializing the neural network model and the problem parameters.
        self.model, self.problem_size, self.n_dims, self.problem = None, None, None, None
        self.optimizer, self.solution, self.best_fit = None, None, None
        
        # initializing the paths for saving the results.
        self.log_path = path_check(file_path="Results", path_return=True)
        self.charts_path = path_check(file_path="Results/Charts", path_return=True)
        
        # initializing the model information (for saving the results in a proper way)
        self.model_info = f"E={self.epoch},PS={self.pop_size},SSA=[{self.PD, self.SD, self.ST}],ANN={self.n_hidden_nodes}"
        

    def create_model(self):
        """
        This function is for creating a neural network model by using keras library.\\
        Sequential model is used for creating the model and the hidden layers are added to the model with a for loop.\\
        The activation function for each hidden layer is determined by the activation_list and the output layer set to "softmax".\\
        Finally desing the problem that SSA will optimize.
        """
        # creating the sequentail model
        model = Sequential()
        
        # adding hidden layers to the model
        for i in range(len(self.n_hidden_nodes)):
            # the input layer is added with the input_dim parameter which determined by "x_train.shape".
            if i == 0:
                model.add(Dense(self.n_hidden_nodes[i], input_dim=self.n_inputs, 
                                activation=self.activation[i]))
            else:
                model.add(Dense(self.n_hidden_nodes[i], activation=self.activation[i]))
                
        # adding the output layer to the model. "softmax" for output later is recommended for classification problems.
        model.add(Dense(self.output_shape, activation='softmax'))
        
        # assing the buit-in model for later training.
        self.model = model
        
        # get the problem size (and number of dims that should SSA solve) by the number of total weights ...
        # that is going to optimize in each epoch.
        self.problem_size = self.n_dims = np.sum([np.size(w) for w in self.model.get_weights()])
        
        # create the problem for SSA to solve. lower boundary and upper boundary set to [-1, 1] to generate weights within this range.
        # The log file will save at "Results" directory.
        # For memory management, save population set to False.
        self.problem = {
            "fit_func": self.objective_function,
            "lb": [-1, ] * self.n_dims,
            "ub": [1, ] * self.n_dims,
            "log_to": "file",
            "log_file": f"{self.log_path}/{self.model_info}.log",
            "save_population": False,
        }

    def decode_solution(self, solution):
        """
        Solution (Optimizer output) is a vector (list) but neural network parameters (weights) is a nested numpy array.\\
        Therefore we should transform to be appropriate for ANN model and finnaly set the results as ANN's weight to train by.

        Arguments:
            solution {list, array-like} -- The solution of SSA optimizer 
        
        Returns:
            None
        """
        
        # Create a list of tuples containing the shape and size of each weight tensor in the model
        weight_sizes = [(w.shape, np.size(w)) for w in self.model.get_weights()]
        
        # Initialize an empty list to store the weights
        weights = []
        
        # cut the solution vector into pieces according to the size of each weight and reshape it to be appropriate for ANN model.
        # initialize the cut point to zero
        cut_point = 0
        
        # Loop through each weight tensor and decode it from the solution vector
        for ws in weight_sizes:
            # Slice the solution and reshape it to the shape of the weight
            temp = np.reshape(solution[cut_point: cut_point + ws[1]], ws[0])
            # Add the decoded weight tensor to the weights list
            weights.append(temp)
            # update the cut_point
            cut_point += ws[1]
        
        # Set the model weights to the decoded weights
        self.model.set_weights(weights)

    def prediction(self, solution, x_data):
        """
        Predict the output of x_data based on trained model.

        Arguments:
            solution (array-like): A solution obtained from SSA.
            x_data (np.array): An array containing input data.

        Returns:
            np.array: An array of predicted outputs.
        """
        
        # Decode the SSA solution and set the new weight for the model.
        self.decode_solution(solution)
        
        # Predict the output of x_data using the model.
        return self.model.predict(x_data)

    def training(self):
        """
        Trains the model using the SSA optimizer.\\
        returns the loss history, solution history, and global best fitness history.

        Returns:
            loss_history (list): List of the loss history for each iteration.
            solution_history (list): List of the best solution for each iteration.
            gbf_history (list): List of the global best fitness for each iteration.
        """
        
        # Create the model
        self.create_model()
        
        # Initialize the optimizer with the given parameters 
        self.optimizer = SSA.BaseSSA(
            self.epoch, self.pop_size,
            PD=self.PD, SD=self.SD, ST=self.ST
        )
        
        # Use the optimizer to solve the problem and get the solution and best fitness
        self.solution, self.best_fit = self.optimizer.solve(self.problem)
        
        # Save the global best fitness chart to a file
        self.optimizer.history.save_global_best_fitness_chart(
                filename=f"{self.charts_path}/{self.model_info}_gbf"
        )

        # Get the global best fitness history
        gbf_history = self.optimizer.history.list_global_best_fit

        # Get the loss history and solution history to use in the validation dataset
        loss_history = self.optimizer.history.list_global_best_fit
        solution_history = self.optimizer.history.list_global_best

        # Return the loss history, solution history, and global best fitness history
        return loss_history, solution_history, gbf_history

    def objective_function(self, solution):
        """
        Calculates the fitness of the given solution for optimizer.\\
        Use CCE loss function to calculate the performance of ANN and then optimize.

        Arguments:
            solution (numpy.ndarray): The solution to evaluate.

        Returns:
            float: The CCE loss of the model.
        """
        
        # Decode the solution and set the new weight for the model.
        self.decode_solution(solution)
        
        # Predict the output of x_data using the model.
        yhat = self.model.predict(self.X_train)
        
        # Calculate the CCE loss of the model.
        cce = CategoricalCrossentropy()
        loss = cce(self.y_train, yhat).numpy()
        
        return loss


# TODO: try-except block for path (to avoid error)
# TODO: write a test for this class.
if __name__=="__main__":
    pass       