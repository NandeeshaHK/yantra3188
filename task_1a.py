'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:			task_1a.py
# Functions:	    [`ideantify_features_and_targets`, `load_as_tensors`,
# 					 `model_loss_function`, `model_optimizer`, `model_number_of_epochs`, `training_function`,
# 					 `validation_functions` ]

####################### IMPORT MODULES #######################
import pandas 
import torch
import numpy
###################### Additional Imports ####################
'''
You can import any additional modules that you require from 
torch, matplotlib or sklearn. 
You are NOT allowed to import any other libraries. It will 
cause errors while running the executable
'''
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split


##############################################################

def data_preprocessing(task_1a_dataframe):
        ''' 
        Purpose:
        ---
        This function will be used to load your csv dataset and preprocess it.
        Preprocessing involves cleaning the dataset by removing unwanted features,
        decision about what needs to be done with missing values etc. Note that 
        there are features in the csv file whose values are textual (eg: Industry, 
        Education Level etc)These features might be required for training the model
        but can not be given directly as strings for training. Hence this function 
        should return encoded dataframe in which all the textual features are 
        numerically labeled.
        
        Input Arguments:
        ---
        `task_1a_dataframe`: [Dataframe]
                            Pandas dataframe read from the provided dataset 	
        
        Returns:
        ---
        `encoded_dataframe` : [ Dataframe ]
                            Pandas dataframe that has all the features mapped to 
                            numbers starting from zero

        Example call:
        ---
        encoded_dataframe = data_preprocessing(task_1a_dataframe)
        '''

        #################	ADD YOUR CODE HERE	##################
        encoded_dataframe = pandas.get_dummies(task_1a_dataframe, columns=[])
        encoded_dataframe = encoded_dataframe.replace({True: 1, False: 0})
        encoded_dataframe = encoded_dataframe.replace({'Bachelors':0,'Masters':1,'PHD':2})
        encoded_dataframe = encoded_dataframe.replace({'Yes':1,'No':0})
        encoded_dataframe = encoded_dataframe.replace({'Bangalore':0, 'Pune':1, 'New Delhi':2})
        encoded_dataframe = encoded_dataframe.replace({'Male':1,'Female':0})
        ##########################################################

        return encoded_dataframe

def identify_features_and_targets(encoded_dataframe):
        '''
        Purpose:
        ---
        The purpose of this function is to define the features and
        the required target labels. The function returns a python list
        in which the first item is the selected features and second 
        item is the target label

        Input Arguments:
        ---
        `encoded_dataframe` : [ Dataframe ]
                            Pandas dataframe that has all the features mapped to 
                            numbers starting from zero
        
        Returns:
        ---
        `features_and_targets` : [ list ]
                                python list in which the first item is the 
                                selected features and second item is the target label

        Example call:
        ---
        features_and_targets = identify_features_and_targets(encoded_dataframe)
        '''
        #################	ADD YOUR CODE HERE	##################
        # Define the list of selected features (columns in the DataFrame)
        selected_features = encoded_dataframe.columns.tolist()
        
        # Remove the target label from the list of selected features
        target_label = 'LeaveOrNot'  # Replace 'TargetColumnName' with your actual target column name
        selected_features.remove(target_label)
        # selected_features.remove('City')
        selected_features.remove('Gender')
        selected_features.remove('Age')
        selected_features.remove('JoiningYear')
        selected_features.remove('PaymentTier')    
        # selected_features.remove('EverBenched')
        # selected_features.remove('ExperienceInCurrentDomain')
        # selected_features.remove('Education')
        # Return a list containing selected features and the target label
        features_and_targets = [selected_features, target_label]
        # print(features_and_targets)
        ###########################################################
        return features_and_targets


def load_as_tensors(features_and_targets):
    ''' 
    Purpose:
    ---
    This function aims at loading your data (both training and validation)
    as PyTorch tensors. Here you will have to split the dataset for training 
    and validation, and then load them as as tensors. 
    Training of the model requires iterating over the training tensors. 
    Hence the training sensors need to be converted to iterable dataset
    object.

    Input Arguments:
    ---
    `features_and targets` : [ list ]
                            python list in which the first item is the 
                            selected features and second item is the target label

    Returns:
    ---
    `tensors_and_iterable_training_data` : [ list ]
                                            Items:
                                            [0]: X_train_tensors: Training features loaded into Pytorch tensors
                                            [1]: X_val_tensors: Feature tensors in validation data
                                            [2]: y_train_tensor: Training labels as Pytorch tensor
                                            [3]: y_val_tensor: Target labels as tensor in validation data
                                            [4]: Iterable dataset object and iterating over it in 
                                                batches, which are then fed into the model for processing

    Example call:
    ---
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets, encoded_dataframe)
    '''
    #################	ADD YOUR CODE HERE	##################

    # Extract features and target label from the input
    features, target_label = features_and_targets

    # Get the selected features from the DataFrame
    X = encoded_dataframe[features].values

    # Get the target labels from the DataFrame
    y = encoded_dataframe[target_label].values

    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=35)

    # Convert data to PyTorch tensors
    X_train_tensors = torch.FloatTensor(X_train)
    X_test_tensors = torch.FloatTensor(X_test)
    y_train_tensor = torch.FloatTensor(y_train)
    y_test_tensor = torch.FloatTensor(y_test)

    # Create a TensorDataset for training data
    train_dataset = TensorDataset(X_train_tensors, y_train_tensor)

    # Create a DataLoader for training data (iterable dataset)
    train_loader = DataLoader(train_dataset, batch_size=64)

    tensors_and_iterable_training_data = [
        X_train_tensors,
        X_test_tensors,
        y_train_tensor,
        y_test_tensor,
        train_loader
    ]
    #####################################################
    return tensors_and_iterable_training_data


class Salary_Predictor(torch.nn.Module):
        '''
        Purpose:
        ---
        The architecture and behavior of your neural network model will be
        defined within this class that inherits from nn.Module. Here you
        also need to specify how the input data is processed through the layers. 
        It defines the sequence of operations that transform the input data into 
        the predicted output. When an instance of this class is created and data
        is passed through it, the `forward` method is automatically called, and 
        the output is the prediction of the model based on the input data.
        
        Returns:
        ---
        `predicted_output` : Predicted output for the given input data
        '''
        def __init__(self,):
            super(Salary_Predictor, self).__init__()
            '''
            Define the type and number of layers
            '''
            #################	ADD YOUR CODE HERE	##################
            # Define the layers of the neural network
            self.fc1 = torch.nn.Linear(4,256)  # Input layer to hidden layer
            self.relu = torch.nn.ReLU()  # Activation function (e.g., ReLU)
            self.fc2 = torch.nn.Linear(256, 1)  # Hidden layer to output layer
            self.sig = torch.nn.Sigmoid()
            ##########################################################
        def forward(self, x):
            '''
            Define the activation functions
            '''
            #################	ADD YOUR CODE HERE	##################
            # Define the forward pass
            predicted_out = self.fc1(x)  # Pass input through the first layer
            predicted_out = self.relu(predicted_out)  # Apply activation function
            predicted_out = self.relu(predicted_out)
            predicted_out = self.fc2(predicted_out) 
            predicted_out = self.sig(predicted_out) # Pass through the second layer (output layer)
            ##########################################################
            return predicted_out

def model_loss_function():
	'''
	Purpose:
	---
	To define the loss function for the model. Loss function measures 
	how well the predictions of a model match the actual target values 
	in training data.
	
	Input Arguments:
	---
	None

	Returns:
	---
	`loss_function`: This can be a pre-defined loss function in PyTorch
					or can be user-defined

	Example call:
	---
	loss_function = model_loss_function()
	'''
	#################	ADD YOUR CODE HERE	##################
	# Define the loss function (MSE for regression)
	loss_function = torch.nn.MSELoss()
	##########################################################
	
	return loss_function

def model_optimizer(model):
    '''
    Purpose:
    ---
    To define the optimizer for the model using the Adam optimizer.
    
    Input Arguments:
    ---
    `model`: An object of the 'Salary_Predictor' class
    
    Returns:
    # selected_features.remove('Age')
    ---
    `optimizer`: Adam optimizer
    
    Example call:
    ---
    optimizer = model_optimizer(model)
    '''
    #################	ADD YOUR CODE HERE	##################
    # Define the optimizer (Adam)
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.00001)  # You can adjust the learning rate
    ##########################################################
    return optimizer

def model_number_of_epochs():
        '''
        Purpose:
        ---
        To define the number of epochs for training the model

        Input Arguments:
        ---
        None

        Returns:
        ---
        `number_of_epochs`: [integer value]

        Example call:
        ---
        number_of_epochs = model_number_of_epochs()
        '''
        #################	ADD YOUR CODE HERE	##################
        number_of_epochs = 75
        ##########################################################
        return number_of_epochs

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
    '''
    Purpose:
    ---
    Train the model for a specified number of epochs.
    
    Input Arguments:
    ---
    1. `model`: An object of the 'Salary_Predictor' class
    2. `number_of_epochs`: For training the model
    3. `tensors_and_iterable_training_data`: List containing training and validation data tensors 
                                             and an iterable dataset object of training tensors
    4. `loss_function`: Loss function defined for the model
    5. `optimizer`: Optimizer defined for the model
    
    Returns:
    ---
    trained_model: The trained model
    
    Example call:
    ---
    trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer)
    '''
    #################	ADD YOUR CODE HERE	##################
    # Unpack tensors and iterable training data
    X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, train_loader = tensors_and_iterable_training_data
    prev_prec = 0.0
    
    # Training loop
    for epoch in range(number_of_epochs):
        model.train()  # Set the model to training mode

        # Initialize variables to track loss and accuracy
        total_loss = 0.0
        correct_predictions = 0
        all_predictions = []  # To accumulate predictions for the entire batch
        
        # Iterate through mini-batches
        for batch in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            
            # Unpack the batch into input data and target labels
            input_data, target_labels = batch
            
            # Forward pass
            predictions = model(input_data)
            
            # Accumulate predictions for the entire batch
            all_predictions.append(predictions)
            
            # Compute the loss for this batch (optional)
            # loss = loss_function(predictions, target_labels)
            
            # Calculate the number of correct predictions (if it's a classification task)
            # For regression, you can omit this part
            if isinstance(loss_function, nn.CrossEntropyLoss):
                _, predicted = torch.max(predictions, 1)
                correct_predictions += (predicted == target_labels).sum().item()
        
        # Combine predictions for the entire batch into one tensor
        all_predictions = torch.cat(all_predictions)
        
        # Calculate the loss for the entire batch
        loss = loss_function(all_predictions, y_train_tensor)
        
        # Backpropagation
        loss.backward()
        
        # Update the model parameters
        optimizer.step()
        
        # Calculate total loss
        total_loss += loss.item()
        
        # Print training statistics for this epoch (optional)
        # print(f"Epoch [{epoch+1}/{number_of_epochs}] - Loss: {total_loss:.4f}")
        
        if prev_prec != total_loss:
            prev_prec = total_loss
        else:
            return model
    ########################################################

    return model


def validation_function(trained_model, tensors_and_iterable_training_data):
    '''
    Purpose:
    ---
    Evaluate the trained model on the validation dataset and calculate the regression metric (e.g., MSE).

    Input Arguments:
    ---
    1. `trained_model`: Model trained using the training function
    2. `tensors_and_iterable_training_data`: List containing training and validation data tensors 
                                             and an iterable dataset object of training tensors
    3. `loss_function`: Loss function defined for the model

    Returns:
    ---
    regression_metric: Regression metric value (e.g., MSE)

    Example call:
    ---
    mse = validation_function(trained_model, tensors_and_iterable_training_data, loss_function)
    '''
    #################	ADD YOUR CODE HERE	##################
    # Unpack tensors and iterable training data
    X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, train_loader = tensors_and_iterable_training_data

    # Set the model to evaluation mode
    trained_model.eval()

    val_dataset = TensorDataset(X_val_tensor,y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64)
    # Initialize variables for calculating the regression metric (MSE)
    total_loss = 0.0
    total_samples = 0
    all_val_predict = []

    # Iterate through mini-batches in the validation DataLoader
    with torch.no_grad():
        for batch in val_loader:
            # Unpack the batch into input data and target labels
            input_data, target_labels = batch

            # Forward pass for validation
            val_predictions = trained_model(input_data)
            
            all_val_predict.append(val_predictions)

    # Concatenate all the predictions into a single tensor
    all_val_predict = torch.cat(all_val_predict)

    _, predicted = torch.max(all_val_predict, 1)  # Get the index of the maximum value as the predicted class
    correct_predictions = (predicted == y_val_tensor).sum().item()
    # print(correct_predictions, str(len(y_val_tensor)))
    # Calculate the loss for all predictions at once
    regression_metric = loss_function(all_val_predict, y_val_tensor)
    model_accuracy = int(correct_predictions)/int((len(y_val_tensor)))
    ##########################################################
    return model_accuracy

########################################################################
########################################################################
######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########	
'''
	Purpose:
	---
	The following is the main function combining all the functions
	mentioned above. Go through this function to understand the flow
	of the script

'''

if __name__ == "__main__":

	# reading the provided dataset csv file using pandas library and 
	# converting it to a pandas Dataframe
	task_1a_dataframe = pandas.read_csv('task_1a_dataset.csv')

	# data preprocessing and obtaining encoded data
	encoded_dataframe = data_preprocessing(task_1a_dataframe)
	
	# selecting required features and targets
	features_and_targets = identify_features_and_targets(encoded_dataframe)

	# training data object
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)

	# model is an instance of the class that defines the architecture of the model
	model = Salary_Predictor()
     	
    # obtaining loss function, optimizer and the number of training epochs
	loss_function = model_loss_function()
	optimizer = model_optimizer(model)
	number_of_epochs = model_number_of_epochs()

	# training the model
	trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, 
					loss_function, optimizer)

	# validating and obtaining accuracy
	model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
	# print(f"Accuracy on the test set = {model_accuracy}")

	X_train_tensor = tensors_and_iterable_training_data[0]
	x = X_train_tensor[0]
	jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")
      
