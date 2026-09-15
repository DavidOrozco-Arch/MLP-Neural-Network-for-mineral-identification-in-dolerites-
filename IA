# -*- coding: utf-8 -*-

"""
Created on Sun Apr 27 17:56:36 2025

@author: davyd
"""
import re
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
### ---------------------------------------------------------------------------
### Neural Networks
### ---------------------------------------------------------------------------
# Global model object
# This object contains a json with the information of all the RNN models
class ModelContainer:
    def __init__(self, excel_file_path):
        self.list_of_models = []
        self.date = datetime.now().strftime("%Y-%m-%d_%H%M")
        
        self.scaler_Y = StandardScaler()
        self.scaler_Z = []
        
        self.train_Y   = None
        self.train_Z   = None
        self.test_size = None
        
        self.num_features = None
        self.num_outputs  = None
        self.num_models   = None
        
        # features description
        self.features_description = pd.read_excel(excel_file_path, sheet_name='X_description', header=None)
        
        # outputs description
        # Read the Excel file without headers
        data = pd.read_excel(excel_file_path, sheet_name='Z_description', header=None)
        self.outputs_description = data.astype(str).values.flatten().tolist()
        
        # inputs and outputs
        Y = pd.read_excel(excel_file_path, 'Y', header=None).values
        Z = pd.read_excel(excel_file_path, 'Z', header=None).values
        
        # Train
        self.train(Y, Z)
        
    def add_models(self, num_features, num_outputs):
        
        for i in range(num_outputs):
            # Add Z scalers
            self.scaler_Z.append( StandardScaler() )
            
            # Models
            output_dict = {"model": [], "model_method": []}
            
            output_dict['model'].append(RNN_functional_regression(num_features=num_features, num_outputs=1, activation_method='relu').build_model())
            output_dict['model_method'].append(f'O{i}M1:RNN_functional_regression, relu')
            
            output_dict['model'].append(RNN_functional_regression(num_features=num_features, num_outputs=1, activation_method='selu').build_model())
            output_dict['model_method'].append(f'O{i}M2:RNN_functional_regression, selu')
                        
            output_dict['model'].append(RNN_sequential(num_features=num_features, num_outputs=1, activation_method='relu').build_model())
            output_dict['model_method'].append(f'O{i}M4:RNN_sequential, relu')
            
            output_dict['model'].append(RNN_sequential(num_features=num_features, num_outputs=1, activation_method='selu').build_model())
            output_dict['model_method'].append(f'O{i}M5:RNN_sequential, selu')
            
            self.list_of_models.append(output_dict)
        
        self.num_models = len( output_dict['model'] )
        
    def train(self, Y, Z, test_size=0.20, patience=100, constant_random_state=False):
        
        self.train_Y   = Y
        self.train_Z   = Z
        self.test_size = test_size        
        
        _, self.num_features = Y.shape
        _, self.num_outputs = Z.shape
        
        # Add models
        self.add_models(self.num_features, self.num_outputs)
        
        # Fit Y scaler
        _ = self.scaler_Y.fit_transform( Y )
        
        for jj, output in enumerate(self.list_of_models):
            for ii in range(0, len(self.list_of_models[0]['model'])):
                # Current output
                ii_Z = Z[:,jj]
                
                # Fit Z scaler
                if ii_Z.ndim == 1:
                    _ = self.scaler_Z[jj].fit_transform( ii_Z.reshape(-1, 1) )
                else:
                    _ = self.scaler_Z[jj].fit_transform( ii_Z )
                
                # Split data into training and validation sets
                if constant_random_state:
                    Y_train, Y_test, Z_train, Z_test = train_test_split(Y, ii_Z, test_size=test_size, random_state=42)
                else:
                    Y_train, Y_test, Z_train, Z_test = train_test_split(Y, ii_Z, test_size=test_size, random_state=42+ii)
                
                # Check proper shape of Z
                if Z_train.ndim == 1:
                    Z_train = Z_train.reshape(-1, 1)
                if Z_test.ndim == 1:
                    Z_test = Z_test.reshape(-1, 1)
    
                print(f"Trainning Model {ii}: {output['model_method'][ii]}")
                model = output['model'][ii]
                
                # Compile model
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                # Early stopping callback
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
                
                # Train
                tf.random.set_seed(42)
                history = model.fit(self.scaler_Y.transform( Y_train ), self.scaler_Z[jj].transform( Z_train ), 
                                    epochs=2000, batch_size=32*2, 
                                    validation_data=(self.scaler_Y.transform( Y_test ), self.scaler_Z[jj].transform( Z_test )), 
                                    callbacks=[early_stopping])
                
                # Plot training loss
                plot_training_history(history, output['model_method'][ii])

# -----------------------------------------------------------------------------
# RNN model definitions, only for building
class RNN_functional_regression:
    def __init__(self, num_features, num_outputs, hidden_layers_neurons_pow2_start=10, hidden_layers_neurons_pow2_end=2, activation_method='relu', double_hidden_layers=True, dropout_rate=None, batch_norm=False):
        """
        Initialize the regression model.

        :param num_features: int, number of features (input dimensions).
        :param num_outputs: int, number of outputs.
        :param hidden_layers_neurons_pow2_start: int, exponent for the first hidden layer size (default=2^10).
        :param hidden_layers_neurons_pow2_end: int, exponent for the last hidden layer size (default=2^2).
        :param activation_method: str, activation function for hidden layers.
        :param double_hidden_layers: bool, whether to add two dense layers per hidden layer configuration.
        :param dropout_rate: float or None, dropout rate to apply after each hidden layer.
        :param batch_norm: bool, whether to apply batch normalization after each hidden layer.
        """
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.activation_method = activation_method
        self.layers_config = [2**ii for ii in range(hidden_layers_neurons_pow2_start, hidden_layers_neurons_pow2_end - 1, -1)]
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.double_hidden_layers = double_hidden_layers

    def build_model(self):
        """
        Build the functional API regression model.

        :return: Keras Model instance.
        """
        # Define the input layer
        inputs = tf.keras.Input(shape=(self.num_features,), name="Input")
        x = inputs

        # Build hidden layers
        for ii, units in enumerate(self.layers_config):
            x = tf.keras.layers.Dense(units, use_bias=not self.batch_norm, name=f"Dense_{ii+1}")(x)
            
            if self.batch_norm:
                x = tf.keras.layers.BatchNormalization(name=f"BatchNorm_{ii+1}")(x)
            
            x = tf.keras.layers.Activation(self.activation_method, name=f"Activation_{ii+1}")(x)

            if self.dropout_rate:
                x = tf.keras.layers.Dropout(self.dropout_rate, name=f"Dropout_{ii+1}")(x)

            if self.double_hidden_layers:
                x = tf.keras.layers.Dense(units, use_bias=not self.batch_norm, name=f"DoubleDense_{ii+1}")(x)
                
                if self.batch_norm:
                    x = tf.keras.layers.BatchNormalization(name=f"DoubleBatchNorm_{ii+1}")(x)

                x = tf.keras.layers.Activation(self.activation_method, name=f"DoubleActivation_{ii+1}")(x)

                if self.dropout_rate:
                    x = tf.keras.layers.Dropout(self.dropout_rate, name=f"DoubleDropout_{ii+1}")(x)

        # Output layer for regression
        outputs = tf.keras.layers.Dense(self.num_outputs, activation='linear', name="Output")(x)

        # Create the model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="FunctionalRegressionModel")
        
        return model
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class RNN_sequential:
    def __init__(self, num_features, num_outputs, hidden_layers_neurons_pow2_start=10, hidden_layers_neurons_pow2_end=2, activation_method='relu', double_hidden_layers=True, dropout_rate=None, batch_norm=False):
        """
        Initialize the regression model using Sequential API.

        :param num_features: int, number of features (input dimensions).
        :param num_outputs: int, number of outputs.
        :param hidden_layers_neurons_pow2_start: int, start of neurons power of 2.
        :param hidden_layers_neurons_pow2_end: int, end of neurons power of 2.
        :param activation_method: str, activation function for hidden layers.
        :param double_hidden_layers: bool, whether to add a second identical layer after each layer.
        :param dropout_rate: float or None, dropout rate to apply after each hidden layer (if provided).
        :param batch_norm: bool, whether to apply batch normalization after each hidden layer.
        """
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.activation_method = activation_method
        self.layers_config = [2**ii for ii in range(hidden_layers_neurons_pow2_start, hidden_layers_neurons_pow2_end - 1, -1)]
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.double_hidden_layers = double_hidden_layers

    def build_model(self):
        """
        Build the Sequential API regression model.

        :return: Keras Model instance.
        """
        model = tf.keras.Sequential(name="SequentialRegressionModel")

        # Input layer        
        model.add(tf.keras.Input(shape=(self.num_features,), name="Input"))

        if self.batch_norm:
            model.add(tf.keras.layers.BatchNormalization(name="BatchNorm_Input"))
        
        model.add(tf.keras.layers.Activation(self.activation_method, name="Activation_Input"))

        if self.dropout_rate:
            model.add(tf.keras.layers.Dropout(self.dropout_rate, name="Dropout_Input"))

        # Hidden layers
        for ii, units in enumerate(self.layers_config):
            model.add(tf.keras.layers.Dense(units, use_bias=not self.batch_norm, name=f"Dense_{ii+1}"))

            if self.batch_norm:
                model.add(tf.keras.layers.BatchNormalization(name=f"BatchNorm_{ii+1}"))
            
            model.add(tf.keras.layers.Activation(self.activation_method, name=f"Activation_{ii+1}"))

            if self.dropout_rate:
                model.add(tf.keras.layers.Dropout(self.dropout_rate, name=f"Dropout_{ii+1}"))

            if self.double_hidden_layers:
                model.add(tf.keras.layers.Dense(units, use_bias=not self.batch_norm, name=f"DoubleDense_{ii+1}"))
                
                if self.batch_norm:
                    model.add(tf.keras.layers.BatchNormalization(name=f"DoubleBatchNorm_{ii+1}"))
                
                model.add(tf.keras.layers.Activation(self.activation_method, name=f"DoubleActivation_{ii+1}"))

                if self.dropout_rate:
                    model.add(tf.keras.layers.Dropout(self.dropout_rate, name=f"DoubleDropout_{ii+1}"))

        # Output layer
        model.add(tf.keras.layers.Dense(self.num_outputs, activation='linear', name="Output"))

        return model

# -----------------------------------------------------------------------------

def plot_training_history(history, title_label=''):
    # Plot training loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim(0, 1)
    plt.title(title_label)
    plt.show()
    print(history.history.keys())

def plot_testing(predicted_Z, true_Z, title_label=''):
    # Plot training loss
    plt.plot(predicted_Z, marker='o', label='Predicted')
    plt.plot(true_Z, label='True')
    plt.legend()
    plt.title(title_label)
    plt.ylim(0, 1)
    plt.show()

### ---------------------------------------------------------------------------
### Read txt file
### ---------------------------------------------------------------------------
def read_txt_file(working_path, file_name):
    # Read file
    file_path = os.path.join(working_path, file_name)
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None, None, None

    # Start variables
    mineral = None
    x = []
    y = []

    # Loop lines to fill variables
    for line in lines:
        if line.startswith('##NAMES='):
            mineral = line.split('=')[1].strip()
        elif re.match(r'^\d+.\d+,\s*\d+.\d+$', line.strip()):
            values = line.strip().split(',')
            x.append(float(values[0]))
            y.append(float(values[1]))

    # Convert x and y to column arrays
    x = np.array(x).reshape(-1, 1)
    y_array = np.array(y).reshape(-1, 1)
    if y_array.size > 0 and np.max(y_array) != 0:
        y = 100 * y_array / np.max(y_array)
    else:
        y = y_array # O maneja el caso de array vacío/cero según tu lógica, por ejemplo, y = np.zeros_like(y_array)
    return mineral, x, y

### ---------------------------------------------------------------------------
### Tailored resample
### ---------------------------------------------------------------------------
def tailored_resample(x, y, dx, start_x, end_x):
    """
    Resample x and y data to a uniform X grid,
    computing maximum y values in each bin.

    Parameters:
    -----------
    x : array-like
        Input x coordinates (can be irregularly spaced)
    y : array-like
        Input y values
    dx : float
        Step size for new uniform X vector
    start_x : float
        Starting value for new X vector
    end_x : float
        Ending value for new X vector

    Returns:
    --------
    X : numpy.ndarray
        New uniformly sampled x coordinates as a column array
    Y : numpy.ndarray
        Maximum y values for each bin (0 if empty) as a column array
    """
    # Convert inputs to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Create the new X vector
    X = np.arange(start_x, end_x + dx, dx)

    # Initialize the Y vector
    Y = np.full_like(X, np.nan, dtype=float)

    # Iterate over the new X vector
    for i in range(len(X)):
        # Find the indices of x that fall within the current range
        indices = np.where((x >= X[i] - dx / 2) & (x < X[i] + dx / 2))

        # Get the maximum y value in the current range
        if len(indices[0]) > 0:
            Y[i] = np.max(y[indices])
        else:
            Y[i] = 0  # or some other value to indicate no data in this range

    # Convert X and Y to column arrays
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    return X, Y

### ---------------------------------------------------------------------------
### Plot spectra
### ---------------------------------------------------------------------------
def plot_spectra(x, y, X, Y, dx, title_label):
    plt.figure(figsize=(10, 6))

    # Plot the original spectra as a line plot
    plt.plot(x, y, label='Original Spectra', color='blue', linewidth=0.75)

    # Plot the resampled spectra as a bar plot
    plt.bar(X.flatten(), Y.flatten(), width=dx, align='center', alpha=0.5, label='Resampled Spectra', color='orange', edgecolor='black')

    plt.xlabel('1/cm')
    plt.ylabel('Intensity')
    plt.title(title_label)
    plt.legend()
    plt.show()

### ---------------------------------------------------------------------------
### Process minerals to create output matrix
### ---------------------------------------------------------------------------
def process_minerals(minerals):
    """
    Process an array of mineral names to extract unique values and create a binary presence matrix.
    
    Parameters:
    -----------
    minerals : list or numpy array
        Array of strings representing mineral names (size Mx1)
    
    Returns:
    --------
    unique_minerals : numpy array
        Array of unique mineral names (size Nx1)
    matrix_minerals : numpy array
        Binary matrix of size MxN where matrix_minerals[i,j] = 1 if minerals[i] == unique_minerals[j], 
        otherwise 0
    """
    # Convert input to numpy array if it's not already
    minerals = np.array(minerals, dtype=str)
    
    # Get unique minerals (size Nx1)
    unique_minerals = np.unique(minerals)
    
    # Create the binary matrix (size MxN)
    M = len(minerals)
    N = len(unique_minerals)
    matrix_minerals = np.zeros((M, N), dtype=float)
    
    # Fill the binary matrix
    for ii in range(M):
        for jj in range(N):
            if minerals[ii] == unique_minerals[jj]:
                matrix_minerals[ii, jj] = 1
    
    return unique_minerals, matrix_minerals
### ---------------------------------------------------------------------------
### Create excel file
### ---------------------------------------------------------------------------
def create_excel_file(X, Y, Z, X_description, Y_description, Z_description, file_name='output.xlsx'):
    """
    Creates an Excel file with the specified sheets and data.

    Parameters:
    -----------
    X : numpy.ndarray
        Column array with size 1xN
    Y : numpy.ndarray
        Array with size MxN
    Z : numpy.ndarray
        Array with size MxU
    X_description : str
        Description for X
    Y_description : str
        Description for Y
    Z_description : list of str
        List of descriptions for Z with U values
    file_name : str, optional
        Name of the output Excel file (default is 'output.xlsx')
    """
    # Create a Pandas Excel writer using openpyxl as the engine.
    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
        # Write X to the "X" sheet
        pd.DataFrame(X.reshape(1, -1)).to_excel(writer, sheet_name='X', index=False, header=False)

        # Write Y to the "Y" sheet
        pd.DataFrame(Y).to_excel(writer, sheet_name='Y', index=False, header=False)

        # Write Z to the "Z" sheet
        pd.DataFrame(Z).to_excel(writer, sheet_name='Z', index=False, header=False)

        # Write X_description to the "X_description" sheet
        pd.DataFrame([X_description]).to_excel(writer, sheet_name='X_description', index=False, header=False)

        # Write Y_description to the "Y_description" sheet
        pd.DataFrame([Y_description]).to_excel(writer, sheet_name='Y_description', index=False, header=False)

        # Write Z_description to the "Z_description" sheet
        pd.DataFrame(Z_description).to_excel(writer, sheet_name='Z_description', index=False, header=False)

    print(f"Excel file '{file_name}' created successfully.")
# -----------------------------------------------------------------------------
# Define function for PCA
# -----------------------------------------------------------------------------
def perform_pca(Y, Z, Y_description, Z_description):
    """
    Perform PCA on input matrix Y (features) with each column of Z as a separate target
    
    Parameters:
    Y : numpy array or pandas DataFrame
        Matrix of features with shape (n_observations, n_features)
    Z : numpy array or pandas DataFrame
        Matrix of outputs with shape (n_observations, n_outputs)
        
    Returns:
    dict: Dictionary containing PCA results for each target column in Z
    """
    
    ### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ### Convert inputs to DataFrames if they aren't already
    if isinstance(Y, np.ndarray):
        Y = pd.DataFrame(Y, columns=[f'{Y_description[i].item()}' for i in range(Y.shape[1])])
    if isinstance(Z, np.ndarray):
        Z = pd.DataFrame(Z, columns=[f'{Z_description[i]}' for i in range(Z.shape[1])])
    
    ### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ### Scale features
    scaler = StandardScaler()
    Y_scaled = scaler.fit_transform(Y)
    
    ### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ### Dictionary to store results for each target
    results = {}
    
    ### Loop through each column in Z
    for target_col in Z.columns:
        print(f"\nPerforming PCA analysis for target: {target_col}")
        
        # Perform PCA
        pca = PCA(n_components=0.95)  # Keep 95% of variance
        Y_pca = pca.fit_transform(Y_scaled)
        
        # Loadings
        loadings = pd.DataFrame(pca.components_.T,
                              index=Y.columns,  # Original feature names
                              columns=[f"PC{ii}" for ii in range(pca.n_components_)])
        print(f"\nPCA loadings for {target_col} (Weights of original features):")
        print(loadings)
        
        # Create a DataFrame with the principal components
        pca_df = pd.DataFrame(data=Y_pca, 
                            columns=[f"PC{ii}" for ii in range(Y_pca.shape[1])])
        
        # Add current target column to the dataframe
        pca_df['Target'] = Z[target_col].values
        
        # Display the first few rows
        print(f"\nPCA results for {target_col}:")
        print(pca_df.head())
        
        # Store results
        results[target_col] = {
            'pca_df': pca_df,
            'loadings': loadings,
            'pca': pca
        }
    plot_pca_results(results)
    ### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return results
### ---------------------------------------------------------------------------
### Plot PCA
### ---------------------------------------------------------------------------
def plot_pca_results(results):
    """
    Create plots for PCA results for each target
    
    Parameters:
    results : dict
        Dictionary containing PCA results for each target column from perform_pca
    """
    
    # Loop through each target in the results
    for target_col, result_dict in results.items():
        pca_df = result_dict['pca_df']
        loadings = result_dict['loadings']
        abs_loadings = loadings.abs()
        
        print(f"\nGenerating plots for target: {target_col}")
        
        # Plot only first PCA
        handle_fig = plt.figure(figsize=(10, 8))
        handle_grid = gridspec.GridSpec(3, 1, height_ratios=[2, 0.1, 1])
        
        # Scatter plot of PC vs Target
        ax1 = handle_fig.add_subplot(handle_grid[0,:])
        ax1.scatter(pca_df[f'PC{0}'], pca_df['Target'], c='blue', alpha=0.5)
        ax1.set_title(f'PC{0} vs {target_col}')
        ax1.set_ylabel(f'{target_col}')
        ax1.set_xlabel(f'Principal Component {0}')
        ax1.grid(True)
        
        # Bar plot of loadings
        ax2 = handle_fig.add_subplot(handle_grid[2,:])
        abs_loadings[f'PC{0}'].plot(kind='bar', ax=ax2, title=f'PC{0} Loadings')
        ax2.set_ylabel('abs(Loadings)')
        ax2.set_xlabel('cm-1')
        # ax2.tick_params(axis='x', rotation=45)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
### ---------------------------------------------------------------------------
### Create Neural Netwroks
### ---------------------------------------------------------------------------
def create_neural_networks(working_path, RNN_file_name, dx):
    
    # List all .txt files in the specified directory
    txt_files = [file for file in os.listdir(working_path) if file.endswith('.txt')]

    # Initiate array of Y
    Y = None
    minerals = []

    # Loop all the txt files in the folder
    for file_name in txt_files:
        mineral, x, y = read_txt_file(working_path, file_name)
        if mineral is None:
            continue
        minerals.append(mineral)
        X, Y_jj = tailored_resample(x, y, dx, start_x=0, end_x=2200)
        plot_spectra(x, y, X, Y_jj, dx, title_label=f"({mineral}), {file_name}")

        if Y is None:
            Y = Y_jj
        else:
            Y = np.hstack((Y, Y_jj))
    # Get minerals matrix (outputs)
    unique_minerals, matrix_minerals = process_minerals(minerals)
    
    # Create excel file
    excel_file_name = os.path.join(working_path, 'output.xlsx')
    create_excel_file(X, Y=Y.transpose(), Z=matrix_minerals, X_description="cm-1", Y_description="intensity", Z_description=unique_minerals, file_name=excel_file_name)
    
    # Perform PCA
    perform_pca(Y.transpose(), matrix_minerals, X, unique_minerals)
    
    # Create Regression Neural Network
    RNN_container = ModelContainer(excel_file_name)
    
    # Save model
    pkl_file_name = f"{RNN_file_name}.pkl"
    with open(f"{os.path.join(working_path, pkl_file_name)}", 'wb') as file:
        pickle.dump(RNN_container, file)
    
    return pkl_file_name
# -----------------------------------------------------------------------------
def predict_neural_network(RNN_file_name, Y):
    '''
    predict
    Y : input matrix, unscaled.

    :return: unscaled prediction.
    '''
    
    # Load model
    if not RNN_file_name.endswith(".pkl"):
        RNN_file_name += ".pkl"
    with open(RNN_file_name, 'rb') as file:
        RNN_container = pickle.load(file)
    
    num_observations, _ = Y.shape
    Z = np.zeros( (num_observations, RNN_container.num_outputs, RNN_container.num_models) )
    
    for jj, output in enumerate(RNN_container.list_of_models):
        for ii in range(0, len(output['model'])):
            print(f"Computing prediction of model {output['model_method'][ii]}")
            ii_Z_scaled = output['model'][ii].predict( RNN_container.scaler_Y.transform( Y ) )
            ii_Z = RNN_container.scaler_Z[jj].inverse_transform( ii_Z_scaled )
            if Z[:,jj,ii].ndim == 1 and ii_Z.ndim == 2:
                Z[:,jj,ii] = ii_Z.ravel()  # Convert (1010,1) → (1010,)
            else:
                Z[:,jj,ii] = ii_Z
        
    Z_mdn = np.median(Z, axis=2)
    Z_avg = np.mean(Z, axis=2)
    Z_std = np.std(Z, axis=2)
    
    return Z_mdn, Z_avg, Z_std, Z, RNN_container.outputs_description
### ---------------------------------------------------------------------------
def read_RRUFF_and_predict(predict_path, RNN_file_name):
    # List all .txt files in the specified directory
    txt_files = [file for file in os.listdir(predict_path) if file.endswith('.txt')]

    # Loop all the txt files in the folder
    for file_name in txt_files:
        mineral, x, y = read_txt_file(predict_path, file_name)
        if mineral is None:
            continue
        X, Y = tailored_resample(x, y, dx, start_x=0, end_x=2200)
        plot_spectra(x, y, X, Y, dx, title_label=f"({mineral}), {file_name}")
        Z_mdn, Z_avg, Z_std, Z, Z_description = predict_neural_network(RNN_file_name, Y.transpose())
        # Get indices that sort from max to min
        sorted_indices = np.argsort(abs(Z_mdn[0]))[::-1]
        print("Here are the estimations my friend...\n(%)\tMineral\n")
        for i, ii in enumerate(sorted_indices):
            print(f"{100*abs(Z_mdn[0][ii]):.0f}\t{Z_description[ii]}")
    return
### ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Working paths
    train_path = r"C:\DAVID\Universidad\TFM\Diabasas 2018 SONIA\Diabasas 2018\base datos minerales"
    predict_path = r"C:\DAVID\Universidad\TFM\Minerales de prueba\TEST\Arqueologico_ya_revisado"
    RNN_file_name = os.path.join(train_path, "RNN_container.pkl")

    # delta x (1/cm)
    dx = 150
    
    # Train, Predict
    train_NN = True
    predict_NN = False
    
    ### -----------------------------------------------------------------------
    ### Create Neural Networks
    if train_NN:
        create_neural_networks(train_path, RNN_file_name, dx)
    
    # -------------------------------------------------------------------------
    # Predict for new Spectra
    if predict_NN:
        read_RRUFF_and_predict(predict_path, RNN_file_name)
