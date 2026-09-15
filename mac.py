# -*- coding: utf-8 -*-

"""
Created on Sun Apr 27 17:56:36 2025

@author: davyd
"""
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
### Plot spectra comparison
### ---------------------------------------------------------------------------
def plot_spectra_comparison(x, y, train_y, title_label):
    plt.figure(figsize=(10, 6))

    # Plot the training spectra
    plt.plot(x.ravel(), train_y[0,:].ravel(), label='Entrenamiento', color='red', linewidth=0.75, alpha=0.5)
    for ii in range(1, train_y.shape[0]):
        plt.plot(x.ravel(), train_y[ii,:].ravel(), color='red', linewidth=0.75, alpha=0.5)

    # Plot the original spectra as a line plot
    plt.plot(x.ravel(), y.ravel(), label='Nuevo Espectro', color='blue', linewidth=1.25)

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
### ---------------------------------------------------------------------------
### Modal Assurance Criteria
### ---------------------------------------------------------------------------
def get_mac(phi_i, phi_j):
    """
    Compute Modal Assurance Criterion (MAC) between two vectors.
    
    Parameters:
    phi_i, phi_j : numpy arrays (vectors)
    
    Returns:
    float : MAC value between 0 and 1
    """
    numerator = np.abs(np.dot(phi_i.conj(), phi_j))**2
    denominator = np.dot(phi_i.conj(), phi_i) * np.dot(phi_j.conj(), phi_j)
    
    return numerator / denominator
### ---------------------------------------------------------------------------
### Process text files in reference folder
### ---------------------------------------------------------------------------
def process_reference_folder(working_path, dx):
    
    # Get only subdirectories (not files)
    subfolders = [f for f in os.listdir(working_path) if os.path.isdir(os.path.join(working_path, f))]
    
    # Create directory
    spectra_directory = {}
    
    # List all .txt files in the specified directory
    for subfolder in subfolders:
        # Get cathegory names
        category_name = re.sub(r'[^a-zA-Z0-9_]', '', subfolder)
        new_path = os.path.join(working_path, subfolder)
        
        # Get text files
        txt_files = [file for file in os.listdir(new_path) if file.endswith('.txt')]
    
        # Loop all the txt files in the folder
        for file_name in txt_files:
            mineral, x, y = read_txt_file(new_path, file_name)
            if mineral is None:
                continue
            X, Y_jj = tailored_resample(x, y, dx, start_x=0, end_x=2200)
            plot_spectra(x, y, X, Y_jj, dx, title_label=f"({mineral}), {file_name}")
            
            # Update spectra directory
            spectra_directory[len(spectra_directory)] = [X.ravel(), Y_jj.ravel(), mineral, category_name]
                
    return spectra_directory
### ---------------------------------------------------------------------------
### Process text files in predict folder
### ---------------------------------------------------------------------------
def process_predict_folder(working_path, dx):
    
    # Create directory
    spectra_directory = {}
    
    # Get text files
    txt_files = [file for file in os.listdir(working_path) if file.endswith('.txt')]

    # Loop all the txt files in the folder
    for file_name in txt_files:
        mineral, x, y = read_txt_file(working_path, file_name)
        if mineral is None:
            continue
        X, Y_jj = tailored_resample(x, y, dx, start_x=0, end_x=2200)
        plot_spectra(x, y, X, Y_jj, dx, title_label=f"({mineral}), {file_name}")
        
        # Update spectra directory
        spectra_directory[len(spectra_directory)] = [X.ravel(), Y_jj.ravel(), file_name]
                
    return spectra_directory
### ---------------------------------------------------------------------------
def predict_using_MAC(ref_spectra_directory, new_spectra_directory):
    # Loop all the new spectra
    for _, ii_new in enumerate(new_spectra_directory):
        best_mac = 0
        
        for _, ii_ref in enumerate(ref_spectra_directory):
                ii_mac = get_mac(new_spectra_directory[ii_new][1], ref_spectra_directory[ii_ref][1])
                if ii_mac > best_mac:
                    best_mac = ii_mac
                    best_ref = ii_ref
        print(f"Best match for '{new_spectra_directory[ii_new][-1]}': {best_mac:.2f} ({ref_spectra_directory[best_ref][-2]})")
    return
### ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Working paths
    reference_path = r"C:\Users\Mateo\Desktop\TFM_david\TFM_v02_MAC\references"
    predict_path = r"C:\Users\Mateo\Desktop\TFM_david\TFM_v02_MAC\prediction"

    # delta x (1/cm)
    dx = 50
    
    ### -----------------------------------------------------------------------
    ### Process reference text files
    ref_spectra_directory = process_reference_folder(reference_path, dx)
    
    ### -----------------------------------------------------------------------
    ### Process text files to predict
    new_spectra_directory = process_predict_folder(predict_path, dx)
    
    
