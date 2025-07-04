QUANTUM MACHINE LEARNING FOR CHEMISTRY ON HPC
What is this project?
This project is a scientific computing pipeline that uses Quantum Machine Learning (QML) to improve the accuracy of computational chemistry simulations. It's designed to run on a High-Performance Computing (HPC) cluster, which is necessary for running many experiments at once.

The core idea is to combine a fast, but less accurate, classical simulation method with a flexible QML model. By training the QML model to predict the error of the classical method, we can add this prediction back to the original result to get a final answer that is much closer to the true, high-accuracy value.

The Project Goal: Delta-Learning (Δ-Learning)
In many areas of science, we have methods that are fast and cheap, but not very accurate. We also have methods that are very accurate, but are so slow and computationally expensive that we can't use them for large-scale studies.

This project uses a technique called Delta-Learning (Δ-Learning) to get the best of both worlds. Here's how it works:

We start with a cheap method (in this case, a semi-empirical quantum chemistry method called PM7) to get a baseline prediction for a property we care about (like the heat of formation of a molecule). We know this prediction has some error.

We also have a small number of high-accuracy results from an expensive method (like Density Functional Theory, or DFT).

The goal of our machine learning model is not to predict the property directly. Instead, it learns to predict the difference (the "delta" or Δ) between the cheap prediction and the true, high-accuracy value.

Δ = (True Value from DFT) - (Cheap Value from PM7)

Once the QML model is trained, we can use it on new molecules. We run the cheap PM7 calculation, and then use our QML model to predict the error (the Δ).

We "correct" our cheap prediction by adding the predicted error back to it:

Corrected Value = (Cheap Value from PM7) + (Predicted Δ)

This way, we can get near-DFT accuracy at a fraction of the computational cost.

What's in this repository?
There are four main files that run the experiments:

HPC_QML3.py: The main Python script that does all the work: loading data, building and training the QML models, and saving the results.

qml_launcher.sh: A shell script that submits many different jobs to the HPC cluster. It's great for exploring a wide range of model settings.

qml_array.sh: Another shell script that uses a SLURM "job array" to submit a pre-defined set of experiments.

QCV.py: A helpful tool for visualizing what the different quantum circuits used in this project actually look like.

And three data files:

train_df.csv & test_df.csv: The datasets used for training and testing the models.

smiles_db.csv: The original source database from which the training and test sets were created.

How to Set Up and Run This Project
Step 1: Get the Code and Data
First, you'll need to get all the project files onto the HPC cluster you're using. This includes the scripts and the data files located in the data/ directory.

Step 2: Understand the Data
The data directory contains the datasets for this project:

train_df.csv: This is the data the model learns from. It contains molecular features and the target "delta" values.

test_df.csv: This data is used to test the model after it has been trained. The model never sees this data during the training process, which gives us an honest measure of its performance.

smiles_db.csv: This is the original database. It likely contains the SMILES strings (a text representation of molecules) and other source information that were used to generate the features in the training and testing sets.

Step 3: Build the Software Container
The Python script has a lot of dependencies (squlearn, scikit-learn, etc.). The best way to manage these on an HPC cluster is with a software container. This project uses Apptainer (formerly Singularity).

A file named QML.def is included in this repository. This is a "definition file" that tells Apptainer how to build the software environment.

To build the container, run the following command on the HPC login node:

apptainer build QML.sif QML.def

This will create a single file, QML.sif, that contains Python and all the necessary libraries. The other scripts are already set up to use this file.

Step 4: IMPORTANT - Update the Directory Paths
The shell scripts (qml_launcher.sh and qml_array.sh) need to know where your project directory is located. You must edit this variable at the top of both files.

Open qml_launcher.sh and qml_array.sh and find this line:

File paths for your project.
BASE_DIR="/home/armaank/projects/def-vikikrpd/armaank"

Change the path to match the location where you cloned the project on your HPC cluster. For example:

File paths for your project.
BASE_DIR="/home/your_username/path/to/your/cloned/project"

Step 5: Run an Experiment
You are now ready to run the experiments. You don't run the Python script directly. Instead, you submit one of the .sh scripts to the SLURM scheduler.

To run a wide range of exploratory jobs, edit the settings at the top of qml_launcher.sh and then submit it:

sbatch qml_launcher.sh

To run a pre-defined set of jobs, edit the settings in qml_array.sh and then submit it:

sbatch qml_array.sh

You can check on the status of your jobs with the command squeue -u <your_username>.

Visualizing the Quantum Circuits (QCV.py)
Before running a big experiment, you might want to see what the quantum circuits actually look like. The QCV.py script is a tool that lets you do exactly that.

What it does
This script is an interactive tool that asks you which circuits you want to see, with how many qubits and layers. It then generates plots of these circuits and displays them on your screen. You can also choose to save the plots as image files (.png, .svg, or .pdf).

How to run it
This tool is meant to be run on a computer with a graphical display (your own laptop, or an HPC node with X11 forwarding).

Make sure you have the necessary Python libraries installed (like squlearn and matplotlib). The easiest way to do this is to run the tool from inside the Apptainer container you already built.

Start an interactive session inside your container. From your project directory, run:

apptainer shell QML.sif

Once you are inside the container's shell, run the script:

python QCV.py

The script will then guide you through the process, asking you which circuits, qubits, and layers you want to visualize.

Understanding the Output
When a job from qml_launcher.sh or qml_array.sh finishes, it will save its results in the Output_Files/QML_Models directory.

..._predictions.csv: A spreadsheet with the detailed predictions from your model.

..._results_plot.png: A plot showing how well the model's predictions match the true values. The closer the dots are to the red dashed line, the better the model is.

summary_log.csv: A master log file that records the final error score and settings for every experiment you run. This is very useful for comparing results later.