# Enhanced Learning of Electroencephalography Signals for Classification

## TL;DR quickstart

```
poetry init
poetry updade
poetry shell
```

### aquire data

```
python src/data/datacollecter.py
```

- data/row/mat/ is mat file.
- data/row/edf is edf file.

### change config.yaml

```yaml
model_config:
    name: "EEGNet"
    n_chans: 32
    n_classes: 2
    n_times: 100
train_config:
    epochs: 100
    batch_size: 512
    optimizer: 
        type: AdamW
        params:
            lr: 1e-5
            weight_decay: 0.01
            amsgrad: True
    loss_fn: BinaryCrossEntropy
    scheduler:
            type: ExponentialLR
            gamma: 9e-1
    

data_dir: "./data/row/mat"
edf_dir: "./data/row/edf"
validation_name: "TDV" # you want to check validation.
sfreq: 1000
output_dir: "results/"
wandb_project: "" # For data save in wandb
wandb_entity: ""
```

```
python src/main.py  --validation_name "TDV"
```

### EEG Experiment Code README

This README provides an overview and instructions for the EEG data machine learning experiment code. The purpose of this code is to compare and analyze the performance of different EEG verification methods (TDV, DDV, TV, SDTDV). It includes details on each method, setup procedures, and execution steps.

#### Prerequisites

- Python execution environment
- Installation of required libraries;python = ">=3.9,\<3.12"
  - This project uses `poetry` for dependency management.
  - Run `poetry install` in the project's root directory to install the necessary packages.

#### Configuration

- Configuration such as `model_config` and `train_config` is managed using `hydra`.
- Create a configuration file like the following to define parameters for the model and training:

```yaml
model_config:
    name: "EEGConfomer"
    n_chans: 32
    n_classes: 2
    n_times: 1000
    sfreq: 1000

train_config:
    epochs: 100
    batch_size: 512
    optimizer: 
        type: AdamW
        params:
            lr: 1e-5
            weight_decay: 0.01
            amsgrad: True
    loss_fn: BinaryCrossEntropy
    scheduler:
            type: ExponentialLR
            gamma: 9e-1
    
flip: True <- label flip
data_dir: "./data/row/mat"
edf_dir: "./data/row/edf"
validation_name: "TDV"
subject_number: 1
sfreq: 1000
output_dir: "results/"
wandb_projrct: "EEG"
wandb_name: "WANDB"
```

#### Execution

- After preparing the configuration file, run `main.py` to train and evaluate the model.
- Use the command `python main.py` to execute the script.

#### Experimental Methods

1. **TDV (Tri-Dataset Verification)**: Train on data from 4 days and test on data from the remaining day.
1. **DDV (Dual-Dataset Verification)**: Similar to TDV, but with additional training using the first 20% of test-day data.
1. **TV (Temporal Verification)**: Train on the first 20% of test-day data and evaluate the remaining 80%.
1. **SDTDV (Subject Dependent Tri-Dataset Verification)**: Build the model using only each subject's data.

#### Notes

- The specific settings for experiments may vary depending on the user's environment and data.
- WandB is used for logging experiments. Adjust WandB settings as needed.

This code is specialized for evaluating EEG data verification methods. Adjust the settings according to the objectives of your experiment.

Certainly! Here's how you can include instructions for running the `datacollector.py` script in your README file in English:

______________________________________________________________________

### Data Collection with `datacollector.py`

To gather data using the `datacollector.py` script, follow these steps:

#### Prerequisites

- Ensure your Python environment is properly set up.
- Verify that any project-specific dependencies are installed.

#### Locating the Script

- Confirm that `datacollector.py` is located in the `src/data/` directory.

#### Running the Script

1. **Open Command Line or Terminal**:

   - Open your command line interface or terminal.

1. **Navigate to Project Root**:

   - Change the directory to the root of the project where `datacollector.py` is located.

1. **Execute the Script**:

   - Run the following command:
     ```
     python src/data/datacollector.py
     ```
   - This command initiates the data collection process by executing the specified Python script.

1. **Monitoring Log and Output**:

   - As the script runs, observe the logs in the terminal for any immediate feedback.
   - Check the script’s output and any data files generated, as necessary.

1. **Troubleshooting**:

   - If you encounter any errors, carefully read the error messages to troubleshoot and resolve them.
   - For issues related to data collection or processing, refer to the relevant sections in the script.

#### Note

- The specific functionality of `datacollector.py` depends on the script's content. What data it collects and how it processes the data will vary based on the script's logic.
- Data collection can be time-consuming. Monitor the progress and be patient if necessary.
- Follow any project-specific configurations or requirements.

Include these steps in your README file to provide clear instructions for using `datacollector.py` in your project.
- I am migrating from jupyter note book. We conducted the experiments in our paper in notebook/Nature_analaysis2.ipynb.
