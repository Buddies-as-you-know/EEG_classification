# Standard Library
from collections import defaultdict
from enum import Enum
from os.path import join
from typing import List, Tuple

# Third Party Library
import numpy as np
import pyedflib
import scipy.io
from beartype import beartype
from braindecode.datasets import BaseConcatDataset, create_from_X_y
from loguru import logger
from mspca import mspca
from sklearn.model_selection import train_test_split


class Validation(Enum):
    TDV: str = "TDV"
    DDV: str = "DDV"
    TV: str = "TV"
    SDTDV: str = "SDTDV"


@beartype
def denoizeing(data: np.ndarray) -> np.ndarray:
    denoize = mspca.MultiscalePCA()
    transposed_array = data.transpose(1, 0, 2).copy()
    reshaped_array2 = transposed_array.reshape(
        32, transposed_array.shape[1] * transposed_array.shape[2]
    )
    x_pred = denoize.fit_transform(
        reshaped_array2, wavelet_func="db4", threshold=0.5
    )
    reshaped_array3 = x_pred.reshape(
        32, transposed_array.shape[1], transposed_array.shape[2]
    )
    data = reshaped_array3.transpose(1, 0, 2)
    return data


class TrainTestDasetCreate:
    def __init__(
        self,
        data_path: str,
        edf_path: str,
        validation_name: str,
        target_date: int,
        sfreq: int,
        flip: bool,
        denoize: bool,
    ) -> None:
        """
        Initialize the TrainTestDasetCreate class.

        Args:
            data_path (str): Path to the data.
            edf_path (str): Path to the EDF files.
            validation_name (str): Name of the validation method.
            target_date (int): Target date for validation.
            sfreq (int): Sampling frequency of the data.
        """
        self.data_path: str = data_path
        self.edf_path: str = edf_path
        self.validation_name: str = validation_name
        self.target_date: int = target_date
        self.sfreq: int = sfreq
        self.channel_names: List[str] = []
        self.train_dataset: List[np.ndarray] = []
        self.train_labels: List[np.ndarray] = []
        self.test_dataset = defaultdict(list)
        self.test_labels = defaultdict(list)
        self.flip: bool = flip
        self.denoize: bool = denoize

    @beartype
    def change_mat_mne(
        self, subj: int, session: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and process the data from a MAT file and an EDF file.

        Args:
            subj (int): Subject ID.
            session (int): Session ID.

        Returns:
            Tuple[List[float], List[int]]: Processed data and labels.
        """
        mat_data = scipy.io.loadmat(
            join(
                self.data_path,
                "sub-"
                + str(subj).zfill(3)
                + "_ses-"
                + str(session).zfill(2)
                + "_task_motorimagery_eeg.mat",
            )
        )
        try:
            edf_file = join(
                self.edf_path,
                f"sub-{str(subj).zfill(3)}_ses-{str(session).zfill(2)}_task_motorimagery_eeg.edf",
            )
            edf = pyedflib.EdfReader(edf_file)
            channels = edf.getSignalLabels()
            self.channel_names = [str(ch) for ch in channels]
        except Exception as e:
            logger.warning(f"Failed to load file {edf_file}: {e}")

        data = mat_data["data"]  # EEG data
        if self.denoize is True:
            data = denoizeing(data)
        labels = mat_data["labels"][0] - 1
        # Convert channel names to a list of strings
        return data, labels

    @beartype
    def get_train_and_test_data(
        self, subj: int = None
    ) -> Tuple[
        BaseConcatDataset,
        BaseConcatDataset,
        defaultdict[int, np.ndarray],
        defaultdict[int, List[int]],
    ]:
        """
        Get the train and test datasets.

        Returns:
            Tuple[BaseConcatDataset, BaseConcatDataset]: Train and test datasets.
        """
        if self.validation_name == "SDTDV" and subj is None:
            raise ValueError(
                "Subject ID must be provided for SDTDV validation method."
            )

        match self.validation_name:
            case Validation.TDV.name:
                self.tri_dataset_verification()
            case Validation.DDV.name:
                self.dual_dataset_verification()
            case Validation.TV.name:
                self.temporal_verification()
            case Validation.SDTDV.name:
                self.subject_dependent_tri_dataset_verification(subj=subj)
            case _:
                raise ValueError("Invalid validation method.")

        if self.flip is True:
            self.train_labels = [1 - x for x in self.train_labels]

        X_train, X_val, y_train, y_val = train_test_split(
            self.train_dataset,
            self.train_labels,
            test_size=0.2,
            random_state=42,
        )

        train_windows_dataset = create_from_X_y(
            X_train,
            y_train,
            sfreq=self.sfreq,
            ch_names=self.channel_names,
            drop_last_window=False,
        )
        val_windows_dataset = create_from_X_y(
            X_val,
            y_val,
            sfreq=self.sfreq,
            ch_names=self.channel_names,
            drop_last_window=False,
        )
        return (
            train_windows_dataset,
            val_windows_dataset,
            self.test_dataset,
            self.test_labels,
        )

    def tri_dataset_verification(self) -> None:
        """
        Perform tri-dataset verification.
        """
        for subj in range(1, 26):
            for session in range(1, 6):
                data, labels = self.change_mat_mne(subj, session)
                if session == self.target_date:
                    self.test_dataset[subj].extend(data)
                    self.test_labels[subj].extend(labels)
                else:
                    self.train_dataset.extend(data)
                    self.train_labels.extend(labels)

    def dual_dataset_verification(self) -> None:
        """
        Perform dual-dataset verification.
        """
        for subj in range(1, 26):
            for session in range(1, 6):
                data, labels = self.change_mat_mne(subj, session)
                if session == self.target_date:
                    self.test_dataset[subj].extend(data[20:])
                    self.test_labels[subj].extend(labels[20:])
                    self.train_dataset.extend(data[:20])
                    self.train_labels.extend(labels[:20])
                else:
                    self.train_dataset.extend(data)
                    self.train_labels.extend(labels)

    def temporal_verification(self) -> None:
        """
        Perform temporal verification.
        """
        for subj in range(1, 26):
            data, labels = self.change_mat_mne(subj, self.target_date)
            self.test_dataset[subj].extend(data[20:])
            self.test_labels[subj].extend(labels[20:])
            self.train_dataset.extend(data[:20])
            self.train_labels.extend(labels[:20])

    def subject_dependent_tri_dataset_verification(
        self, subj: int = None
    ) -> None:
        """
        Perform subject-dependent tri-dataset verification.

        Args:
            subj (int, optional): Subject ID. Defaults to None.

        Raises:
            ValueError: If subject ID is not provided.
        """
        if subj is None:
            raise ValueError("Subject ID must be provided.")
        for session in range(1, 6):
            data, labels = self.change_mat_mne(subj, session)
            if session == self.target_date:
                self.test_dataset[subj].extend(data)
                self.test_labels[subj].extend(labels)
            else:
                self.train_dataset.extend(data)
                self.train_labels.extend(labels)


if __name__ == "__main__":
    dataset = TrainTestDasetCreate(
        data_path="/home/iplslam/EEG_Classification/data/row/mat",
        edf_path="/home/iplslam/EEG_Classification/data/row/edf",
        validation_name="TDV",
        target_date=1,
        sfreq=1000,
        flip=False,
        denoize=False,
    )
    (
        train_dataset,
        val_dataset,
        test_dataset,
        test_labels,
    ) = dataset.get_train_and_test_data(subj=10)
    print(test_dataset, test_labels)
