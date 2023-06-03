"""
Custom Speech Dataset class to load the dataset
"""

import os
import pandas as pd
from typing import Tuple
import torch
import torchaudio


class CustomSpeechDataset(torch.utils.data.Dataset):
    
    """
    Custom torch dataset class to load the dataset 
    """
    
    def __init__(self, manifest_file: str, audio_dir: str, is_test_set: bool=False) -> None:

        """
        manifest_file: the csv file that contains the filename of the audio, and also the annotation if is_test_set is set to False
        audio_dir: the root directory of the audio datasets
        is_test_set: the flag variable to switch between loading of the train and the test set. Train set loads the annotation whereas test set does not
        """

        self.audio_dir = audio_dir
        self.is_test_set = is_test_set

        self.manifest = pd.read_csv(manifest_file)

        
    def __len__(self) -> int:
        
        """
        To get the number of loaded audio files in the dataset
        """

        return len(self.manifest)
    
    
    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:

        """
        To get the values required to do the training
        """

        if torch.is_tensor(index):
            index.tolist()
            
        audio_path = self._get_audio_path(index)
        signal, sr = torchaudio.load(audio_path)
        
        if not self.is_test_set:
            annotation = self._get_annotation(index)
            return audio_path, signal, annotation
        
        return audio_path, signal
    
    
    def _get_audio_path(self, index: int) -> str:

        """
        Helper function to retrieve the audio path from the csv manifest file
        """
        
        path = os.path.join(self.audio_dir, self.manifest.iloc[index]['path'])

        return path
    
    
    def _get_annotation(self, index: int) -> str:

        """
        Helper function to retrieve the annotation from the csv manifest file
        """

        return self.manifest.iloc[index]['annotation']