"""
Data preprocessing and transformation of the audio files into melspectrogram
"""

import torch
import torchaudio

from modules import TextTransform


class DataProcessor:

    """
    Transforms the audio waveform tensors into a melspectrogram
    """

    def __init__(self) -> None:
        pass
    
    
    def _audio_transformation(self, is_train: bool=True):

        return torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
                torchaudio.transforms.TimeMasking(time_mask_param=100)
            ) if is_train else torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
    

    def data_processing(self, data, data_type='train'):

        """
        Process the audio data to retrieve the spectrograms that will be used for the training
        """

        text_transform = TextTransform()
        spectrograms = []
        input_lengths = []
        audio_path_list = []

        audio_transforms = self._audio_transformation(is_train=True) if data_type == 'train' else self._audio_transformation(is_train=False)

        if data_type != 'test':  
            labels = []
            label_lengths = []

            for audio_path, waveform, utterance in data:

                spec = audio_transforms(waveform).squeeze(0).transpose(0, 1)
                spectrograms.append(spec)
                label = torch.Tensor(text_transform.text_to_int(utterance))
                labels.append(label)
                input_lengths.append(spec.shape[0]//2)
                label_lengths.append(len(label))

            spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
            return audio_path, spectrograms, labels, input_lengths, label_lengths

        else:
            for audio_path, waveform in data:

                spec = audio_transforms(waveform).squeeze(0).transpose(0, 1)
                spectrograms.append(spec)
                input_lengths.append(spec.shape[0]//2)
                audio_path_list.append(audio_path)

            spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
            return audio_path_list, spectrograms, input_lengths