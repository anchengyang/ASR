"""
Entry point of the code to do model inference, also the code to use to generate the submission
"""

import torch
import torch.nn.functional as F

from time import time
from typing import Dict
import pandas as pd
from tqdm import tqdm
import os

from modules import DataProcessor, GreedyDecoder, SpeechRecognitionModel, CustomSpeechDataset

# setting the random seed for reproducibility
SEED = 2022


def infer(hparams, test_dataset, model_path) -> Dict[str, str]:
    
    print('\ngenerating inference ...')

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(SEED)
    
    greedy_decoder = GreedyDecoder()
    data_processor = DataProcessor()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=lambda x: data_processor.data_processing(x, 'test'),
        **kwargs
    )
    
    # load the pretrained model
    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], 
        hparams['n_rnn_layers'], 
        hparams['rnn_dim'],
        hparams['n_class'], 
        hparams['n_feats'], 
        hparams['stride'], 
        hparams['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    output_dict = {}
    
    with torch.no_grad():
        for i, _data in tqdm(enumerate(test_loader)):
            audio_path, spectrograms, input_lengths = _data
            spectrograms = spectrograms.to(device)
            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class) 
            decoded_preds_batch = greedy_decoder.decode(output.transpose(0, 1), labels=None, label_lengths=None, is_test=True)
            
            # batch prediction
            for decoded_idx in range(len(decoded_preds_batch[0])):
                output_dict[audio_path[decoded_idx]] = decoded_preds_batch[0][decoded_idx]
                
    print('done!\n')
    return output_dict


if __name__ == "__main__":

    # same hyperparams as what you have used to train the model
    hparams = {
            "n_cnn_layers": 3,
            "n_rnn_layers": 5,
            "rnn_dim": 512,
            "n_class": 28, # 26 alphabets in caps + <SPACE> + blanks
            "n_feats": 128,
            "stride": 2,
            "dropout": 0.1,
            "learning_rate": 1e-4,
            "batch_size": 8,
            "epochs": 100
        }

    # change the filepath as according
    SAVED_MODEL_PATH = '/home/nicholas/models/til2023/model.pt'
    SUBMISSION_PATH = '/home/nicholas/models/til2023/Submission_Advanced.csv' # or '/home/nicholas/models/til2023/Submission_Novice.csv' if novice tier

    MANIFEST_FILE_TEST = '/home/nicholas/datasets/til2023_asr_dataset/Test_Advanced.csv' # or '/home/nicholas/datasets/til2023_asr_dataset/Test_Novice.csv' if novice tier 
    AUDIO_DIR_TEST = '/home/nicholas/datasets/til2023_asr_dataset/Test_Advanced/' # or '/home/nicholas/datasets/til2023_asr_dataset/Test_Novice/' if novice tier
    
    dataset_test = CustomSpeechDataset(
        manifest_file=MANIFEST_FILE_TEST, 
        audio_dir=AUDIO_DIR_TEST, 
        is_test_set=True
    )

    start_time = time()

    submission_dict = infer(
        hparams=hparams, 
        test_dataset=dataset_test, 
        model_path=SAVED_MODEL_PATH
    )
    
    # producing the final csv file for submission
    submission_list = []

    for key in submission_dict:
        submission_list.append(
            {
                "path": os.path.basename(key),
                "annotation": submission_dict[key]
            }
        )

    submission_df = pd.DataFrame(submission_list)
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    end_time = time()

    print(f"Time taken for inference: {(end_time-start_time)/60} min")

    