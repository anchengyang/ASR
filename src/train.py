"""
Entry point of the code to do model training
"""

import os
import torch
from time import time

from modules import DataProcessor, TextTransform, IterMeter, SpeechRecognitionModel, TrainingLoop, CustomSpeechDataset

# setting the random seed for reproducibility
SEED = 2022


def main(hparams, train_dataset, dev_dataset, saved_model_path) -> None:

    """
    The main method to call to do model training
    """ 

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(SEED)
    
    data_processor = DataProcessor()
    iter_meter = IterMeter()
    text_transform = TextTransform()
    trainer = TrainingLoop()
    
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        collate_fn=lambda x: data_processor.data_processing(x, 'train'),
        **kwargs
    )
    
    dev_loader = torch.utils.data.DataLoader(
        dataset=dev_dataset,
        batch_size=hparams['batch_size'],
        shuffle=False,
        collate_fn=lambda x: data_processor.data_processing(x, 'dev'),
        **kwargs
    )

    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], 
        hparams['n_rnn_layers'], 
        hparams['rnn_dim'],
        hparams['n_class'], 
        hparams['n_feats'], 
        hparams['stride'], 
        hparams['dropout']
    ).to(device)

    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = torch.optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = torch.nn.CTCLoss(blank=text_transform.get_char_len()).to(device)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=3, verbose=True, factor=0.05)
    
    for epoch in range(1, hparams['epochs'] + 1):
        trainer.train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter)
        trainer.dev(model, device, dev_loader, criterion, scheduler, epoch, iter_meter)
        
    # save the trained model
    torch.save(model.state_dict(), saved_model_path)


if __name__ == "__main__":

    MANIFEST_FILE_TRAIN = '/home/nicholas/datasets/til2023_asr_dataset/Train.csv'
    AUDIO_DIR_TRAIN = '/home/nicholas/datasets/til2023_asr_dataset/Train'
    SAVED_MODEL_PATH = '/home/nicholas/models/til2023/model.pt'

    # simple check on the saved model path, will raise error if no directory found
    if not os.path.exists(os.path.dirname(SAVED_MODEL_PATH)):
        raise FileNotFoundError

    # loads the dataset
    dataset = CustomSpeechDataset(
        manifest_file=MANIFEST_FILE_TRAIN, 
        audio_dir=AUDIO_DIR_TRAIN, 
        is_test_set=False
    )

    # train_dev_split
    train_proportion = int(0.8 * len(dataset))
    dataset_train = list(dataset)[:train_proportion]
    dataset_dev = list(dataset)[train_proportion:]

    
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

    start_time = time()

    # start training the model
    main(
        hparams=hparams, 
        train_dataset=dataset_train, 
        dev_dataset=dataset_dev, 
        saved_model_path=SAVED_MODEL_PATH
    )
    
    end_time = time()
    
    print(f"Time taken for training: {(end_time-start_time)/(60*60)} hrs")
    