"""
The helper class for the training loop to do model training
"""

import torch
import torch.nn.functional as F
from jiwer import wer, cer

from modules import GreedyDecoder


class IterMeter(object):

    """
    Keeps track of the total iterations during the training and validation loop
    """
    
    def __init__(self) -> None:
        self.val = 0


    def step(self):
        self.val += 1


    def get(self):
        return self.val
    

class TrainingLoop:

    """
    The main class to set up the training loop to train the model
    """

    def __init__(self) -> None:
        pass
    

    def train(self, model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter) -> None:

        """
        Training Loop
        """
        
        model.train()
        data_len = len(train_loader.dataset)
        
        for batch_idx, _data in enumerate(train_loader):
            audio_path, spectrograms, labels, input_lengths, label_lengths = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            loss.backward()

            optimizer.step()
            iter_meter.step()
            
            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(spectrograms), data_len,
                    100. * batch_idx / len(train_loader), loss.item()))


    def dev(self, model, device, dev_loader, criterion, scheduler, epoch, iter_meter) -> None:

        """
        Validation Loop
        """
        
        print('\nevaluating...')
        model.eval()
        val_loss = 0
        test_cer, test_wer = [], []
        greedy_decoder = GreedyDecoder()
        
        with torch.no_grad():
            for i, _data in enumerate(dev_loader):
                audio_path, spectrograms, labels, input_lengths, label_lengths = _data 
                spectrograms, labels = spectrograms.to(device), labels.to(device)

                output = model(spectrograms)  # (batch, time, n_class)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1) # (time, batch, n_class)

                loss = criterion(output, labels, input_lengths, label_lengths)
                val_loss += loss.item() / len(dev_loader)

                decoded_preds, decoded_targets = greedy_decoder.decode(output.transpose(0, 1), labels=labels, label_lengths=label_lengths, is_test=False)
                print(decoded_preds)
                print(decoded_targets)
                
                for j in range(len(decoded_preds)):
                    test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                    test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

        avg_cer = sum(test_cer)/len(test_cer)
        avg_wer = sum(test_wer)/len(test_wer)
        
        scheduler.step(val_loss)

        print('Dev set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(val_loss, avg_cer, avg_wer))