"""
building the model with adaption of deepspeech2 -> https://arxiv.org/abs/1512.02595

code adapted from https://towardsdatascience.com/customer-case-study-building-an-end-to-end-speech-recognition-model-in-pytorch-with-assemblyai-473030e47c7c
"""

import torch
import torch.nn.functional as F


class CNNLayerNorm(torch.nn.Module):
    
    """
    Layer normalization built for CNNs input
    """
    
    def __init__(self, n_feats: int) -> None:
        super(CNNLayerNorm, self).__init__()

        self.layer_norm = torch.nn.LayerNorm(n_feats)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x of dimension -> (batch, channel, feature, time)
        """
        
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)

        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 


class ResidualCNN(torch.nn.Module):

    """
    Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf except with layer norm instead of batch norm
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, dropout: float, n_feats: int) -> None:
        super(ResidualCNN, self).__init__()

        self.cnn1 = torch.nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = torch.nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Model building for the Residual CNN layers
        
        Input x of dimension -> (batch, channel, feature, time)
        """

        residual = x
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual

        return x # (batch, channel, feature, time)


class BidirectionalGRU(torch.nn.Module):

    """
    The Bidirectional GRU composite code block which will be used in the main SpeechRecognitionModel class
    """
    
    def __init__(self, rnn_dim: int, hidden_size: int, dropout: int, batch_first: int) -> None:
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = torch.nn.GRU(
            input_size=rnn_dim, 
            hidden_size=hidden_size,
            num_layers=1, 
            batch_first=batch_first, 
            bidirectional=True
        )
        self.layer_norm = torch.nn.LayerNorm(rnn_dim)
        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Transformation of the layers in the Bidirectional GRU block
        """

        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)

        return x


class SpeechRecognitionModel(torch.nn.Module):

    """
    The main ASR Model that the main code will interact with
    """
    
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1) -> None:
        super(SpeechRecognitionModel, self).__init__()
        
        n_feats = n_feats//2
        self.cnn = torch.nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = torch.nn.Sequential(*[
            ResidualCNN(
                in_channels=32, 
                out_channels=32, 
                kernel=3, 
                stride=1, 
                dropout=dropout, 
                n_feats=n_feats
            ) for _ in range(n_cnn_layers)
        ])
        self.fully_connected = torch.nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = torch.nn.Sequential(*[
            BidirectionalGRU(
                rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                hidden_size=rnn_dim, 
                dropout=dropout, 
                batch_first=i==0
            ) for i in range(n_rnn_layers)
        ])
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(rnn_dim, n_class)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Transformation of the layers in the ASR model block
        """

        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        
        return x