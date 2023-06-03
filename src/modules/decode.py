"""
Decodes the logits into characters to form the final transciption using the greedy decoding approach
"""

import torch
from typing import List

from modules import TextTransform


class GreedyDecoder:

    """
    Decodes the logits into characters to form the final transciption using the greedy decoding approach
    """

    def __init__(self) -> None:
        pass


    def decode(
            self, 
            output: torch.Tensor, 
            labels: torch.Tensor=None, 
            label_lengths: List[int]=None, 
            collapse_repeated: bool=True, 
            is_test: bool=False
        ):
        
        """
        Main method to call for the decoding of the text from the predicted logits
        """
        
        text_transform = TextTransform()
        arg_maxes = torch.argmax(output, dim=2)
        decodes = []

        # refer to char_map_str in the TextTransform class -> only have index from 0 to 26, hence 27 represents the case where the character is decoded as blank (NOT <SPACE>)
        decoded_blank_idx = text_transform.get_char_len()

        if not is_test:
            targets = []

        for i, args in enumerate(arg_maxes):
            decode = []

            if not is_test:
                targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))

            for j, char_idx in enumerate(args):
                if char_idx != decoded_blank_idx:
                    if collapse_repeated and j != 0 and char_idx == args[j-1]:
                        continue
                    decode.append(char_idx.item())
            decodes.append(text_transform.int_to_text(decode))

        return decodes, targets if not is_test else decodes