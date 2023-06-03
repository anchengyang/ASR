"""
Transforms text by encoding the characters and decoding the integers corresponding to the characters
"""

from typing import List


class TextTransform:

    """
    Map characters to integers and vice versa (encoding/decoding)
    """
    
    def __init__(self) -> None:

        char_map_str = """
            <SPACE> 0
            A 1
            B 2
            C 3
            D 4
            E 5
            F 6
            G 7
            H 8
            I 9
            J 10
            K 11
            L 12
            M 13
            N 14
            O 15
            P 16
            Q 17
            R 18
            S 19
            T 20
            U 21
            V 22
            W 23
            X 24
            Y 25
            Z 26
        """
        
        self.char_map = {}
        self.index_map = {}
        
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch

        self.index_map[0] = ' '


    def get_char_len(self) -> int:

        """
        Gets the number of characters that are being encoded and decoded in the prediction
        Returns:
        --------
            the number of characters defined in the __init__ char_map_str
        """

        return len(self.char_map)
    

    def get_char_list(self) -> List[str]:

        """
        Gets the list of characters that are being encoded and decoded in the prediction
        
        Returns:
        -------
            a list of characters defined in the __init__ char_map_str
        """

        return list(self.index_map.values())
    

    def text_to_int(self, text: str) -> List[int]:

        """
        Use a character map and convert text to an integer sequence 
        Returns:
        -------
            a list of the text encoded to an integer sequence 
        """
        
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)

        return int_sequence
    

    def int_to_text(self, labels) -> str:

        """
        Use a character map and convert integer labels to an text sequence 
        
        Returns:
        -------
            the decoded transcription
        """
        
        string = []
        for i in labels:
            string.append(self.index_map[i])

        return ''.join(string).replace('<SPACE>', ' ')