import json
import os

# Legacy method! Do not manually prepend white space to the sentences.
# Instead, use tokenizer's configuration to handle this.
def load_sentences(
        data_path: str,
        with_indent: bool):
    """
    Given the data path, load the sentences

    Parameters
    ----------
    data_path: str
    with_indent: bool
        if add indent to the sentences

    Return
    ----------
    sentences: list[sentence_1,sentence_2, ... sentence_N]
                list of player word id, index begins with 0,
    """
    assert data_path.endswith('.txt')
    with open(data_path,'r') as f:
        sentences = f.readlines()
        if with_indent:
            sentences = [' ' + sentence.strip() for sentence in sentences]
        else:
            sentences = [sentence.strip() for sentence in sentences]
    return sentences

