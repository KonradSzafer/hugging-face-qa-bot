from typing import List


def find_max_split_index(text: str, char: str) -> int:
    char_idx = text.rfind(char)
    if char_idx > 0:
        # If a character is found, return the index after the splitting character
        split_idx = char_idx + len(char)
        if split_idx >= len(text):
            return len(text)
        else:
            return split_idx
    return -1


def find_max_split_index_from_sequence(text: str, split_characters: List[str]) -> int:
    split_index = max((
        find_max_split_index(text, sequence)
        for sequence in split_characters
    ), default=-1)
    return split_index


def split_text_into_chunks(
    text: str,
    split_characters: List[str] = [],
    min_size: int = 20,
    max_size: int = 250,
    ) -> List[str]:

    chunks = []
    start_idx = 0
    end_idx = max_size
    text_len = len(text)
    while start_idx < text_len:
        search_chunk = text[start_idx+min_size:end_idx]
        split_idx = find_max_split_index_from_sequence(
            text=search_chunk,
            split_characters=split_characters
        )
        # if no spliting element found, set the maximal size
        if split_idx < 1:
            split_idx = end_idx
        # if found - offset it by the starting idx of the chunk
        else:
            split_idx += start_idx + min_size

        chunk = text[start_idx:split_idx]
        chunks.append(chunk)

        start_idx = split_idx
        end_idx = split_idx + max_size

    return chunks
