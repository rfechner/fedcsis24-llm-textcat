from datasets import ClassLabel
from typing import Dict, Tuple

def create_mappings(c : ClassLabel) -> Tuple[Dict[str, int], Dict[int, str]]:

    label2id = {s : c.str2int(s) for s in c.names}
    id2label = {i : c.int2str(i) for i in range(c.num_classes)}

    return len(label2id), label2id, id2label