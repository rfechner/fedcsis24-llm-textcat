import numpy as np
import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import torch

from datasets import DatasetDict
from util.datasetsutils import create_mappings

from typing import Union
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import warnings


def visualize_transformers(*,
        checkpoint : Union[str, Path],
        sequence : str,
        verbose : bool = False,
        plot=True):
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    # Create a visualization pipeline for attention
    vis_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=model.device, output_attentions=True)

    # Get attention weights
    attention_outputs = vis_pipeline()

    # Extract attention weights from the output
    attention_weights = attention_outputs["attentions"]

    # Visualize attention weights (assuming a single-layer model for simplicity)
    plt.imshow(attention_weights[0][0].cpu().detach().numpy(), cmap="viridis", aspect="auto")
    plt.xlabel("Attention Heads")
    plt.ylabel("Tokens")
    plt.title("Attention Weights")
    plt.show()

if __name__ == '__main__':

    kwargs = {
        'checkpoint' : 'your-model-checkpoint',
        'sequence' : "Heythere buddy how are you today"
    }

visualize_transformers(**kwargs)
