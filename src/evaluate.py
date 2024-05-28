import numpy as np
import torch

from datasets import DatasetDict

from typing import Union
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import os 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import warnings

from src.util.datasetsutils import create_mappings

def evaluate_transformers(*,
        checkpoint : Union[str, Path],
        training_data_path : Union[str, Path],
        batch_size : int,
        verbose : bool = False,
        plot=True,
        save=True):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dsd = DatasetDict.load_from_disk(training_data_path)

    if verbose:
        print('#----------- Dataset Summary --------------#')
        print(dsd, dsd['train'].features['label'].names)
        print(f'... Initializing weights from checkpoint {checkpoint}.')
        
    
    # ------------------------------ TOKENIZATION ------------------------------   
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    num_labels, label2id, id2label = create_mappings(dsd['train'].features['label'])
    
    # ------------------------------ MODEL ------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels, id2label=id2label, label2id=label2id
    ).to(device)

    if verbose:
        print('#----------- Model Architecture --------------#')
        print(model)
    
    # ------------------------------ EVALULATION ------------------------------
    def eval_preprocess_function(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length')
    
    dataset_test = dsd['test'].map(eval_preprocess_function, batched=True)
    dataset_length = len(dataset_test)
    labels = dataset_test.features['label'].names

    y_true = []
    y_pred = []

    for i in tqdm(range(0, dataset_length, batch_size)):

        batch = dataset_test[i:i+batch_size]
        input_ids = torch.tensor(batch['input_ids']).to('cuda')
        attention_mask = torch.tensor(batch['attention_mask']).to('cuda')
        cur_labels = torch.tensor(batch['label']).to('cuda')

        logits = model.forward(input_ids=input_ids, attention_mask=attention_mask).logits
        pred = torch.argmax(logits, dim=1)
        
        y_pred.append(pred.cpu().numpy())
        y_true.append(cur_labels.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)

        accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
        recall_micro = recall_score(y_pred=y_pred, y_true=y_true, average='micro')
        f1_micro = f1_score(y_pred=y_pred, y_true=y_true, average='micro')
        class_report = classification_report(y_true, y_pred, target_names=labels)

    eval_string = f"""
    Testing checkpoint {checkpoint}
    On dataset: {training_data_path}
    [TEST]\t Accuracy: {accuracy:.2f}\t Recall: {recall_micro:.2f}\t F1: {f1_micro:.2f}\n\n
    Class-wise Metrics:
    {class_report}
    """
    print(eval_string)
    if save:
        with open(os.path.join(checkpoint, "report.txt"), 'w') as file:
            file.write(eval_string)

    if plot:

        # confusion matrix
        cmatrix = confusion_matrix(y_true=y_true, y_pred =y_pred, labels=list(range(num_labels)))

        # accuracy in percentages
        row_sum = cmatrix.sum(axis=1)
        zero_i = (row_sum == 0)
        div = row_sum + np.ones_like(row_sum) * zero_i.astype(int)

        cmatrix_ = cmatrix / div[:, None] # broadcasting the right way, in order to divide
        cmatrix_.sum(axis=1)


        # Create a Plotly heatmap
        trace = go.Heatmap(z=cmatrix_[::-1],
                        x=labels,
                        y=labels[::-1],
                        hoverongaps=False,
                        hovertemplate='%{z:.2f}%, true: %{y}, pred: %{x}',
                        colorscale='viridis')

        layout = go.Layout(title=f'Confusion Matrix checkpoint: {checkpoint}',
                        xaxis=dict(title='Predicted Label'),
                        yaxis=dict(title='True Label'),
                        width=700,  # Adjust the width of the figure
                        height=700)

        fig = go.Figure(data=[trace], layout=layout)
        
        if save:
            fig.write_html(os.path.join(checkpoint, 'confusion.html'))
        
        fig.show()

        # sorted by class accuracy
        accuracies = cmatrix_.diagonal()

        # Create a DataFrame
        data = pd.DataFrame({"Class": labels, "Accuracy": accuracies})

        # Sort the DataFrame by accuracy in descending order
        data = data.sort_values(by="Accuracy", ascending=False)

        # Create an interactive bar plot with Plotly Express
        fig = px.bar(data, x="Class", y="Accuracy", text="Accuracy", title=f"TP per class checkpoint: {checkpoint}",
                    labels={"Accuracy": "Accuracy", "Class": "Class Label"},
                    color="Accuracy", color_continuous_scale="Viridis")

        # Customize layout and display the plot
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_layout(xaxis_title="Class Label", yaxis_title="Accuracy", coloraxis_colorbar=dict(title="Accuracy"))
        
        if save:
            fig.write_html(os.path.join(checkpoint, 'accuracies.html'))
        
        fig.show()

if __name__ == '__main__':
    kwargs = {
        'checkpoint' : "out/jp2wz08/baseline/checkpoint-710",
        'training_data_path' : 'data/human_annotated_jp2wz08/transformers/merged_jobpostings.trfds',
        'batch_size' : 8,
        'verbose' :  False,
        'plot' : True
    }

    evaluate_transformers(**kwargs)