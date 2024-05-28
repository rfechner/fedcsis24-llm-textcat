import torch
import transformers
import numpy as np
import warnings
import os
import ollama


from tqdm import tqdm
from typing import Union
from pathlib import Path
from datasets import DatasetDict
from src.prompts.prompt_strategy import PromptStrategy
from src.util.datasetsutils import create_mappings
from src.util.llm_utils import extract_json_from_string
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def evaluate_ollama_llm(
        model_id : str,
        pstrat : str,
        ptype : str,
        training_data_path : Union[str, Path],
        out_path : Union[str, Path],
        verbose=True,
        plot=True,
        save=True):
    
    dsd = DatasetDict.load_from_disk(training_data_path)

    if verbose:
        print('#----------- Dataset Summary --------------#')
        print(dsd, dsd['test'].features['label'].names)
        print(f'... Initializing model from model_id {model_id}.')

    # ------------------------------ PROMPT STRATEGY ---------------------   
    num_labels, label2id, id2label = create_mappings(dsd['test'].features['label'])
    labels = dsd['test'].features['label'].names
    ps = PromptStrategy.create_from_string(pstrat, ptype=ptype, label2id=label2id, id2label=id2label, valid_classes=labels)

    # ------------------------------ EVALUATION ------------------------------
    y_true = []
    y_pred = []
    failed_sample_indeces = []

    #debug_list = zip(dsd['test'][:5]['text'], dsd['test'][:5]['label'])
    for i, entry in tqdm(enumerate(dsd['test'].iter(1))):
        
        extract_success = True
        text, label = entry['text'][0], entry['label'][0] 
        system_msg, usr_msgs = ps.get_messages(text, label)
        
        messages = [
            {"role": "system", "content": system_msg},
        ]
        
        for msg, prompt_manager in usr_msgs:    
            messages.append(
                {"role" : "user", "content" : msg}
            )
            try:
                outputs = prompt_manager.run_prompt(messages, model_id)
            except ValueError:
                # model output faulty. Skip this entry.
                extract_success = False
                break

            assistant_response = outputs['message']['content']
            messages.append({"role" : "assistant", "content" : assistant_response})
        
        if not extract_success:
            # model output faulty.
            print(f'Unable to parse model output at index {i}. Continuing.')
            failed_sample_indeces.append(i)
            continue
        
        # at this point, we're sure that we are able to extract ypred,
        # as we've validated the final output in `prompt_manager.run_prompt`
        result = extract_json_from_string(assistant_response)
        prediction = ps.extract_ypred_from_json(result)

        y_pred.append(prediction)
        y_true.append(label)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)

        accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
        recall_micro = recall_score(y_pred=y_pred, y_true=y_true, average='micro')
        f1_micro = f1_score(y_pred=y_pred, y_true=y_true, average='micro')

        try:
            labels = [id2label[idx] for idx in set(y_true).union(set(y_pred))]
            class_report = classification_report(y_true, y_pred, target_names=labels)
        except:
            class_report = "Couldn't compute class report, as we encountered a key error. This is most likely due to a mis-prediction."

    eval_string = f"""
    Testing model_id {model_id} with prompt strategy: {ps}
    On dataset: {training_data_path}.
    Failed {len(failed_sample_indeces)} samples in total: {failed_sample_indeces} (indices)
    [TEST]\t Accuracy: {accuracy:.2f}\t Recall: {recall_micro:.2f}\t F1: {f1_micro:.2f}\n\n
    Class-wise Metrics:
    {class_report}
    """
    print(eval_string)
    if save:
        with open(os.path.join(out_path, "report.txt"), 'w') as file:
            file.write(eval_string)
            

if __name__ == '__main__':
    evaluate_ollama_llm(
        model_id="llama3",
        pstrat='ec',
        ptype='zero_shot',
        training_data_path='data/human_annotated_jp2wz08/transformers/merged_jobpostings.trfds',
        out_path = "out/jp2wz08/default",
        verbose=True,
        plot=False,
        save=False
    )
