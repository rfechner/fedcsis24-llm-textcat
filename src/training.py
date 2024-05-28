
import numpy as np
import evaluate
import warnings 

from datasets import DatasetDict
from sklearn.exceptions import UndefinedMetricWarning
from src.util.datasetsutils import create_mappings
from src.util.wzutils import organization_activity2wz08section

from typing import Union
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification
)
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.integrations.integration_utils import TensorBoardCallback

from src.trainer import ImbaTrainer
import torch
import os


def train_transformers(*,
        checkpoint : Union[str, Path],
        output_path : Union[str, Path],
        pretrained_model_name : str,
        training_data_path : Union[str, Path],
        batch_size : int,
        num_epochs : int,
        verbose : bool = False,
        save : bool = True):
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dsd = DatasetDict.load_from_disk(training_data_path)
    pretrained_model_name = checkpoint if checkpoint else pretrained_model_name

    if verbose:
        print('#----------- Dataset Summary --------------#')
        print(dsd, dsd['train'].features['label'].names)
        print(f'... Initializing weights from {pretrained_model_name}.')
        
    
    # ------------------------------ TOKENIZATION ------------------------------   
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    # we omit the keyword padding=True, as we want to pad dynamically
    # in training time.
    def preprocess_function(batch):
        return tokenizer(batch["text"], truncation=True)

    tokenized = dsd.map(preprocess_function, batched=True)
    
    # ------------------------------ METRICS ------------------------------
    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    

    num_labels, label2id, id2label = create_mappings(dsd['train'].features['label'])
    
    # ------------------------------ MODEL ------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    if verbose:
        print('#----------- Model Architecture --------------#')
        print(model)

    training_args = TrainingArguments(
        output_dir=output_path,
        resume_from_checkpoint=checkpoint,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        save_total_limit=5,
        gradient_accumulation_steps=3,
        log_level= 'error' # Disable default logging of raw Python dictionaries
    )
    trainer = ImbaTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[TensorBoardCallback(), EarlyStoppingCallback()]
    )

    print(f'Starting training on device: {device}...')
    
    trainer.train()
    
    # manually evaluate on test :(
    def eval_preprocess_function(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length')
    
    dataset_test = dsd['test'].map(eval_preprocess_function, batched=True)
    dataset_length = len(dataset_test)
    labels = dataset_test.features['label'].names
    num_labels = len(labels)
    model = model.to(device)

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

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
        recall_micro = recall_score(y_pred=y_pred, y_true=y_true, average='micro')
        f1_micro = f1_score(y_pred=y_pred, y_true=y_true, average='micro')
        class_report = classification_report(y_true, y_pred, target_names=labels)

    eval_string = f"[TEST]\t Accuracy: {accuracy:.2f}\t Recall: {recall_micro:.2f}\t F1: {f1_micro:.2f}" \
    + f"Class-wise Metrics:\n{class_report}" 

    print(eval_string)

    with open(os.path.join(output_path, "report.txt"), 'w') as file:
        file.write(eval_string)
    
        
