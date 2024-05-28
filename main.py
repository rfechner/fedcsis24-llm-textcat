import argparse
import os

from src.training import train_transformers
from src.evaluate import evaluate_transformers

def train(args):

    output_path = args.out_path
    training_data_path = args.training_data_path

    assert os.path.isdir(output_path)
    assert os.path.isdir(training_data_path) 
    assert training_data_path.endswith('.trfds')

    train_transformers(
            checkpoint=args.checkpoint_path,
            output_path=output_path,
            pretrained_model_name=args.pretrained_model_name,
            training_data_path=training_data_path,
            batch_size=args.batch_size,
            num_epochs=args.nr_epochs,
            verbose=args.verbose
        )

def evaluate(args):
    
    output_path = args.out_path
    training_data_path = args.training_data_path

    assert os.path.isdir(output_path)
    assert os.path.isdir(training_data_path) and training_data_path.endswith('.trfds')

    evaluate_transformers(
            checkpoint=args.checkpoint_path,
            training_data_path=training_data_path,
            batch_size=args.batch_size,
            verbose=args.verbose,
            plot=True
        )

if __name__ == "__main__":
    print('Currently only supporting transformers library.')
    main_parser = argparse.ArgumentParser()
    
    main_parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the folder containing transformers config.json, model.safetensors etc."
    )
    main_parser.add_argument(
        "--out_path",
        type=str,
        default="./out/jp2wz08/transformers/default",
        help="Save path of the trained model and checkpoints."
    )
    main_parser.add_argument(
        "--training_data_path",
        type=str,
        default="./data/jp2wz08/transformers/default",
        help="Training/Test/Validation data path"
    )

    main_parser.add_argument(
        "--nr_epochs",
        type=int,
        default=10,
        help="Number of training epochs."
    )
    main_parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    main_parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="distilbert-base-german-cased",
        help="Path or handle of pretrained model to initialize architecture and weights from."
    )
    main_parser.add_argument(
        "--eval",
        action="store_true",
        help="Whether to only evaluate on the test split of the data provided under @param training_data_path."
    )

    main_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Whether to print more information to the screen during training."
    )
    main_parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Whether to set debugging arguments."
    )

    args = main_parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)


