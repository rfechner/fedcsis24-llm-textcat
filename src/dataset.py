from time import time
from typing import Union, List, Tuple
from pathlib import Path
from src.util.wzutils import timing
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
import pickle
from datasets import Dataset, Features, Value, ClassLabel, DatasetDict
import os

class TransformersDatasetWrapper():
    
    def __init__(self, path : Union[Path, str],
                data_column : str,
                label_column : str,
                label_transformation = lambda x: x,
                delimiter=';',
                drop_mask = None,
                split_with_uniform_test = False # can evaluate using macro average either way
                ):
        
        assert os.path.isfile(path)
        assert path.endswith('.csv')

        self.filename = Path(path).parts[-1].replace('.csv', '')

        # load df, filter the df for columns we want to keep
        self.df = pd.read_csv(filepath_or_buffer=path,
                    delimiter=delimiter,
                    low_memory=False,
                    usecols=[data_column, label_column])

        # drop NaN valued columns
        self.drop_mask = self.df[label_column].isna() | self.df[data_column].isna()

        # drop additional columns, if specified in args
        if drop_mask is not None:
            self.drop_mask = self.drop_mask | drop_mask

        print(f"Dropping {self.drop_mask.astype(int).sum()} out of {len(self.df)} rows.")
        self.df = self.df[~self.drop_mask].reset_index(drop=True)


        # some data sources require a transform of their label. E.g. in the case of textkernel data,
        # we need to transform the label '[organization_activity]' of form XXX.XX.XX into a WZ section.
        # a transform might look like lambda x: group2section(x.split('.')[0])
        self.df['label'] = self.df[label_column].apply(func=label_transformation)
        self.df.rename(columns={data_column : 'data'}, inplace=True)

        # test df with uniform distribution using oversampling of minority classes.
        if split_with_uniform_test:
            self.train_df, self.test_df, self.validate_df =\
                self.split_with_uniform_test(oversample=True)
        # keep data distribution
        else:
            self.train_df, self.test_df, self.validate_df =\
                self.split_with_data_distribution()

    def split_with_uniform_test(self, train_size=0.8, val_size=0.1, test_size=0.1, oversample=True, sample_random_state=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into three train/test/validate sets. The test set will have a uniform distribution
        of classes, potentially using over/undersampling.

        Parameters
        ----------
        train_size : float, optional
            The proportion of the dataset to include in the training split, default is 0.8.
        val_size : float, optional
            The proportion of the dataset to include in the validation split, default is 0.1.
        test_size : float, optional
            The proportion of the dataset to include in the test split, default is 0.1.
        oversample : bool, optional
            If True, oversampling will be used to create a uniform distribution in the test set.
            If False, undersampling will be used. Default is True.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            A triple containing train/test/validate DataFrames according to the given splits.

        Raises
        ------
        AssertionError
            If the sum of train_size, test_size, and val_size is not equal to 1.
        """

        # get dataframes with data distributions. Takes care of low-prevalence classes by matching val and test set.
        initial_train, test, val = self.split_with_data_distribution(train_size=train_size, test_size=test_size, val_size=val_size)

        # Resample test set to have uniform distribution
        if oversample:
            max_count = test['label'].value_counts().max()
            dfs_test = [resample(group, replace=True, n_samples=max_count, random_state=sample_random_state) for label, group in test.groupby('label')]
        else:
            min_count = test['label'].value_counts().min()
            dfs_test = [resample(group, replace=False, n_samples=min_count, random_state=sample_random_state) for label, group in test.groupby('label')]

        test_uniform = pd.concat(dfs_test)

        return initial_train, test_uniform, val 


    def split_with_data_distribution(self, train_size=0.8, test_size=0.1, val_size=0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into three train/test/validate sets while ensuring that the training data contains all classes.

        All classes of the data must be present in the training dataset.
        This method iterates over all groups of the DataFrame, grouped by the class label,
        then splits each individual DataFrame into train/test/validate DataFrames.
        The resulting DataFrames are concatenated and returned.

        Parameters
        ----------
        train_size : float, optional
            The proportion of the dataset to include in the training split, default is 0.8.
        test_size : float, optional
            The proportion of the dataset to include in the test split, default is 0.1.
        val_size : float, optional
            The proportion of the dataset to include in the validation split, default is 0.1.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            A triple containing train/test/validate DataFrames according to the given splits.
            It is ensured that all classes of the data are present in the training DataFrame.

        Raises
        ------
        AssertionError
            If the sum of train_size, test_size, and val_size is not equal to 1.

        See Also
        --------
        sklearn.model_selection.train_test_split : The scikit-learn function used for data splitting.

        Examples
        --------
        # Example usage:
        instance = TransformersDatasetWrapper(...)
        train_data, test_data, validate_data = instance.train_test_val_split(train_size=0.8, test_size=0.1, val_size=0.1)
        """

        assert train_size + test_size + val_size == 1

        test_val_split = test_size / (1 - train_size)
        dfs_train, dfs_test, dfs_validate = [], [], []
        groups = self.df.groupby(by='label').groups

        # iterate over groups like {'A' : [1,2,3,5], 'B' : [12, 13, 4, 42] ...}
        for key, indices in groups.items():

            sub_df = self.df.iloc[indices]

            # Edgecase: small group sizes
            if len(sub_df) == 1:
                print('Encountered class label {key} with only one sample. Training data will be added to Test/Validate split.')
                train_df, test_df, validate_df = sub_df.copy(), sub_df.copy(), sub_df.copy()
            elif len(sub_df) <= 10:
                print(f"Encountered class label {key} with <= 10 samples. Test/Validate Splits will equal.")
                train_df, test_df = train_test_split(sub_df, train_size=train_size)
                validate_df = test_df.copy()
            else:
                train_df, test_df = train_test_split(sub_df, train_size=train_size)
                test_df, validate_df = train_test_split(test_df, train_size=test_val_split)

            dfs_train.append(train_df)
            dfs_test.append(test_df)
            dfs_validate.append(validate_df)

        # all dataframes in the lists should be non-empty
        df_train = pd.concat(dfs_train)
        df_test = pd.concat(dfs_test)
        df_validate = pd.concat(dfs_validate)

        return df_train, df_test, df_validate

    def to_disk(self, path : Union[Path, str]):
        
        clean_filename = path + '.ds.pkl' if not path.endswith('.ds.pkl') else path

        with open(file=clean_filename, mode='wb') as f:
            pickle.dump(obj=self, file=f)

    @classmethod
    def from_disk(cls, path : Union[Path, str]):

        
        clean_filename = path + '.ds.pkl' if not path.endswith('.ds.pkl') else path
        assert os.path.isfile(clean_filename)

        with open(file=clean_filename, mode='rb') as f:
            return pickle.load(file=f)
        
    def plot_frequency(self):
        
        fig, (ax_train, ax_test, ax_validate) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
        fig.tight_layout()

        train_counts = self.train_df.groupby('label').count()
        test_counts = self.test_df.groupby('label').count()
        validate_counts = self.validate_df.groupby('label').count()

        ax_train.bar(x=train_counts.index, height=train_counts.iloc[:, 0])
        ax_train.set_title('Trainingset frequencies')

        ax_test.bar(x=test_counts.index, height=test_counts.iloc[:, 0])
        ax_test.set_title('Testset frequencies')

        ax_validate.bar(x=validate_counts.index, height=validate_counts.iloc[:, 0])
        ax_validate.set_title('Validationset frequencies')

    @timing
    def to_transformers(self, store_folder: Union[Path, str] = None) -> DatasetDict:
        """
        Serializes datasets.Dataset objects from the train_df, test_df, and validate_df attributes of this class
        using the Hugging Face Transformers library.

        Parameters
        ----------
        store_folder : Union[Path, str]
            The path where the serialized datasets.Dataset objects will be stored.
        data_column : str
            The column name in the DataFrame containing the input data.
        label_column : str
            The column name in the DataFrame containing the labels.

        Raises
        ------
        AssertionError
            If store_folder is not a valid directory or if data_column or label_column are not present in the DataFrame.
        
        Examples
        --------
        # Example usage:
        instance = TransformersDatasetWrapper()
        instance.to_transformers(store_folder='/path/to/store', data_column='text_column', label_column='label_column')
        """
        assert store_folder is None or os.path.isdir(store_folder)

        class_names = list(self.train_df['label'].unique())
        dsd = DatasetDict()

        for split, df in zip(['train', 'test', 'validation'],
                              [self.train_df, self.test_df, self.validate_df]):
            
            print(f'Serializing : {split} data to .transformers format.')
            
            features = Features({'text': Value('string'),
                                'label': ClassLabel(names=class_names),
                                '__index_level_0__' : Value('int64')})
            
            df_renamed = df.rename(columns={'data' : 'text', 'label' : 'label'})
            dsd[split] = Dataset.from_pandas(df_renamed[['text', 'label']],
                                            features=features,
                                             split=split)
        if store_folder:
            dsd.save_to_disk(os.path.join(store_folder, f"{self.filename}.trfds"))

        return dsd
                    





            







