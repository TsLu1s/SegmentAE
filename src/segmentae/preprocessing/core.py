from typing import List, Optional, Literal
import pandas as pd
from segmentae.preprocessing.simplifier import Simplifier
import warnings
warnings.filterwarnings("ignore", category=Warning) 

class Preprocessing: 
    def __init__(self, 
                 encoder : Optional[Literal["IFrequencyEncoder", "LabelEncoder", "OneHotEncoder", None]] = None, 
                 scaler: Optional[Literal["MinMaxScaler", "StandardScaler","RobustScaler", None]] = "MinMaxScaler", 
                 imputer: Optional[Literal["Simple", "RandomForest", "ExtraTrees", "GBR", 
                                           "KNN", "XGBoost", "Lightgbm", "Catboost", None]] = "Simple"):
        """
        The Preprocessing class facilitates the preprocessing of a dataset by setting up and applying encoding, scaling, and imputing techniques. 
        This class ensures that categorical and numerical data are appropriately transformed to prepare the data for machine learning models.

        Parameters:
        - encoder (str): The type of encoder to be used for categorical variables. Options include:
          - 'IFrequencyEncoder'
          - 'LabelEncoder'
          - 'OneHotEncoder'
          - None (default is None)
        - scaler (str): The type of scaler to be used for numerical variables. Options include:
          - 'MinMaxScaler'
          - 'StandardScaler'
          - None (default is 'MinMaxScaler')
        - imputer (str): The type of imputer to be used for handling missing values. Options include:
          - 'Simple'
          - 'RandomForest'
          - 'ExtraTrees'
          - 'GBR'
          - 'KNN'
          - 'XGBoost'
          - 'Lightgbm'
          - 'Catboost'
          - None (default is 'Simple')

        Attributes:
        - encoder (object): Placeholder for the encoder object that will be instantiated based on the specified encoder type.
        - scaler (object): Placeholder for the scaler object that will be instantiated based on the specified scaler type.
        - imputer (object): Placeholder for the imputer object that will be instantiated based on the specified imputer type.
        - _X (pd.DataFrame): Placeholder for the DataFrame after initial transformations.
        - cat_cols (list): List of categorical columns in the DataFrame.
        - num_cols (list): List of numerical columns in the DataFrame.
        """
        self._encoder = encoder
        self._scaler = scaler
        self._imputer = imputer
        self.imputer = None
        self.encoder = None
        self.scaler = None
        self._X=None
        self.cat_cols: List[str] = []
        self.num_cols: List[str] = []
   
    def _setup_encoder(self, X: pd.DataFrame):
        """
        Sets up the encoder based on the DataFrame's categorical columns and the specified encoder type.

        This method identifies the categorical columns in the DataFrame and applies the specified encoding technique. The transformed data is stored in the `_X` attribute.

        Parameters:
        - X (pd.DataFrame): The input DataFrame to be processed.
        """
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if self._encoder is not None and self.cat_cols:
            self.encoder = Simplifier.create_encoder(self._encoder)
            self.encoder.fit(X[self.cat_cols])
            self._X = self.encoder.transform(X).copy()
        else: self._X = X
    
    def _setup_scaler(self):
        """
        Sets up the scaler based on the DataFrame's numerical columns and the specified scaler type.

        This method identifies the numerical columns in the DataFrame and applies the specified scaling technique. The fitted scaler is stored in the `scaler` attribute.
        """
        self.num_cols = self._X.select_dtypes(include=['int', 'float']).columns.tolist()
        # If there are numerical columns, create and fit a scaler for them.
        if self._scaler is not None and self.num_cols:
            self.scaler = Simplifier.create_scaler(self._scaler)
            self.scaler.fit(self._X[self.num_cols])

    def _setup_imputer(self):
        """
        Sets up the imputer based on the DataFrame's missing values and the specified imputer type.

        This method checks for missing values in the DataFrame and applies the specified imputation technique. The fitted imputer is stored in the `imputer` attribute.
        """
        if self._imputer is not None and self._X.isnull().sum().sum() > 0:
            self.imputer = Simplifier.create_imputer(self._imputer)
            if self.imputer is not None:
                if self._imputer=="Simple":
                    self.imputer.fit(X=self._X)
                elif self._imputer!="Simple":
                    self.imputer.fit_imput(X=self._X)

    def fit(self, X: pd.DataFrame) -> 'Preprocessing':
        """
        Fits the preprocessing components (encoder, scaler, imputer) to the DataFrame.

        This method sets up the encoder, scaler, and imputer based on the DataFrame and the specified types. It prepares the preprocessing components for subsequent transformation.

        Parameters:
        - X (pd.DataFrame): The input DataFrame to be fitted.

        Returns:
        - self (Preprocessing): The instance of the Preprocessing class with fitted components.
        """
        # Set up each component by assessing the needs of the DataFrame.
        self._setup_encoder(X)
        self._setup_scaler()
        self._setup_imputer()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame using the fitted preprocessing components.

        This method applies the fitted encoder, scaler, and imputer to the DataFrame, transforming the categorical and numerical columns accordingly. The transformed data is returned.

        Parameters:
        - X (pd.DataFrame): The input DataFrame to be transformed.

        Returns:
        - X_ (pd.DataFrame): The transformed DataFrame.
        """
        # Make a copy of the DataFrame to avoid modifying the original data.
        X_ = X.copy()
        # Apply the encoder if it has been set up and is not None.
        if self.encoder is not None:
            X_ = self.encoder.transform(X_)
        # Apply the scaler to the numerical columns if it has been set up and is not None.
        if self.scaler is not None and self.num_cols:
            X_[self.num_cols] = self.scaler.transform(X_[self.num_cols].copy())
        # Apply the imputer if it has been set up and is not None.
        if self.imputer is not None:
            if self._imputer=="Simple":
                X_[self.num_cols] = self.imputer.transform(X=X_[self.num_cols].copy())
            elif self._imputer!="Simple":
                X_[self.num_cols] = self.imputer.transform_imput(X=X_[self.num_cols].copy())

        return X_


        
