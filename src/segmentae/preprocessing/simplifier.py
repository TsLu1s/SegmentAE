from mlimputer.imputation import MLimputer
from atlantic.imputers.imputation import AutoSimpleImputer
from atlantic.processing.encoders import (AutoIFrequencyEncoder,
                                         AutoLabelEncoder,
                                         AutoOneHotEncoder)
from atlantic.processing.scalers import (AutoMinMaxScaler,
                                         AutoStandardScaler,
                                         AutoRobustScaler)

class Simplifier:
    """
    A factory class for creating various types of data preprocessing components including imputers, encoders, and scalers.
    This class abstracts the creation logic, making the main Preprocessing class cleaner and focusing on logic rather than instantiation.
    """
    @staticmethod
    def create_imputer(imputer_type):
        """
        Factory method for creating an imputer based on the specified imputer type and parameters.

        :param imputer_type: A string specifying the type of imputer to create. Options include "None", "Simple", and various ML-based imputers.
        :param parameters: Optional dictionary of parameters to initialize the imputer. Defaults to None.
        :return: An instance of the specified imputer, or None if "None" is passed as the imputer_type.
        """
        if imputer_type == None:
            return None
        elif imputer_type == "Simple":
            return AutoSimpleImputer(strategy = "mean")
        else:
            return MLimputer(imput_model = imputer_type)
    
    @staticmethod
    def create_encoder(encoder_type):
        """
        Factory method for creating an encoder based on the specified encoder type.

        :param encoder_type: A string specifying the type of encoder to create. Options include "LabelEncoder" and "OneHotEncoder".
        :return: An instance of the specified encoder.
        :raises ValueError: If an unknown encoder type is specified.
        """
        if encoder_type == None:
            return None
        elif encoder_type == "IFrequencyEncoder":
            return AutoIFrequencyEncoder()
        elif encoder_type == "LabelEncoder":
            return AutoLabelEncoder()
        elif encoder_type == "OneHotEncoder":
            return AutoOneHotEncoder()
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    @staticmethod
    def create_scaler(scaler_type):
        """
        Factory method for creating a scaler based on the specified scaler type.

        :param scaler_type: A string specifying the type of scaler to create. Options include "MinMaxScaler" and "StandardScaler".
        :return: An instance of the specified scaler.
        :raises ValueError: If an unknown scaler type is specified.
        """
        if scaler_type == None:
            return None
        elif scaler_type == "MinMaxScaler":
            return AutoMinMaxScaler()
        elif scaler_type == "StandardScaler":
            return AutoStandardScaler()
        elif scaler_type == "RobustScaler":
            return AutoRobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")