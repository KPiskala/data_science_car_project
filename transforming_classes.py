# -*- coding: utf-8 -*-
"""
@author: Katarzyna Piskała
"""

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class BrandModelEncoder(TransformerMixin):
    '''
    Create categorical columns for most common brands and models.
    '''
    
    def __init__(self, common_brands, common_brands_models, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.common_brands = common_brands
        self.common_brands_models = common_brands_models
        self.feature_names = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        brand_columns = pd.DataFrame(
            {val: X_['Name'].apply(
                lambda x: self.is_a_common_brand(x, val)
                ) for val in self.common_brands},
            columns = self.common_brands
            )
        brand_model_columns = pd.DataFrame(
            {val: X_['Name'].apply(
                lambda x: self.is_a_common_brand_model(x, val)
                ) for val in self.common_brands_models},
            columns = self.common_brands_models
            )
        X_ = pd.concat([X_, brand_columns, brand_model_columns], axis=1)
        X_.drop(['Name'], axis=1, inplace=True)
        self.feature_names = X_.columns
        return X_
    
    def is_a_common_brand(self, x, common_brand):
        return (common_brand == x.split()[0])
    
    def is_a_common_brand_model(self, x, common_brand_model):
        return (common_brand_model == ' '.join(x.split()[:2]))
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names


class UnitsRemovingTransformer(TransformerMixin):
    '''
    Remove units from the columns values.
    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_names = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        for column in X_.columns:
            X_[column] = X_[column].apply(lambda x: float(str(x).split()[0]) 
                                          if str(x).split()[0] != 'null' 
                                          else None)
        X_.loc[X_['Mileage'] == 0, ['Mileage']] = None
        self.feature_names = X_.columns
        return X_
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names


class ReplaceZeros(TransformerMixin):
    '''
    Replace zeros with None in a Mileage column.
    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_names = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_.loc[X_['Mileage'] == 0, ['Mileage']] = None
        self.feature_names = X_.columns
        return X_
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names


class MappingTransformer(TransformerMixin):
    '''
    Map the values based on mapping dict.
    '''
    
    def __init__(self, mapping, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapping = mapping
        self.feature_names = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        for column in X_.columns:
            X_[column] = X_[column].apply(lambda x: self.mapping[str(x)] if x \
                                          else None)
        self.feature_names = X_.columns
        return X_
        
    def get_feature_names_out(self, input_features=None):
        return self.feature_names


class ReplaceOutliers(TransformerMixin):
    '''
    Replace outliers with the 95th percentile.
    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_names = None

    def fit(self, X, y=None):
        return self
    
    def replace_outlier(self, x, q25, q75, q80, q10, iqr):
        if x < q25 - 1.5*iqr:
            return q10
        if x > q75 + 1.5*iqr:
            return q80
        return x
        
    def transform(self, X, y=None):
        X_ = X.copy()
        for column in X_.columns:
            if (X_[column].dtype.kind in 'iufc') \
                and len(X_[column].unique()) > 2:
                q25 = np.percentile(X_[column], 25)
                q75 = np.percentile(X_[column], 75)
                q80 = np.percentile(X_[column], 80)
                q10 = np.percentile(X_[column], 10)
                iqr = q75 - q25
                X_[column] = X_[column].apply(
                    lambda x: self.replace_outlier(x, q25, q75, q80, q10, iqr))
        self.feature_names = X_.columns
        return X_
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names
