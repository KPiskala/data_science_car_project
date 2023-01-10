# -*- coding: utf-8 -*-
"""
@author: Katarzyna Piskala
"""

import numpy as np
from sklearn.metrics import max_error, mean_absolute_percentage_error, \
    mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def get_scores(y_true, y_pred):
    '''
    Calculate errors and scores of the prediction using sklearn metrics.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Correct labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels.

    Returns
    -------
    scores : dict
        Prediction's errors and scores.

    '''
    scores = {
        'mean_abs_error': round(mean_absolute_error(y_true, y_pred), 2),
        'median_abs_error': round(median_absolute_error(y_true, y_pred), 2),
        'mean_abs_perc_error': 
            round(mean_absolute_percentage_error(y_true, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        'max_error': round(max_error(y_true, y_pred), 2),
        'r2': round(r2_score(y_true, y_pred), 2)
    }
    return scores


def display_scores(model, scores):
    '''
    Display model's scores.

    Parameters
    ----------
    model : str
        Model name.
    scores : dict
        Errors and scores of the model prediction.

    Returns
    -------
    None.

    '''
    print('Results for model', model)
    print('Mean absolute error:', scores['mean_abs_error'])
    print('Median absolute error:', scores['median_abs_error'])
    print('Mean absolute percentage error:', scores['mean_abs_perc_error'])
    print('Root mean squared error:', scores['rmse'])
    print('Max error:', scores['max_error'])
    print('R2 score:', scores['r2'])


def get_common_brands():
    '''
    Get common cars' brands chosen previously in data analysis.

    Returns
    -------
    common_brands : list
        most common cars' brands

    '''
    common_brands = [
        'Maruti',
        'Hyundai',
        'Honda',
        'Toyota',
        'Volkswagen',
        'Mercedes-Benz',
        'Ford',
        'Mahindra',
        'BMW',
        'Audi',
        'Tata',
        'Skoda',
        'Renault',
        'Chevrolet',
        'Nissan',
        'Land',
        'Jaguar'
    ]
    return common_brands


def get_common_models():
    '''
    Get common cars' models chosen previously in data analysis.

    Returns
    -------
    common_brands_models : list
        most common cars' models

    '''
    common_brands_models = [
         'Maruti Swift',
         'Honda City',
         'Hyundai i20',
         'Toyota Innova',
         'Hyundai Grand',
         'Hyundai Verna',
         'Hyundai i10',
         'Maruti Wagon',
         'Volkswagen Polo',
         'Maruti Alto',
         'Toyota Fortuner',
         'Mahindra XUV500',
         'Ford Figo',
         'Honda Amaze',
         'BMW 3',
         'Mercedes-Benz E-Class',
         'Volkswagen Vento',
         'Mercedes-Benz New',
         'Hyundai Creta',
         'Audi A4',
         'Maruti Ritz',
         'Renault Duster',
         'Toyota Corolla',
         'BMW 5',
         'Maruti Ciaz',
         'Hyundai Santro',
         'Mahindra Scorpio',
         'Hyundai EON',
         'Maruti Ertiga',
         'Maruti Baleno',
         'Honda Brio',
         'Maruti Celerio',
         'Honda Jazz',
         'Hyundai Xcent',
         'Land Rover',
         'Ford Ecosport',
         'Toyota Etios',
         'Skoda Rapid',
         'Ford EcoSport',
         'Skoda Superb',
         'Maruti Vitara',
         'Audi A6',
         'Chevrolet Beat',
         'Tata Indica',
         'Renault KWID',
         'Ford Endeavour',
         'Ford Fiesta',
         'Audi Q7',
         'Skoda Octavia',
         'Jaguar XF',
         'Maruti SX4',
         'BMW X1',
         'Nissan Sunny',
         'Audi Q3',
         'Honda CR-V',
         'Tata Nano',
         'Honda Civic',
         'Nissan Terrano',
         'Volkswagen Jetta',
         'Audi Q5',
         'Skoda Laura',
         'Nissan Micra',
         'Maruti Dzire',
         'Honda Accord',
         'Mercedes-Benz M-Class',
         'BMW X5',
         'Maruti Zen',
         'Hyundai Elantra',
         'Mahindra Xylo',
         'Tata Zest',
         'Mini Cooper',
         'Tata Indigo',
         'Mitsubishi Pajero',
         'Maruti Omni',
         'Mahindra Bolero',
         'Chevrolet Cruze',
         'Chevrolet Aveo',
         'Tata Manza',
         'Honda Mobilio',
         'Hyundai Accent',
         'Maruti Eeco',
         'Jeep Compass',
         'Mercedes-Benz GLA',
         'Volkswagen Ameo',
         'Ford Ikon',
         'Mahindra KUV',
         'Hyundai Santa',
         'Hyundai Elite',
         'Mahindra Ssangyong'
    ]
    return common_brands_models
