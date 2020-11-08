import logging
import os
from typing import Dict
from pathlib import Path

from src.model.abstract_model import AbstractModel


logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, 
                 models: Dict[str, AbstractModel], 
                 model_path: os.PathLike = Path('models/')):
        """ Modeltrainer trains/evaluates all models in the models dict.

         Parameters
         ----------
         models: Dict[str, AbstractModel]
             Dictionary containing a model name, and a model instance
         model_path: os.PathLike
             Path to store serialized models.
         """
        self.models = models
        self.model_path = model_path

    def train(self, X, y, fit_kwargs={}, save_kwargs={}):
        for m_name, m in self.models.items():
            logger.info(f'Start training {m_name}')
            m.fit(X, y, **fit_kwargs.get(m_name, {}))
            model_path = os.path.join(self.model_path, m_name)
            m.save(filepath=model_path, **save_kwargs.get(m_name, {}))
            logger.info(f'Done training {m_name} and saved to file.')
    
    def generate_quantile_predictions(self, X, predict_kwargs={}):
        preds = {}
        for m_name, m in self.models.items():
            logger.info(f'Start predicting using model: {m_name}')
            m.load(filepath=os.path.join(self.model_path, m_name))
            pkwargs = predict_kwargs.get(m_name, {})
            preds[m_name] = m.predict_quantiles(X, **pkwargs)
            logger.info(f'Done predicting.')

        return preds
    
    def generate_posterior_samples(self, X, sample_kwargs={}):
        preds = {}
        for m_name, m in self.models.items():
            logger.info(f'Start predicting using model: {m_name}')
            m.load(filepath=os.path.join(self.model_path, m_name))
            skwargs = sample_kwargs.get(m_name, {})
            preds[m_name] = m.sample_posterior_predictive(X, **skwargs)
            logger.info(f'Done predicting.')

        return preds
