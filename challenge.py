from utils import Utilities as utils
import numpy as np
from tqdm import *


##
## @brief      Provides methods to interact with the Recommender Objects in order 
##             to retrieve results in the format required by the challenge
##
class RecommenderAdapter(object):

    ##
    ## @brief      Constructs the object.
    ##
    ## @param      self         The object
    ## @param      recommender  (Recommender) An instance of a fitted Recommender Object
    ## @param      dataset      (DatasetPrerocessing) An intance of DatasetPreprocessing Object
    ##
    def __init__(self, recommender, dataset):
        self.recommender = recommender
        self.dataset = dataset
        self.playlists = dataset.get_playlists()
        self.tracks = dataset.get_tracks()
        self.predictions = None

    ##
    ## @brief      Predicts Top N track recomendation for all the requested playlists
    ##
    ## @param      self          The object
    ## @param      to_predict    (Dataframe) A dataframe containing identifiers of the playlists
    ## @param      targets       (Dataframe) A dataframe containing identifiers of possible tracks
    ## @param      top_n         (Integer) The number of items to include in the top predictions
    ## @param      remove_known  (Boolean) Whether to remove known interactions
    ## @param      mask          (Boolean) Whether to remove predictions not in the set of target tracks
    ##
    ## @return     A list of the ids of the top N predicted tracks
    ##
    def predict(self, to_predict, targets, top_n=5, remove_known=True, mask=True):
        targets = targets['track_id'].unique()
        test = to_predict['playlist_id'].unique()
        predictions = []
        for i in tqdm(test):
            playlist_index = utils.get_target_index(i, self.playlists)
            current_ratings = self.recommender.urm[playlist_index, :]
            ratings = self.recommender.predict(current_ratings, remove_known)
            if mask:
                not_selected = np.where(~np.in1d(self.tracks, targets))[0]
                ratings[not_selected] = 0
            top_n_indexes = utils.get_n_best_indexes(ratings, top_n)
            top_n_predictions = self.tracks[top_n_indexes]
            predictions.append((i, top_n_predictions))
        self.predictions = predictions

    ##
    ## @brief      Computed the MAP5 score on the targets
    ##
    ## @param      self     The object
    ## @param      targets  (Dataframe) a dataframe containing future interactions for each playlist
    ##
    ## @return     the MAP5 score
    ##
    def evaluate(self, targets):
        labels = list(targets.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id'])))
        predictions = [p[1] for p in sorted(self.predictions, key=lambda x: x[0])]
        return utils.map5(predictions, labels)

    ##
    ## @brief      Computes predictions for a submission
    ##
    ## @param      self  The object
    ##
    ## @return     The predictions for the full dataset
    ##
    def predict_submission(self):
        test = self.dataset.dataset["sample_submission"]
        targets = self.dataset.dataset["target_tracks"]
        self.predict(test, targets)
        return self.predictions

