from sklearn.preprocessing import MultiLabelBinarizer
from utils import Utilities as utils
from sklearn.utils import shuffle
from scipy import sparse
import pandas as pd
import ast


class DatasetPreprocessing(object):
    """This class offers the methods for preprocessing the "Recommender System 2017 Challenge Polimi" dataset

    Attributes:
        dataset: a dictionary containing all the dataframes for all the csv files
    """

    def __init__(self, path_to_data="Datasets/"):

        self.dataset = {
            "tracks_final": pd.read_csv(path_to_data + 'tracks_final.csv', sep='\t'),
            "playlists_final": pd.read_csv(path_to_data + 'playlists_final.csv', sep='\t'),
            "target_playlists": pd.read_csv(path_to_data + 'target_playlists.csv', sep='\t'),
            "target_tracks": pd.read_csv(path_to_data + 'target_tracks.csv', sep='\t'),
            "train_final": pd.read_csv(path_to_data + 'train_final.csv', sep='\t'),
            "sample_submission": pd.read_csv(path_to_data + 'sample_submission.csv', sep=','),
        }

    def split_sets(self, test_size, items_threshold, n_to_remove):
        """Splits the available data into train and test set.

        Splits the dataframe of playlist interactions into three different dataframes.

            Args:
                test_size: The fraction of playlists to reserve for testing
                items_threshold: The minimum number of tracks for a playlist to be in the test set
                n_to_remove: The number of tracks to reserve for testing in each playlist of the test set
            Returns:
                train_data: A dataframe that contains the playlist/track interactions for the train set
                test_data: A dataframe that contains the playlist/track interactions for the test set minus
                the ones removed for testing
                test_data_target: A dataframe that contains the playlist/track interactions remove for testing
        """

        data = self.dataset['train_final']
        grouped = data.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))
        split_index = int(test_size * len(grouped))
        filtered_idx = grouped[[len(g) > items_threshold for g in grouped]].index
        test_idx = shuffle(filtered_idx)[:split_index]
        test_mask = data['playlist_id'].isin(test_idx)
        test_data = data[test_mask]
        train_data = data[~test_mask]
        test_data_target = test_data.groupby('playlist_id').apply(lambda x: x['track_id'].sample(n_to_remove))
        test_data_target = test_data_target.reset_index(level=0)
        test_data = test_data.drop(test_data_target.index)
        return train_data, test_data, test_data_target

    def get_playlists(self):
        """Get the list of the ids of all playlists in the dataset with no duplicates

            Returns:
                a list of integer ids in ascending order
        """
        return self.dataset['train_final'].sort_values(by='playlist_id', ascending=True)['playlist_id'].unique()

    def get_tracks(self):
        """Get the list of the ids of all tracks in the dataset with no duplicates

            Returns:
                a list of integer ids in ascending order
        """
        return self.dataset['tracks_final'].sort_values(by='track_id', ascending=True)['track_id'].unique()

    def get_target_tracks(self, test_target):
        return test_target['track_id'].unique()

    def get_list_of_playcount(self):
        """Get the list of the ids of all tags in the dataset with no duplicates

            Returns:
                a list of integer ids
        """
        playcount_for_item = self.dataset['tracks_final']['playcount']
        list_of_playcount = [p for p in playcount_for_item]
        return list_of_playcount

    def get_list_of_tags(self):
        """Get the list of the ids of all tags in the dataset with no duplicates

            Returns:
                a list of integer tags' ids
        """
        tags_per_item = self.dataset['tracks_final']['tags']
        list_of_tags = [t for sub in tags_per_item for t in ast.literal_eval(sub)]
        return utils.remove_duplicates(list_of_tags)

    def get_list_of_artists(self):
        """Get the list of the ids of all artists in the dataset with no duplicates

            Returns:
                a list of integer artists' ids
        """
        artist_per_item = self.dataset['tracks_final']['artist_id']
        list_of_artists = [a for a in artist_per_item]
        return utils.remove_duplicates(list_of_artists)

    def get_list_of_albums(self):
        """Get the list of the ids of all albums in the dataset with no duplicates

            Returns:
                a list of integer albums' ids
        """
        album_per_item = self.dataset['tracks_final']['album']
        list_of_albums = [
            ast.literal_eval(i)[0] if (len(ast.literal_eval(i)) > 0) and (ast.literal_eval(i)[0] is not None) else -1
            for i in album_per_item]
        return utils.remove_duplicates(list_of_albums)

    def build_urm(self, train_data=None, test_data=None):
        """Builds the user rating matrix (URM) i.e. the matrix containing the user rating for each item,
         if no test or train is provided uses the full dataset

            Args:
                train_data: (optional) The fraction of train playlists
                test_data: (optional) The fraction of test playlists
            Returns:
                urm: a csr scipy sparse matrix containing the URM
        """

        if test_data is not None or train_data is not None:
            data = train_data.append(test_data)
            train = data.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))
        else:
            train = self.dataset['train_final'].groupby('playlist_id', as_index=True)
            train = train.apply(lambda x: list(x['track_id']))

        # one hot encoded representation of the interactions
        urm = MultiLabelBinarizer(classes=self.get_tracks(), sparse_output=True).fit_transform(train)

        return urm.tocsr()

    def build_icm(self, tfidf=False, normalized=True):
        """Builds the item content matrix (ICM) i.e. the matrix of the items' features

            Args:
                tfidf: (optional) if True applies TF-IDF transformation to the output matrix, default False
                normalized: (optional) if True normalized the matrix, default True
            Returns:
                icm: a csc scipy sparse matrix containing the ICM
        """

        features = self.dataset['tracks_final'].sort_values(by='track_id', ascending=True)

        # list of tags for each item
        tags_per_item = [ast.literal_eval(i) for i in list(features['tags'])]
        # list of albums for each item
        album_per_item = [
            ast.literal_eval(i)
            if (len(ast.literal_eval(i)) > 0) and (ast.literal_eval(i)[0] is not None)
            else []
            for i in list(features['album'])
        ]
        # list of artists for each item
        artist_per_item = [[i] for i in list(features['artist_id'])]

        # list of all tags in the dataset
        l_tags = self.get_list_of_tags()
        # list of all tags in the dataset
        l_albums = self.get_list_of_albums()
        # list of all tags in the dataset
        l_artists = self.get_list_of_artists()

        # one hot encoded representation of tags for each item
        enc_tags = MultiLabelBinarizer(classes=l_tags, sparse_output=True).fit_transform(tags_per_item)
        # one hot encoded representation of albums for each item
        enc_albums = MultiLabelBinarizer(classes=l_albums, sparse_output=True).fit_transform(album_per_item)
        # one hot encoded representation of artists for each item
        enc_artists = MultiLabelBinarizer(classes=l_artists, sparse_output=True).fit_transform(artist_per_item)

        icm = sparse.hstack([enc_artists, enc_albums, enc_tags])

        if tfidf:
            icm = utils.tfidf_normalize(icm)

        if normalized:
            icm = utils.normalize_matrix(icm.tocsr(), axis=0)

        return icm