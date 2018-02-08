from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from scipy import sparse
import numpy as np


class Utilities(object):
    """This class offers generic utilities methods
    """

    @staticmethod
    def remove_duplicates(l):
        """Removes duplicates from a list

            Args:
                l: the list from which to remove the duplicates
            Returns:
                a list with no duplicates
        """
        return list(set(l))

    @staticmethod
    def remove_nones(l):
        """Removes Nones from a list

            Args:
                l: the list from which to remove the Nones
            Returns:
                a list with Nones removed
        """
        return [e for e in l if e is not None]

    @staticmethod
    def get_n_best_indexes(l, n):
        """Computes the indices for the n largest values in a list

            Args:
                l: input list or array
                n: number of indices to get
            Returns:
                an array that contains the n indices
        """
        return np.flip(np.argsort(l)[-n:], 0)

    @staticmethod
    def get_target_index(target_id, targets):
        """Get index of target given 

            Args:
                target_id: identifier of the target
                targets: vector of all the targets
            Returns:
                position of the target_id in the target vector
        """
        return np.where(target_id == targets)[0][0]

    @staticmethod
    def normalize_matrix(x, axis=0):
        """Normalizes a matrix

            Args:
                x: matrix to be normalized
            Returns:
                x: normalized matrix
        """
        x = normalize(x, norm='l2', axis=axis)
        return x

    @staticmethod
    def knn(mat, k):
        """Given a similarity matrix removes all but the k most similar elements from each row

            Args:
                mat: similarity matrix
                k: number of neighbours
            Returns:
                mat: similarity matrix with k most similar
        """
        mat = sparse.csr_matrix(mat)
        i = mat.data.argsort()[:-k]
        mat.data[i] = 0
        sparse.csr_matrix.eliminate_zeros(mat)
        return mat

    @staticmethod
    def map(recommended, relevant):
        """Compute map score of a single example

            Args:
                recommended: list containing the predictions
                relevant: list containing relevant items
            Returns:
                mat: map_score 
        """
        is_relevant = np.in1d(recommended, relevant, assume_unique=True)
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
        map_score = np.sum(p_at_k) / np.min([len(relevant), is_relevant.shape[0]])
        return map_score

    @staticmethod
    def map5(predictions, labels):
        """Compute average map score of a set of predictions

            Args:
                recommended: list of lists containing the predictions
                relevant: list of list containing relevant items
            Returns:
                mat: map_score 
        """
        map_tmp = 0
        for i in range(len(predictions)):
            map_tmp += Utilities.map(predictions[i], labels[i])
        return map_tmp / len(predictions)

    @staticmethod
    def tfidf_normalize(mat):
        """Applies tf-idf transformation to a matrix

            Args:
                mat: matrix
            Returns:
                transformed matrix
        """
        return TfidfTransformer().fit_transform(mat)

    @staticmethod
    def make_submission_csv(predictions, name):
        """Creates the submission file 

            Args:
                predictions: a vector of 5 identifiers of the predictions
                name: name of the output csv file
            Returns:
                None
        """
        csv_file = open(name + '.csv', 'w')
        csv_file.write('playlist_id,track_ids\n')
        for p in predictions:
            s = str(p[0]) + ',' + ' '.join([str(i) for i in p[1]]) + '\n'
            csv_file.write(s)
        csv_file.close()