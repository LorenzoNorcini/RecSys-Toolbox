from utils import Utilities as utils
from scipy.sparse.linalg import svds
from abc import ABC, abstractmethod
from scipy.sparse import linalg
from scipy import sparse
import numpy as np
from tqdm import *

##
## @brief      Class for recommender.
##
class Recommender(ABC):

    ##
    ## @brief      Constructs the object.
    ##
    ## @param      self                The object
    ## @param      user_rating_matrix  (numpy array) The user rating matrix
    ##
    @abstractmethod
    def __init__(self, user_rating_matrix):
        self.urm = user_rating_matrix

    ##
    ## @brief      Abstract method providing an interface for the computation of the similarity matrix
    ##
    ## @param      self    The object
    ## @param      kwargs  Eventual optional arguments
    ##
    ## @return     The similarity matrix.
    ##
    @abstractmethod
    def _compute_similarity_matrix(self, **kwargs):
        pass

    ##
    ## @brief      Abstract method providing the interface for fitting the model
    ##
    ## @param      self    The object
    ## @param      kwargs  Eventual optional arguments
    ##
    ## @return     None
    ##
    @abstractmethod
    def fit(self, **kwargs):
        pass

    ##
    ## @brief      Abstract method providing the interface for the prediction
    ##
    ## @param      self    The object
    ## @param      kwargs  Eventual optional arguments
    ##
    ## @return     The preficted ratings
    ##
    @abstractmethod
    def predict(self, target):
        pass

##
## @brief      Class for basic content based filtering.
##
class BasicContentBasedFiltering(Recommender):

    ##
    ## @brief      Constructs the object.
    ##
    ## @param      self                The object
    ## @param      user_rating_matrix  (numpy array) The user rating matrix
    ##
    def __init__(self, user_rating_matrix):
        super().__init__(user_rating_matrix)
        self.sm = None

    ##
    ## @brief      Fits the model computing the similarity between items according to their features
    ##
    ## @param      self                  The object
    ## @param      item_content_matrix   The item content matrix
    ## @param      k_nearest_neighbours  The k nearest neighbours
    ##
    ## @return     None
    ##
    def fit(self, item_content_matrix, k_nearest_neighbours):
        self.sm = self._compute_similarity_matrix(item_content_matrix, k_nearest_neighbours)

    ##
    ## @brief      Calculates the similarity matrix.
    ##
    ## @param      self  The object
    ## @param      icm   The icm
    ## @param      knn   The knn
    ##
    ## @return     The similarity matrix.
    ##
    def _compute_similarity_matrix(self, icm, knn):
        s_tmp = []
        n_items = icm.shape[0]
        m = icm.tocsr()
        m_t = m.T.tocsr()
        for i in tqdm(range(n_items)):
            mat = m[i, :].dot(m_t)
            s_tmp.append(utils.knn(mat, knn))
        s = sparse.vstack(s_tmp, format='csr')
        s.setdiag(0)
        return s

    ##
    ## @brief      Predicts the rating for the given target
    ##
    ## @param      self          The object
    ## @param      target        The user vector of interactions
    ## @param      remove_known  Whether to remove known interactions
    ##
    ## @return     The predicted ratings
    ##
    def predict(self, target, remove_known=True):
        ratings = (target * self.sm).toarray().flatten()
        if remove_known:
            known_items = np.nonzero(target)[1]
            ratings[known_items] = 0
        return ratings

##
## @brief      Class for basic collaborative filtering.
##
class BasicCollaborativeFiltering(Recommender):

    ##
    ## @brief      Constructs the object.
    ##
    ## @param      self                The object
    ## @param      user_rating_matrix  (numpy array) The user rating matrix
    ##
    def __init__(self, user_rating_matrix):
        super().__init__(user_rating_matrix)
        self.sm = None

    ##
    ## @brief      Fits the model computing the similarity between items according to their interactions
    ##
    ## @param      self                  The object
    ## @param      k_nearest_neighbours  The number of nearest neighbours
    ##
    ## @return     None
    ##
    def fit(self, k_nearest_neighbours):
        self.sm = self._compute_similarity_matrix(k_nearest_neighbours)

    ##
    ## @brief      Calculates the similarity matrix.
    ##
    ## @param      self  The object
    ## @param      knn   The number of nearest neighbours
    ##
    ## @return     The similarity matrix.
    ##
    def _compute_similarity_matrix(self, knn):
        ucm = self.urm.T
        s_tmp = []
        n_items = ucm.shape[0]
        m = ucm.tocsr()
        m_t = m.T.tocsr()
        for i in tqdm(range(n_items)):
            mat = m[i, :].dot(m_t)
            s_tmp.append(utils.knn(mat, knn))
        s = sparse.vstack(s_tmp, format='csr')
        s.setdiag(0)
        return s

    ##
    ## @brief      Predicts the rating for the given target
    ##
    ## @param      self          The object
    ## @param      target        (numpy array) The user vector of interactions
    ## @param      remove_known  (Boolean) Whether to remove known interactions
    ##
    ## @return     (numpy array) The predicted ratings
    ##
    def predict(self, target, remove_known=True):
        ratings = (target * self.sm).toarray().flatten()
        if remove_known:
            known_items = np.nonzero(target)[1]
            ratings[known_items] = 0
        return ratings

##
## @brief      Class for svd matrix factorization.
##
class SVDMatrixFactorization(Recommender):

    ##
    ## @brief      Constructs the object.
    ##
    ## @param      self                The object
    ## @param      user_rating_matrix  (numpy array) The user rating matrix
    ## @param      n_factors           (Integer) The number of factor to use in the decomposition
    ##
    def __init__(self, user_rating_matrix, n_factors):
        super().__init__(user_rating_matrix)
        self.n_factors = n_factors
        self.sm = None

    ##
    ## @brief      Computes the SVD decomposition of the user rating matrix
    ##
    ## @param      self  The object
    ##
    ## @return     (numpy array) The item factors of the decomposed matrix
    ##
    def SVD(self):
        _, _, v_t = svds(urm.tocsc(), self.n_factors, return_singular_vectors='vh')
        return v_t

    ##
    ## @brief      Fits the model computing the similarity between items computed with the item latent factors of the SVD factorization
    ##
    ## @param      self                    The object
    ## @param      k_nearest_neighbours    (Integer) The number of nearest neighbours
    ## @param      precomputed_similarity  (numpy array) The precomputed similarity matrix computed by the dot product of the item factors
    ##
    ## @return     None
    ##
    def fit(self, k_nearest_neighbours, precomputed_similarity=None):
        if precomputed_similarity is None:
            self.sm = self._compute_similarity_matrix(k_nearest_neighbours, n_factors, lam, n_iterations)
        else:
            self.sm = precomputed_similarity

    ##
    ## @brief      Calculates the similarity matrix.
    ##
    ## @param      self          The object
    ## @param      knn           (Integer) The number of nearest neighbours
    ## @param      n_factors     (Integer) The number of factor to use in the decomposition
    ##
    ## @return     (numpy array) The similarity matrix.
    ##
    def _compute_similarity_matrix(self, knn, n_factors):
        s_tmp = []
        item_factors = self.ALS(self.urm, n_factors, lam, n_iterations)
        n_items = item_factors.shape[0]
        item_factors = sparse.csr_matrix(item_factors)
        item_factors_T = item_factors.T
        for i in tqdm(range(n_items)):
            mat = item_factors[i, :].dot(item_factors_T)
            s_tmp.append(utils.knn(mat, knn))
        s = sparse.vstack(s_tmp, format='csr')
        s.setdiag(0)
        return s

    ##
    ## @brief      Predicts the rating for the given target
    ##
    ## @param      self          The object
    ## @param      target        (numpy array) The user vector of interactions
    ## @param      remove_known  (Boolean) Whether to remove known interactions
    ##
    ## @return     (numpy array) The predicted ratings
    ##
    def predict(self, target, remove_known=True):
        ratings = (target * self.sm).toarray().flatten()
        if remove_known:
            known_items = np.nonzero(target)[1]
            ratings[known_items] = 0
        return ratings


##
## @brief      Class for als matrix factorization.
##
class ALSMatrixFactorization(Recommender):

    ##
    ## @brief      Constructs the object.
    ##
    ## @param      self                The object
    ## @param      user_rating_matrix  (numpy array) The user rating matrix
    ## @param      n_factors           (Integer) The number of factor to use in the decomposition
    ## @param      regularization      (Float) The regularization factor for the 
    ## @param      n_iterations        (Integer) The number of iterations of the ALS algorithm
    ##
    def __init__(self, user_rating_matrix, n_factors, regularization, n_iterations):
        super().__init__(user_rating_matrix)
        self.n_factors = n_factors
        self.lam = regularization
        self.n_iterations = n_iterations
        self.sm = None

    ##
    ## @brief      Computes the ALS decomposition of the user rating matrix
    ##
    ## @param      self  The object
    ##
    ## @return     (numpy array) The item factors of the decomposed matrix
    ##
    def ALS(self):
        m, n = self.urm.shape
        Y = np.mat(np.random.rand(self.n_factors, n))
        for i in range(self.n_iterations):
            X = np.mat(linalg.spsolve((Y * Y.T) + self.lam * sparse.eye(self.n_factors), (Y * self.urm.T)).T)
            Y = np.mat(linalg.spsolve((X.T * X) + self.lam * sparse.eye(self.n_factors), (X.T * self.urm)))
        return np.array(X), np.array(Y.T)

    ##
    ## @brief      Fits the model computing the similarity between items computed with the item latent factors of the ALS factorization
    ##
    ## @param      self                    The object
    ## @param      k_nearest_neighbours    (Integer) The number of nearest neighbours
    ## @param      precomputed_similarity  (numpy array) The precomputed similarity matrix computed by the dot product of the item factors
    ##
    ## @return     None
    ##
    def fit(self, k_nearest_neighbours, precomputed_similarity=None):
        if precomputed_similarity is None:
            self.sm = self._compute_similarity_matrix(k_nearest_neighbours)
        else:
            self.sm = precomputed_similarity

    ##
    ## @brief      Calculates the similarity matrix.
    ##
    ## @param      self  The object
    ## @param      knn   The knn
    ##
    ## @return     The similarity matrix.
    ##
    def _compute_similarity_matrix(self, knn):
        s_tmp = []
        user_factors, item_factors = self.ALS()
        n_items = item_factors.shape[0]
        item_factors = sparse.csr_matrix(item_factors)
        item_factors_T = item_factors.T
        for i in tqdm(range(n_items)):
            mat = item_factors[i, :].dot(item_factors_T)
            s_tmp.append(utils.knn(mat, knn))
        s = sparse.vstack(s_tmp, format='csr')
        s.setdiag(0)
        return s

    ##
    ## @brief      Predicts the rating for the given target
    ##
    ## @param      self          The object
    ## @param      target        (numpy array) The user vector of interactions
    ## @param      remove_known  (Boolean) Whether to remove known interactions
    ##
    ## @return     (numpy array) The predicted ratings
    ##
    def predict(self, target, remove_known=True):
        ratings = (target * self.sm).toarray().flatten()
        if remove_known:
            known_items = np.nonzero(target)[1]
            ratings[known_items] = 0
        return ratings

##
## @brief      Class for slimbpr.
##
class SLIMBPR(Recommender):

    ##
    ## @brief      Constructs the object.
    ##
    ## @param      self                The object
    ## @param      user_rating_matrix  (numpy array) The user rating matrix
    ## @param      learning_rate       (Float) The learning rate
    ## @param      epochs              (Integer) The number of epochs of training
    ## @param      pir                 (Float) The positive item regularization
    ## @param      nir                 (Float) The negative item regularization
    ##
    def __init__(self, user_rating_matrix, learning_rate, epochs, pir, nir):
        super().__init__(user_rating_matrix)
        self.epochs = epochs
        self.n_users = self.urm.shape[0]
        self.n_items = self.urm.shape[1]
        self.learning_rate = learning_rate
        self.positive_item_regularization = pir
        self.negative_item_regularization = nir
        self.sm = np.zeros((self.n_items, self.n_items))

    ##
    ## @brief      Samples a random triplet from the user rating matrix
    ##
    ## @param      self  The object
    ##
    ## @return     The index of the sampled user, the index of a positive interaction, the index of a negative interaction
    ##
    def sample(self):
        user_index = np.random.choice(self.n_users)
        interactions = self.urm[user_index].indices
        interaction_index = np.random.choice(interactions)
        selected = False
        while not selected:
            negative_interaction_index = np.random.randint(0, self.n_items)
            if negative_interaction_index not in interactions: selected = True
        return user_index, interaction_index, negative_interaction_index

    ##
    ## @brief      updates the similarity matrix once for each positive interaction
    ##
    ## @param      self  The object
    ##
    ## @return     None
    ##
    def iteration(self):
        num_positive_iteractions = int(self.urm.nnz)
        for _ in tqdm(range(num_positive_iteractions)):
            user_index, positive_item_id, negative_item_id = self.sample()
            user_interactions = self.urm[user_index, :].indices
            x_i = self.sm[positive_item_id, user_interactions].sum()
            x_j = self.sm[negative_item_id, user_interactions].sum()
            z = 1. / (1. + np.exp(x_i - x_j))
            for v in user_interactions:
                d = z - self.positive_item_regularization * x_i
                self.sm[positive_item_id, v] += self.learning_rate * d
                d = z - self.negative_item_regularization * x_j
                self.sm[negative_item_id, v] -= self.learning_rate * d
                self.sm[positive_item_id, positive_item_id] = 0
                self.sm[negative_item_id, negative_item_id] = 0

    ##
    ## @brief      Fits the model computing the similarity between items that maximises
    ##
    ## @param      self                  The object
    ## @param      k_nearest_neighbours  (Integer) The number of nearest neighbours
    ##
    ## @return     None
    ##
    def fit(self, k_nearest_neighbours):
        self._compute_similarity_matrix(k_nearest_neighbours)

    ##
    ## @brief      Calculates the similarity matrix.
    ##
    ## @param      self  The object
    ## @param      knn   The knn
    ##
    ## @return     None
    ##
    def _compute_similarity_matrix(self, knn):
        for e in range(self.epochs):
            self.iteration()
        s_tmp = []
        for i in tqdm(range(self.n_items)):
            mat = self.sm[i, :]
            s_tmp.append(utils.knn(mat, knn))
        s = sparse.vstack(s_tmp, format='csr')
        s.setdiag(0)
        self.sm = s

    ##
    ## @brief      Predicts the rating for the given target
    ##
    ## @param      self          The object
    ## @param      target        (numpy array) The user vector of interactions
    ## @param      remove_known  (Boolean) Whether to remove known interactions
    ##
    ## @return     (numpy array) The predicted ratings
    ##
    def predict(self, target, remove_known=True):
        ratings = (target * self.sm).toarray().flatten()
        if remove_known:
            known_items = np.nonzero(target)[1]
            ratings[known_items] = 0
        return ratings

##
## @brief      Class for cbfcb hybrid.
##
class CBF_CB_Hybrid(Recommender):

    ##
    ## @brief      Constructs the object.
    ##
    ## @param      self                The object
    ## @param      user_rating_matrix  (numpy array) The user rating matrix
    ## @param      cbf_weight          (Float) The cbf weight
    ## @param      cf_weight           (Float) The cf weight
    ##
    def __init__(self, user_rating_matrix, cbf_weight, cf_weight):
        super().__init__(user_rating_matrix)
        self.cbf_weight = cbf_weight
        self.cf_weight = cf_weight
        self.sm = None

    ##
    ## @brief      Fits the model computing the similarity between items according the  weighted average of 
    ##             the similarities computed according to the interactions and the similarities computed according to their features 
    ##             
    ## @param      self                  The object
    ## @param      k_nearest_neighbours  The number of nearest neighbours
    ##
    ## @return     None
    ##
    def fit(self, item_content_matrix, k_nearest_neighbours):
        self.sm = self._compute_similarity_matrix(item_content_matrix, k_nearest_neighbours)

    ##
    ## @brief      Calculates the similarity matrix.
    ##
    ## @param      self  The object
    ## @param      icm   The icm
    ## @param      knn   The knn
    ##
    ## @return     The similarity matrix.
    ##
    def _compute_similarity_matrix(self, icm, knn):
        s_tmp = []
        ucm = self.urm.T
        n_items = icm.shape[0]
        m1 = icm.tocsr()
        m1_t = m1.T.tocsr()
        m2 = ucm.tocsr()
        m2_t = m2.T.tocsr()
        for i in tqdm(range(n_items)):
            cfb = m1[i, :].dot(m1_t)
            cf = m2[i, :].dot(m2_t)
            mat = self.cbf_weight*cfb + self.cf_weight*cf
            s_tmp.append(utils.knn(mat, knn))
        s = sparse.vstack(s_tmp, format='csr')
        s.setdiag(0)
        return s

    ##
    ## @brief      Predicts the rating for the given target
    ##
    ## @param      self          The object
    ## @param      target        (numpy array) The user vector of interactions
    ## @param      remove_known  (Boolean) Whether to remove known interactions
    ##
    ## @return     (numpy array) The predicted ratings
    ##
    def predict(self, target, remove_known=True):
        ratings = (target * self.sm).toarray().flatten()
        if remove_known:
            known_items = np.nonzero(target)[1]
            ratings[known_items] = 0
        return ratings

##
## @brief      Class for CBF CF SLIM BPR hybrid.
##
class CBF_CF_SLIMBPR_Hybrid(Recommender):

    ##
    ## @brief      Constructs the object.
    ##
    ## @param      self                The object
    ## @param      user_rating_matrix  (numpy array) The user rating matrix
    ## @param      slim_lr             (Float) Slim learning rate
    ## @param      slim_epochs         (Integer) Number of slim epochs
    ## @param      slim_pir            (Float) Slim positive item regularization
    ## @param      slim_nir            (Float) Slim negative item regularization
    ## @param      slim_knn            (Integer) Number of nearest neighbours of the slim similarity
    ## @param      cbf_weight          (Float) The cbf weight
    ## @param      cf_weight           (Float) The cf weight
    ## @param      slim_weight         (Float) The slim weight
    ##
    def __init__(self, user_rating_matrix, slim_lr, slim_epochs, slim_pir, slim_nir,
                 slim_knn, cbf_weight, cf_weight, slim_weight):
        super().__init__(user_rating_matrix)
        self.cbf_weight = cbf_weight
        self.cf_weight = cf_weight
        self.slim_weight = slim_weight
        self.slim_knn = slim_knn
        self.slim_bpr = SLIMBPR(self.urm, slim_lr, slim_epochs, slim_pir, slim_nir)
        self.sm = None

    ##
    ## @brief      Fits the model computing the similarity between items according the weighted average of 
    ##             the similarities computed according to the interactions and the similarities computed 
    ##             according to their features also fits the slim bpr model
    ##
    ## @param      self                  The object
    ## @param      item_content_matrix   (numpy array) The item content matrix
    ## @param      k_nearest_neighbours  (Integer) The number of nearest neighbours
    ##
    ## @return     { description_of_the_return_value }
    ##
    def fit(self, item_content_matrix, k_nearest_neighbours):
        self.sm = self._compute_similarity_matrix(item_content_matrix, k_nearest_neighbours)
        self.slim_bpr.fit(self.slim_knn)

    ##
    ## @brief      Calculates the similarity matrix.
    ##
    ## @param      self  The object
    ## @param      icm   The icm
    ## @param      knn   The knn
    ## @param      als   The als
    ## @param      svd   The svd
    ##
    ## @return     The similarity matrix.
    ##
    def _compute_similarity_matrix(self, icm, knn):
        s_tmp = []
        ucm = self.urm.T
        n_items = icm.shape[0]
        m1 = icm.tocsr()
        m1_t = m1.T.tocsr()
        m2 = ucm.tocsr()
        m2_t = m2.T.tocsr()
        for i in tqdm(range(n_items)):
            cfb_i = m1[i, :].dot(m1_t)
            cf_i = m2[i, :].dot(m2_t)
            mat = self.cbf_weight*cfb_i + self.cf_weight*cf_i
            s_tmp.append(utils.knn(mat, knn))
        s = sparse.vstack(s_tmp, format='csr')
        s.setdiag(0)
        return s

    ##
    ## @brief      Predicts the rating for the given target combining the slim ratings and the similairty hybrid ratings
    ##
    ## @param      self          The object
    ## @param      target        (numpy array) The user vector of interactions
    ## @param      remove_known  (Boolean) Whether to remove known interactions
    ##
    ## @return     (numpy array) The predicted ratings
    ##
    def predict(self, target, remove_known=True):
        slim_ratings = self.slim_bpr.predict(target, False)
        hybrid_ratings = (target * self.sm).toarray().flatten()
        ratings = self.slim_weight * slim_ratings + (1. - self.slim_weight) * hybrid_ratings
        if remove_known:
            known_items = np.nonzero(target)[1]
            ratings[known_items] = 0
        return ratings


##
## @brief      Class for full hybrid.
##
class MixedHybrid(Recommender):

    ##
    ## @brief      Constructs the object.
    ##
    ## @param      self                The object
    ## @param      user_rating_matrix  (numpy array) The user rating matrix
    ## @param      slim_lr             (Float) Slim learning rate
    ## @param      slim_epochs         (Integer) Number of slim epochs
    ## @param      slim_pir            (Float) Slim positive item regularization
    ## @param      slim_nir            (Float) Slim negative item regularization
    ## @param      slim_knn            (Integer) Number of nearest neighbours of the slim similarity
    ## @param      cbf_weight          (Float) The cbf weight
    ## @param      cf_weight           (Float) The cf weight
    ## @param      als_weight          (Float) The als weight
    ## @param      svd_weight          (Float) The svd weight
    ## @param      slim_weight         (Float) The slim weight
    ##
    def __init__(self, user_rating_matrix, slim_lr, slim_epochs, slim_pir, slim_nir,
                 slim_knn, cbf_weight, cf_weight, als_weight, svd_weight, slim_weight):
        super().__init__(user_rating_matrix)
        self.cbf_weight = cbf_weight
        self.cf_weight = cf_weight
        self.als_weight = als_weight
        self.svd_weight = svd_weight
        self.slim_weight = slim_weight
        self.slim_knn = slim_knn
        self.slim_bpr = SLIMBPR(self.urm, slim_lr, slim_epochs, slim_pir, slim_nir)
        self.sm = None

    ##
    ## @brief      Fits the model computing the similarity between items according the weighted average of 
    ##             the similarities computed according to the interactions, the similarities computed according to their features 
    ##             and the similarities computed with factorization, also fits the slim bpr model
    ##
    ## @param      self                  The object
    ## @param      item_content_matrix   (numpy array) The item content matrix
    ## @param      k_nearest_neighbours  (Integer) The number of nearest neighbours
    ## @param      computed_als          (numpy array) The precomputed als similarity matrix
    ## @param      computed_svd          (numpy array) The precomputed svd similarity matrix
    ##
    ## @return     { description_of_the_return_value }
    ##
    def fit(self, item_content_matrix, k_nearest_neighbours, computed_als, computed_svd):
        self.sm = self._compute_similarity_matrix(item_content_matrix, k_nearest_neighbours, computed_als, computed_svd)
        self.slim_bpr.fit(self.slim_knn)

    ##
    ## @brief      Calculates the similarity matrix.
    ##
    ## @param      self  The object
    ## @param      icm   The icm
    ## @param      knn   The knn
    ## @param      als   The als
    ## @param      svd   The svd
    ##
    ## @return     The similarity matrix.
    ##
    def _compute_similarity_matrix(self, icm, knn, als, svd):
        s_tmp = []
        ucm = self.urm.T
        n_items = icm.shape[0]
        m1 = icm.tocsr()
        m1_t = m1.T.tocsr()
        m2 = ucm.tocsr()
        m2_t = m2.T.tocsr()
        for i in tqdm(range(n_items)):
            cfb_i = m1[i, :].dot(m1_t)
            cf_i = m2[i, :].dot(m2_t)
            als_i = als[i, :]
            svd_i = svd[i, :]
            mat = self.cbf_weight*cfb_i + self.cf_weight*cf_i + self.als_weight*als_i + self.svd_weight*svd_i
            s_tmp.append(utils.knn(mat, knn))
        s = sparse.vstack(s_tmp, format='csr')
        s.setdiag(0)
        return s

    ##
    ## @brief      Predicts the rating for the given target combining the slim ratings and the similairty hybrid ratings
    ##
    ## @param      self          The object
    ## @param      target        (numpy array) The user vector of interactions
    ## @param      remove_known  (Boolean) Whether to remove known interactions
    ##
    ## @return     (numpy array) The predicted ratings
    ##
    def predict(self, target, remove_known=True):
        slim_ratings = self.slim_bpr.predict(target, False)
        hybrid_ratings = (target * self.sm).toarray().flatten()
        ratings = self.slim_weight * slim_ratings + (1. - self.slim_weight) * hybrid_ratings
        if remove_known:
            known_items = np.nonzero(target)[1]
            ratings[known_items] = 0
        return ratings