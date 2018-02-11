from preprocessing import DatasetPreprocessing
from recommender import *
from challenge import RecommenderAdapter


def test_content_based_filtering():

    knn = 300

    bcbf = BasicContentBasedFiltering(user_rating_matrix=urm)

    print("Building Content Based Similarity Matrix")
    bcbf.fit(item_content_matrix=icm, k_nearest_neighbours=knn)

    adapter = RecommenderAdapter(recommender=bcbf, dataset=dataset)

    adapter.predict(to_predict=test_data, targets=test_data_target, top_n=5, remove_known=True, mask=True)

    return adapter.evaluate(targets=test_data_target)


def test_collaborative_filtering():

    knn = 300

    bcf = BasicCollaborativeFiltering(user_rating_matrix=urm)

    print("Building Collaborative Filtering Similarity Matrix")
    bcf.fit(k_nearest_neighbours=knn)

    adapter = RecommenderAdapter(recommender=bcf, dataset=dataset)

    adapter.predict(to_predict=test_data, targets=test_data_target, top_n=5, remove_known=True, mask=True)

    return adapter.evaluate(targets=test_data_target)


def test_ALS_matrix_factorization():

    n_factors = 1
    reg = .01
    n_iter = 1
    knn = 300

    als_mf = ALSMatrixFactorization(user_rating_matrix=urm, n_factors=n_factors, regularization=reg, n_iterations=n_iter)

    print("Building ALS Similarity Similarity Matrix")
    als_mf.fit(k_nearest_neighbours=knn)

    adapter = RecommenderAdapter(recommender=als_mf, dataset=dataset)

    adapter.predict(to_predict=test_data, targets=test_data_target, top_n=5, remove_known=True, mask=True)

    return adapter.evaluate(targets=test_data_target)


def test_SLIM_BPR():

    lr = 0.01
    e = 1
    pir = 1.
    nir = 1.
    knn = 300

    slim_bpr = SLIMBPR(user_rating_matrix=urm, learning_rate=lr, epochs=e, pir=pir, nir=nir)

    print("Building SLIM Similarity Matrix")
    slim_bpr.fit(k_nearest_neighbours=knn)

    adapter = RecommenderAdapter(recommender=slim_bpr, dataset=dataset)

    adapter.predict(to_predict=test_data, targets=test_data_target, top_n=5, remove_known=True, mask=True)

    return adapter.evaluate(targets=test_data_target)


def test_Hybrid_CBF_CF():

    knn = 300

    bcbf = CBF_CB_Hybrid(user_rating_matrix=urm, cbf_weight=0.8, cf_weight=0.2)

    print("Building Hybrid CBF CF Similarity Matrix")
    bcbf.fit(item_content_matrix=icm, k_nearest_neighbours=knn)

    adapter = RecommenderAdapter(recommender=bcbf, dataset=dataset)

    adapter.predict(to_predict=test_data, targets=test_data_target, top_n=5, remove_known=True, mask=True)

    return adapter.evaluate(targets=test_data_target)

def test_CBF_CF_SLIMBPR_Hybrid():

    lr = 0.01
    e = 1
    pir = 1.
    nir = 1.
    knn_slim = 600
    al = 0.8
    be = 0.2
    ga = 0.8
    knn = 300

    bcbf = CBF_CF_SLIMBPR_Hybrid(user_rating_matrix=urm, slim_lr=lr, slim_epochs=e, slim_pir=pir, slim_nir=nir, slim_knn=knn_slim,
                      cbf_weight=al, cf_weight=be, slim_weight=ga)

    print("Building CBF CF Hybrid Similarity Matrix")
    bcbf.fit(item_content_matrix=icm, k_nearest_neighbours=knn)

    adapter = RecommenderAdapter(recommender=bcbf, dataset=dataset)

    adapter.predict(to_predict=test_data, targets=test_data_target, top_n=5, remove_known=True, mask=True)

    return adapter.evaluate(targets=test_data_target)

def test_Mixed_Hybrid():

    lr = 0.01
    e = 1
    pir = 1.
    nir = 1.
    knn_slim = 600
    al = 0.8
    be = 0.2
    ga = 0.8
    de = 0.7
    te = 0.07
    knn = 300

    print("Loading Precomputed Factorization Matrices")
    svd = sparse.load_npz('SVD_5000_knn_600.npz')
    als = sparse.load_npz('ALS_5000_knn_1000.npz')

    mh = MixedHybrid(user_rating_matrix=urm, slim_lr=lr, slim_epochs=e, slim_pir=pir, slim_nir=nir, slim_knn=knn_slim,
                      cbf_weight=al, cf_weight=be, als_weight=te, svd_weight=de, slim_weight=ga)

    print("Building Full Hybrid Similarity Matrix")
    mh.fit(item_content_matrix=icm, k_nearest_neighbours=knn, computed_als=als, computed_svd=svd)

    adapter = RecommenderAdapter(recommender=mh, dataset=dataset)

    submission_predictions = adapter.predict_submission()

    utils.make_submission_csv(submission_predictions, "submission")


if __name__ == "__main__":

    CBF_CF = []
    CBF_CF_SLIM = []
    CBF = []
    CF = []
    SLIM = []

    for i in range(10):
        dataset = DatasetPreprocessing()

        test_size = .2
        item_th = 10
        n_to_remove = 5

        print("Splitting Dataset")
        splits = dataset.split_sets(test_size=test_size, items_threshold=item_th, n_to_remove=n_to_remove)
        train_data, test_data, test_data_target = splits

        print("Building User Rating Matrix")
        urm = dataset.build_urm(train_data=train_data, test_data=test_data)
        print("Building Item Content Matrix")
        icm = dataset.build_icm(tfidf=True, normalized=True)
        urm = utils.tfidf_normalize(urm.T).T.tocsr()

        # Hybrid CBF CF #

        s = test_Hybrid_CBF_CF()
        print("Hybrid CBF CF MAP5 Score: ", s)

        CBF_CF.append(s)

        # CBF CF Slim BPR #

        s = test_CBF_CF_SLIMBPR_Hybrid()
        print("Hybrid CBF CF SLIM MAP5 Score: ", s)

        CBF_CF_SLIM.append(s)

        # Mixed Hybrid #

        # test_Mixed_Hybrid()
        
        # Content Based Filtering #

        s = test_content_based_filtering()

        print("CBF MAP5 Score: ", s)

        CBF.append(s)

        # Collaborative Filtering #

        s = test_collaborative_filtering()

        print("CF MAP5 Score: ", s)

        CF.append(s)

        # SLIM BPR #

        s = test_SLIM_BPR()

        print("SLIM BPR MAP5 Score: ", s)

        SLIM.append(s)

    print("CBF_CF_SLIM")
    print(CBF_CF_SLIM)
    print("CBF_CF")
    print(CBF_CF)
    print("CBF")
    print(CBF)
    print("CF")
    print(CF)
    print("SLIM")
    print(SLIM)
    print()
