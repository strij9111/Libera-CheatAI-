# %%
"""
### Imports
"""

# %%
import numpy as np
import pandas as pd
from unidecode import unidecode
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm import LightFM
from skopt import forest_minimize
from scipy.sparse import csr_matrix
from dotenv import load_dotenv
import random
import sys
import pickle
from lightfm.data import Dataset
import datetime
import csv
import pickle
import itertools
import matplotlib.pyplot as plt


def get_book_features(books):
    return csv.DictReader(books, delimiter=",")


# %%
"""
### hyperparameter search
"""

# %%


# Вычеслиение map@k для LightFM
def compute_map(data, model, k=5, num_threads=1):
    res = []
    sc = []
    for k in range(1, 6):
        sc.append(precision_at_k(model, data, k=k).mean())
    res.append(np.array(sc).mean())

    return res.mean()


def sample_hyperparameters():
    """
    Yield possible hyperparameter choices.
    """

    while True:
        yield {
            "no_components": np.random.randint(16, 64),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "loss": np.random.choice(["warp", "warp-kos"]),
            "learning_rate": np.random.exponential(0.05),
            "item_alpha": np.random.exponential(1e-8),
            "user_alpha": np.random.exponential(1e-8),
            "max_sampled": np.random.randint(5, 15),
            "num_epochs": 31,
        }


def random_search(test, num_samples=8, num_threads=1):
    """
    Sample random hyperparameters, fit a LightFM model, and evaluate it
    """
    i = 1
    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        num_epochs = hyperparams.pop("num_epochs")

        model = LightFM(**hyperparams)

#        for epoch in range(60):
#            model.fit_partial(test, epochs=1, verbose=True)
#            sc = []
#            for k in range(1, 6):
#                sc.append(precision_at_k(model, test, k=k).mean())
#            warp_auc.append(np.array(sc).mean())

        model.fit(test, epochs=31, verbose=True)
        pickle.dump(model, open('cluster{}.pkcls'.format(i), "wb"))
        i += 1
#        model.fit(test, epochs=num_epochs,
#                  num_threads=num_threads, verbose=True)

        score = compute_map(test, model, num_threads=80)

        hyperparams["num_epochs"] = num_epochs

        print("score {} at {}".format(score, hyperparams))

        yield (score, hyperparams, model)


# %%
"""
### Create datasets
"""

# %%

MODEL_DIR = "./models/"
DATA_DIR = "./dataset/"

books = pd.read_csv(DATA_DIR + "books.csv", index_col=False)
books_tsne = pd.read_csv(DATA_DIR + "ann_tsne.csv", index_col=False)
books_bert_tsne = pd.read_csv(DATA_DIR + "books_bert_tsne.csv", index_col=False)

cluster1 = pd.read_csv(DATA_DIR + "ann_tsne_cluster.csv", index_col=False)
cluster2 = pd.read_csv(DATA_DIR + "ann_tsne_cluster_cosine.csv", index_col=False)
cluster3 = pd.read_csv(DATA_DIR + "ann_tsne_cluster_4.csv", index_col=False)
cluster4 = pd.read_csv(DATA_DIR + "ann_tsne_cluster_20.csv", index_col=False)
cluster5 = pd.read_csv(DATA_DIR + "ann_tsne_cluster_110.csv", index_col=False)
cluster_bert = pd.read_csv(DATA_DIR + "books_bert_clusters_2.csv", index_col=False)

books_bert_tsne = pd.concat([books_bert_tsne, cluster_bert], axis=1)

books_tsne = pd.merge(books_tsne, cluster1, how='left')
books_tsne = pd.merge(books_tsne, cluster2, how='left')
books_tsne = pd.merge(books_tsne, cluster3, how='left')
books_tsne = pd.merge(books_tsne, cluster4, how='left')
books_tsne = pd.merge(books_tsne, cluster5, how='left')

books_tsne.drop(['Silhouette'], axis=1, inplace=True)
books_bert_tsne.drop(['Silhouette'], axis=1, inplace=True)
books = pd.merge(books, books_tsne, on='book_id', how='left')
books = pd.merge(books, books_bert_tsne, on='book_id', how='left')

circulation = pd.read_csv(
    DATA_DIR + "circulation.csv", parse_dates=['datestamp'], dtype={'book_id': np.int32, 'user_id': np.int32, 'rating': np.int8})

# circulation['diff'] = circulation.groupby(
#    'user_id').datestamp.transform(pd.Series.diff)

circulation = circulation[(circulation['datestamp']
                           >= datetime.datetime(2021, 1, 1, 0, 0, 0))]

# circulation = circulation[(circulation['datestamp']
#                           <= datetime.datetime(2020, 12, 31, 0, 0, 0))]

circulation.drop(['datestamp', 'Unnamed: 0'], axis=1, inplace=True)

# Подсчет количества взятых книг
x = circulation.groupby(['book_id'])['book_id'].count().reset_index(
    name='count').sort_values(['count'], ascending=False)

# Данные о книгах, которые брали за всё время 1 раз, считаем нерелевантными
x = x[(x['count']) < 2]
circulation = circulation[~circulation['book_id'].isin(x['book_id'])]

# Убираем книги из books, которых нет в circulation
books = books[(books['book_id'].isin(circulation['book_id']))]

# Фильтруем кластеры
# Кластер по ru_core_news_lg
#books = books[(books['Cluster4'] == 'C1')]
# Кластер по LaBSE
#books = books[(books['bert_cluster'] == 'C1')]

# Убираем книги из circulation, которых нет в books
circulation = circulation[circulation['book_id'].isin(books['book_id'])]

books_features = {}

# Собираем данные о книгах
for i, x in books.iterrows():
    books_features[x['book_id']] = (x['book_id'], x['bert_t1'], x['bert_t2'], x['bert_cluster'], x['Cluster1'], x['Cluster5'], x['Cluster4'], x['Cluster3'], x['Cluster2'], x['author_fullname'], x['changed'], x['annotation'], x['bibliographylevel'], x['language_id'], x['material_id'], x['agerestriction_id'],
                                    x['rubric_id'], x['author_id'], x['publisher_id'], x['place_id'], x['serial_id'], x['year_value'], x['totaloutcount'], x['available'])

book_ids = []
user_ids = []
mixed_ids = []
book_mixed = []
bf = []

for i, x in circulation.iterrows():
    user_ids.append(x['user_id'])
    book_ids.append(x['book_id'])
    mixed_ids.append([x['user_id'], x['book_id']])
#    book_mixed.append(books_features[x['book_id']])
    bf.append(books_features[x['book_id']])


bf = pd.DataFrame(bf, columns=['book_id', 'bert_t1', 'bert_t2', 'bert_cluster', 'Cluster1', 'Cluster5', 'Cluster4', 'Cluster3', 'Cluster2', 'author_fullname', 'changed', 'annotation', 'bibliographylevel', 'language_id', 'material_id', 'agerestriction_id',
                  'rubric_id', 'author_id', 'publisher_id', 'place_id', 'serial_id', 'year_value', 'totaloutcount', 'available'])

bf.to_csv(DATA_DIR + "bf.csv")

# %%
"""
### Building interactions matrix
"""

# %%


with open(DATA_DIR + "bf.csv", 'r', newline='', encoding="utf-8") as csvfile:
    books1 = get_book_features(csvfile)

    print(len(list(books1)), len(book_ids))
    dataset = Dataset()
    dataset.fit(user_ids, book_ids)

    (interactions, weights) = dataset.build_interactions(mixed_ids)

    dataset.fit_partial(items=(x['book_id'] for x in books1),
                        item_features=((x['title'], x['author_fullname'], x['annotation'], x['bibliographylevel'], x['changed'], x['publicationtype'], x['language_id'], x['material_id'], x['agerestriction_id'], x['rubric_id'], x['author_id'], x['author_dates'], x['publisher_id'], x['place_id'], x['serial_id'], x['year_value'], x['totaloutcount'], x['available'], x['is_ebook'], x['is_portr'], x['is_il'], x['is_cv']) for x in books1))

    item_features = dataset.build_item_features(((x['book_id'], [x['title'], x['author_fullname'], x['annotation'], x['bibliographylevel'], x['changed'], x['publicationtype'], x['language_id'], x['material_id'], x['agerestriction_id'], x['rubric_id'], x['author_id'], x['author_dates'], x['publisher_id'], x['place_id'], x['serial_id'], x['year_value'], x['totaloutcount'], x['available'], x['is_ebook'], x['is_portr'], x['is_il'], x['is_cv']])
                                                 for x in books1))

    pickle.dump(dataset, open(DATA_DIR + "dataset.pkcls", "wb"))
# %%
"""
### hyperparameters search
"""

# %%

(score, hyperparams, model) = max(random_search(
    interactions, num_threads=80), key=lambda x: x[0])

print("Best score {} at {}".format(score, hyperparams))


# %%
"""
### model training on a full dataset
"""

# %%

# hyperparams = {'no_components': 43, 'learning_schedule': 'adagrad', 'loss': 'warp-kos', 'learning_rate': 0.048768266170686546,
#               'item_alpha': 1.4745729251128184e-08, 'user_alpha': 1.4942739625538447e-08, 'max_sampled': 9}
#model = LightFM(**hyperparams)
#model.fit(interactions, epochs=31, verbose=True)
