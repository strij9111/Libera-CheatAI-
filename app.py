import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
import pickle
import time
import sys
import scipy.sparse as sp
from annoy import AnnoyIndex
import csv
import json
from flask import Flask, request, render_template, Response, send_file
import requests
import datetime


app = Flask(__name__)


NUM_MODELS = 8
MODEL_DIR = "./models/"
DATA_DIR = "./dataset/"

i = 0

# Для быстроты создаем словари книг пользователей
user_books = {}
books_vec = {}
user_books_mean = {}

print("Загрузка данных")

print("Загружаем данные по всем книгам")
# Загружаем данные книг
#books_data = pickle.load(open("d:\\books_data.pkcls", 'rb'))

books_data = {}

b_data = pd.read_csv(DATA_DIR + "all_books.csv")

# Выбираем самые популярные книги за последний месяц для "холодного старта"
tops = pd.read_csv(DATA_DIR + "circulation.csv", parse_dates=["datestamp"])
now = datetime.datetime.now()
last_month = now.month-1 if now.month > 1 else 12
tops = tops[(tops['datestamp'] >= datetime.datetime(
    now.year, last_month, 1, 0, 0, 0))]

x = tops.groupby(['book_id'])['book_id'].count().reset_index(
    name='count').sort_values(['count'], ascending=False)

cold_top = x['book_id'].tolist()
cold_top = cold_top[:5]

b_data = b_data.fillna('')
for i, x in b_data.iterrows():
    books_data[x['id']] = [x['id'], x['title'], x['author_fullname']]

# pickle.dump(books_data, open("d:\\books_data.pkcls", "wb"))
dataset = pickle.load(open(DATA_DIR + "dataset.pkcls", 'rb'))

i = 0
with open(DATA_DIR + "circulation.csv", 'r') as f:
    for line in f.readlines():
        if i > 0:
            idx, book_id, user_id, date, rank = line.strip().split(',')
            book_id = int(book_id)
            user_id = int(user_id)
            if user_id in user_books:
                user_books[user_id].append(book_id)
            else:
                user_books[user_id] = []
                user_books[user_id].append(book_id)
        i += 1

print("Загружаем векторы книг")
# Словарь векторов книг на основе BERT-модели
i = 0
with open(DATA_DIR + "books_bert_tsne.csv", 'r') as f:
    for line in f.readlines():
        if i > 0:
            book_id, t1, t2 = line.strip().split(',')
            book_id = int(book_id)
            t1 = float(t1)
            t2 = float(t2)
            if book_id in books_vec:
                books_vec[book_id].append([t1, t2])
            else:
                books_vec[book_id] = []
                books_vec[book_id].append([t1, t2])
        i += 1

# Создаем "усредненный" вектор прочитанных книг каждого пользователя
for id in user_books:
    readed_books = user_books[id]
    v = []
    for b in readed_books:
        if b in books_vec:
            v.append(books_vec[b])
    user_books_mean[id] = np.mean(v, axis=0).ravel()

print("Загрузка моделей")

u = AnnoyIndex(2, 'angular')
u.load(DATA_DIR + "books.ann")  # super fast, will just mmap the file

models = []
for i in range(1, NUM_MODELS + 1):
    with open(MODEL_DIR + "cluster{}.pkcls".format(i), "rb") as f:
        models.append(pickle.load(f))


@app.route('/')
def index():
    return render_template("index.tpl")


@app.route('/api', methods=['GET'])
def interact():

    user_id = int(request.args.get("id"))

    # Обрабатываем "холодный старт"
    if user_id == 0:

        top_items = []
        for item in cold_top:
            if item in books_data.keys():
                top_items.append(books_data[item])

        result = []
        for z in top_items[:5]:
            result.append({'id': z[0], 'title': z[1], 'author': z[2]})

        return json.dumps({'recommendations': result, 'history': [], "time": 0}), 200, {'content-type': 'application/json'}

    tx = time.time()
    b = user_books_mean[user_id]

    # Выбираем 6000 потенциальных кандидатов на рекомендацию
    candidates = u.get_nns_by_vector(
        b, 6000, search_k=-1, include_distances=False)

    cn = []
    for c in candidates:
        if c in dataset.mapping()[2]:
            cn.append(dataset.mapping()[2][c])

    # Получаем прочитанные книги для запрашиваемого пользователя
    b = user_books[user_id]

    for i in range(1, NUM_MODELS + 1):
        try:
            scores = models[i-1].predict(user_id, cn)
            top_items = []
            t = np.argsort(-scores)

            for item in t:
                if cn[item] in books_data.keys():
                    top_items.append(books_data[cn[item]])

            result = []
            for z in top_items[:5]:
                result.append({'id': z[0], 'title': z[1], 'author': z[2]})

            history = []
            for item in b:
                if item in books_data.keys():
                    z = books_data[item]
                    history.append({'id': z[0], 'title': z[1], 'author': z[2]})

            return json.dumps({'recommendations': result, 'history': history, "time": time.time()-tx}), 200, {'content-type': 'application/json'}

        except Exception as e:
            pass


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005)
