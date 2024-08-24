import cv2
import iso3166
import numpy as np
import os
import random
import requests

from datetime import date, datetime
from flask import Flask, redirect, render_template, request, url_for
from skimage import color


app = Flask(__name__)

SIMILARITY_CONSTANT = 20

def get_flag(country_code):
    resp = requests.get(f'https://flagpedia.net/data/flags/w702/{country_code}.jpg')
    if resp.status_code == 200 and 'jpeg' in resp.headers['content-type']:
        img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400, 267))
        return img
    return None

@app.route('/')
def index():
    images_path = os.path.join('static', 'images')
    if not os.path.isdir(images_path):
        os.mkdir(images_path)

    i = 1
    while os.path.isfile(os.path.join(images_path, f'overlap{i}.jpg')):
        i += 1

    return render_template(
        template_name_or_list='index.html',
        img_paths=[url_for('static', filename=f'images/overlap{n}.jpg') for n in range(1, i)]
    )

@app.route('/submit', methods=['POST'])
def submit():
    # es/se
    # si/so
    # rs/sc
    # mc/id
    # in/ie
    # fm/mn
    # sv/ee
    # dj/cu

    random.seed(datetime.combine(date.today(), datetime.min.time()).timestamp())
    country = random.choice(list(iso3166.countries_by_apolitical_name.keys()))
    iso_code = iso3166.countries_by_apolitical_name[country].alpha2.lower()

    flag1_rgb = get_flag(iso_code)
    flag2_rgb = get_flag(iso3166.countries_by_apolitical_name[request.form['country'].upper()].alpha2.lower())

    flag1_lab = color.rgb2lab(flag1_rgb)
    flag2_lab = color.rgb2lab(flag2_rgb)

    overlap = np.ones((267, 400, 3)) * 30

    color_diff = np.sqrt(np.sum((flag1_lab - flag2_lab) ** 2, axis=2))
    color_diff = np.stack([color_diff] * 3, axis=-1).astype(np.uint8)

    overlap = np.where(color_diff <= SIMILARITY_CONSTANT, flag2_rgb, overlap).astype(np.uint8)

    # row = cv2.hconcat([flag1_rgb, overlap, flag2_rgb])
    # col = cv2.vconcat([flag1_rgb, overlap, flag2_rgb])
    # cv2.imshow('row', row[:, :,::-1])
    # cv2.waitKey(0)
    # cv2.imshow('col', col[:, :,::-1])
    # cv2.waitKey(0)

    images_path = os.path.join('static', 'images')
    if not os.path.isdir(images_path):
        os.mkdir(images_path)

    i = 1
    while os.path.isfile(os.path.join(images_path, f'overlap{i}.jpg')):
        i += 1
    save_loc = os.path.join(images_path, f'overlap{i}.jpg')
    cv2.imwrite(save_loc, overlap[:, :, ::-1])
    return redirect(url_for('index'))
