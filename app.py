import cv2
import numpy as np
import requests

from flask import Flask, render_template, url_for
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

# es/se
# si/so
# rs/sc
# mc/id
# in/ie
# fm/mn
# sv/ee
# dj/cu
flag1_rgb = get_flag('dj')
print(flag1_rgb)
flag2_rgb = get_flag('cu')
print(flag2_rgb)

flag1_lab = color.rgb2lab(flag1_rgb)
flag2_lab = color.rgb2lab(flag2_rgb)

overlap = np.ones((267, 400, 3)) * 30

test = np.sqrt(np.sum((flag1_lab - flag2_lab) ** 2, axis=2))
test = np.stack([test] * 3, axis=-1)
test = test.astype(np.uint8)

overlap = np.where(test <= SIMILARITY_CONSTANT, flag2_rgb, overlap)
overlap = overlap.astype(np.uint8)

print(overlap)
print(overlap.shape)
print(test)
print(test.shape)

row1 = cv2.hconcat([flag1_rgb, overlap, flag2_rgb])
# row2 = cv2.hconcat([test, overlap])
col = cv2.vconcat([flag1_rgb, overlap, flag2_rgb])
cv2.imshow('row', row1[:, :,::-1])
cv2.waitKey(0)
cv2.imshow('col', col[:, :,::-1])
cv2.waitKey(0)

cv2.imwrite('static/images/overlap1.jpg', overlap[:, :, ::-1])


@app.route('/')
def index():
    return render_template(
        template_name_or_list='index.html',
        img_path=url_for('static', filename='images/overlap1.jpg')
    )
