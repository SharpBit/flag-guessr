import cv2
import numpy as np
import requests


def get_flag(country_code):
    resp = requests.get(f'https://flagpedia.net/data/flags/w702/{country_code}.jpg')
    if resp.status_code == 200 and 'jpeg' in resp.headers['content-type']:
        img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400, 267))
        return img
    return None


def get_main_color(img):
    pass


flag1 = get_flag('si')
print(flag1)
flag2 = get_flag('so')
print(flag2)

overlap = np.zeros((267, 400, 3))

test = np.sqrt(np.sum((flag1 - flag2) ** 2, axis=2))
test = test / np.max(test) * 255
test = np.stack([test] * 3, axis=-1)
test = test.astype(np.uint8)
print(test)
print(test.shape)
overlap = np.where(test < 94, flag2, overlap)
overlap = overlap.astype(np.uint8)

print(overlap)
print(overlap.shape)

row1 = cv2.hconcat([flag1, overlap, flag2])
# row2 = cv2.hconcat([test, overlap])
# res = cv2.vconcat([row1, row2])
cv2.imshow('res', row1[:, :,::-1])
cv2.waitKey(0)
