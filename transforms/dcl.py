import random
from PIL import Image
from utils.util import get_image_crops


def swap(img, crop, ran):
    width_cut, hight_cut = img.size
    img = img.crop((10, 10, width_cut - 10, hight_cut - 10))
    images = get_image_crops(img, crop)
    tracker = list(range(1, 50))
    tmp_x = []
    tmp_y = []
    tmp_x_ind = []
    tmp_y_ind = []
    count_x = 0
    count_y = 0
    k = 1
    for i in range(crop[1] * crop[0]):
        tmp_x.append(images[i])
        tmp_x_ind.append(tracker[i])
        count_x += 1
        if len(tmp_x) >= k:
            tmp = tmp_x[count_x - ran:count_x]
            tmp_ind = tmp_x_ind[count_x - ran:count_x]
            combined = list(zip(tmp, tmp_ind))
            random.shuffle(combined)
            if len(combined) != 0:
                tmp, tmp_ind = zip(*combined)
            tmp_x[count_x - ran:count_x] = tmp
            tmp_x_ind[count_x - ran:count_x] = tmp_ind
        if count_x == crop[0]:
            tmp_y.append(tmp_x)
            tmp_y_ind.append(tmp_x_ind)
            count_x = 0
            count_y += 1
            tmp_x = []
            tmp_x_ind = []
        if len(tmp_y) >= k:
            tmp2 = tmp_y[count_y - ran:count_y]
            tmp2_ind = tmp_y_ind[count_y - ran:count_y]
            combined2 = list(zip(tmp2, tmp2_ind))
            random.shuffle(combined2)
            tmp2, tmp2_ind = zip(*combined2)
            tmp_y[count_y - ran:count_y] = tmp2
            tmp_y_ind[count_y - ran:count_y] = tmp2_ind
    jigsaw_ind = [item for sublist in tmp_y_ind for item in sublist]
    random_im = []
    for line in tmp_y:
        random_im.extend(line)
    width, high = img.size
    iw = int(width / crop[0])
    ih = int(high / crop[1])
    to_image = Image.new('RGB', (iw * crop[0], ih * crop[1]))
    x = 0
    y = 0
    for i in random_im:
        i = i.resize((iw, ih), Image.ANTIALIAS)
        to_image.paste(i, (x * iw, y * ih))
        x += 1
        if x == crop[0]:
            x = 0
            y += 1
    to_image = to_image.resize((width_cut, hight_cut))
    return to_image, jigsaw_ind


class RandomSwap(object):
    def __init__(self, size, swap_range):
        self.size = size
        self.ran = swap_range

    def __call__(self, img):
        return swap(img, self.size, self.ran)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
