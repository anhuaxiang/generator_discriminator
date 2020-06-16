import scipy
import numpy as np
from glob import glob
from PIL import Image


class DataLoader:
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.test_dataset_name = 'DIV2K_valid_HR'
        self.img_res = img_res
        self.lr_img_res = int(self.img_res[0] / 4), int(self.img_res[1] / 4)

    def load_data(self, batch_size=1, is_testing=False):

        path = glob(f'data/{self.dataset_name}/*.png')
        test_path = glob(f'data/{self.test_dataset_name}/*.png')

        if is_testing:
            batch_images = np.random.choice(test_path, size=batch_size)
        else:
            batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            img_hr, img_lr = self.im_read(img_path)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr

    def im_read(self, path):
        img = Image.open(path)
        hr_img = img.resize(self.img_res)
        lr_img = img.resize(self.lr_img_res)
        return np.array(hr_img).astype(np.float), np.array(lr_img).astype(np.float)

    def read_predict_image(self, path):
        img = Image.open(path)
        w, h = img.height, img.width
        img = np.array(img.resize(self.lr_img_res))
        return img / 127.5 - 1, h * 4, w * 4

    @classmethod
    def reverse_image(cls, data, h, w, save_path='result.png'):
        if data.ndim > 3:
            data = data[0]
        data = (data + 1) * 127.5
        data = data.astype(np.uint8)
        image = Image.fromarray(data)
        image = image.resize((h, w))
        image.show()
        image.save(save_path)


if __name__ == '__main__':
    dl = DataLoader('DIV2K_valid_HR')
    dl.load_data(1)
