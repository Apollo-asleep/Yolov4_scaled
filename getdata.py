import os
import glob
import numpy as np
import cv2
from threading import Thread
from tqdm import tqdm

train_data = []
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def save_images_to_text():
    imgs = glob.glob(os.path.join('dataset/images/', '*.jpg'))
    indices = np.random.randint(0, len(imgs), len(imgs) * 95 // 100)
    train_test_image = np.array(imgs)[indices.astype(int)]
    train_image = np.random.choice(train_test_image, size=len(train_test_image) * 80 // 100, replace=False)

    def train_path():
        with open('data/images/train.txt', 'w') as f:
            for image_path in train_image:
                f.write(("/content/drive/MyDrive//" + image_path).replace("\\", "/") + '\n')

    def val_path():
        with open('data/images/val.txt', 'w') as f:
            for image_path in imgs:
                if (image_path not in train_test_image):
                    f.write(("/content/drive/MyDrive//" + image_path).replace("\\", "/") + '\n')

    def test_path():
        with open('data/images/test.txt', 'w') as f:
            for image_path in train_test_image:
                if (image_path not in train_image):
                    f.write(("/content/drive/MyDrive/" + image_path).replace("\\", "/") + '\n')

    t1 = Thread(target=train_path)
    t2 = Thread(target=val_path)
    t3 = Thread(target=test_path)
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()


def change_address_save_labels_images():
    imgs = glob.glob(os.path.join('data/images/save_image_labels/', '*.jpg'))
    texts = glob.glob(os.path.join('data/images/save_image_labels/', '*.txt'))

    def save_images(imgs):
        for img in imgs:
            read_img = cv2.imread(img)
            new_img = img.replace("data/images/save_image_labels\\", '')
            cv2.imwrite('dataset/images/' + new_img, read_img)

    def save_labels(texts):
        for text in texts:
            new_text = text.replace("data/images/save_image_labels\\", '')
            with open("data/images/save_image_labels/" + new_text, 'r') as f:
                letter = f.readlines()
            with open('dataset/labels/' + new_text, 'w') as f:
                f.writelines(letter)

    t1 = Thread(target=save_images, args=(imgs,))
    t2 = Thread(target=save_labels, args=(texts,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()


def Save_image_from_video():
    count = 1
    videos = glob.glob(os.path.join('./video/', '*'))
    for video in videos:
        cap = cv2.VideoCapture(video)
        start = time.time()
        while (True):
            ret, frame = cap.read()
            if (ret):
                if ((time.time() - start < 0.6 and time.time() - start > 0.2)
                        or (time.time() - start < 1.6 and time.time() - start > 1.2)
                        or (time.time() - start < 2.6 and time.time() - start > 2.2)
                        or (time.time() - start < 4.6 and time.time() - start > 4.2)
                        or (time.time() - start < 5.6 and time.time() - start > 5.2)
                        or (time.time() - start < 9.6 and time.time() - start > 9.2)):
                    cv2.imwrite('./image/frame%d.jpg' % count, frame)
                    count += 1
            else:
                break
        cap.release()
        cv2.destroyAllWindows()


def brightness_adjustment(img):
    # turn the image into the HSV space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # creates a random bright
    ratio = .5 + np.random.uniform()
    # convert to int32, so you don't get uint8 overflow
    # multiply the HSV Value channel by the ratio
    # clips the result between 0 and 255
    # convert again to uint8
    hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.int32) * ratio, 0, 255).astype(np.uint8)
    # return the image int the BGR color space
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def image_generator():
    datagen = ImageDataGenerator(preprocessing_function=brightness_adjustment,
                                 rotation_range=20,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 channel_shift_range=4.,
                                 # vertical_flip=True,
                                 horizontal_flip=True)

    image_paths = glob.glob(os.path.join('./image/', '*'))
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (890, 890))
        images.append(image)
    datagen.fit(np.stack(images))
    count = 1
    for i in np.arange(len(images)):
        no_img = 0
        for x in datagen.flow(np.expand_dims(images[i], axis=0), batch_size=1):
            cv2.imwrite('./image_generator/frame%d.jpg' % count, (x[0]).astype(np.int32))
            print((x[0]).astype(np.int32).shape)
            count += 1
            no_img += 1
            if no_img == 15:
                break


if __name__ == '__main__':
    save_images_to_text()
    # Save_image_from_video()
    # image_generator()
    pass
