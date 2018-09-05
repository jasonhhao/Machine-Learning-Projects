import os
import tensorflow as tf
from PIL import Image
import numpy as np



## classes = {'label 1', 'label 2','label 3','label 4', ...} 与文件夹名称相对应


def image_encode():

    cwd = cwd = os.getcwd()
    classes = {'yes', 'no'}
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for index, name in enumerate(classes):
        class_path = cwd + '/' + name + '/'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name

            img = Image.open(img_path)
            img = img.resize((64, 64))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())

    writer.close()


def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [64, 64, 3])
    label = tf.cast(features['label'], tf.int32)

    return img, label

def load_data():


    img, label = read_and_decode("train.tfrecords")
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=1, capacity=10,
                                                    min_after_dequeue=1)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(1000):  ##训练集的样本总和
            val, l= sess.run([img_batch, label_batch])

            x_train_flatten = val.reshape(val.shape[0], -1).T
            x = x_train_flatten / 255

            if i == 0:
                x_train = x
                y_train = l
            else:
                x_train = np.hstack((x_train,x))
                y_train = np.hstack((y_train,l))


    return x_train, y_train
