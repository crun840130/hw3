import matplotlib.image
import os
import numpy as np
from glob import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as im
from scipy import misc
import PIL
import pickle
def get_File(file_dir,shuffle=False):
    images = []
    subfolders = []
    dirPath=glob(file_dir+"/*")
    image_list=[]
    for label in os.listdir(file_dir):
        file_list=glob(os.path.join(file_dir,label)+"/*")
        for i,f in enumerate(file_list):
            file_list[i]=f.replace("\\","/")
        image_list=image_list+file_list
        subfolders=subfolders+[label]*len(file_list)
    image_list=np.array(image_list)
    subfolders=np.array(subfolders)
    if shuffle==True:
        n=len(image_list)
        temp=np.random.permutation([i for i in range(n)])
        image_list=image_list[temp]
        subfolders=subfolders[temp]
    return image_list, subfolders
def toDigitLabel(labels):
    label2digit = {}
    digit2label = []
    digit_label = []
    for l in labels:
        if l in label2digit:
            digit_label.append(label2digit[l])
        else:
            label2digit[l] = len(digit2label)
            digit_label.append(len(digit2label))
            digit2label.append(l)
    return label2digit, digit2label, digit_label
def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_TFRecord(images, labels, filename,h=128,w=128,c=3):
    n = len(labels)
    TFWriter = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start...')
    for i in np.arange(0,n):
        try:
            image = im.imread(images[i])
            image=misc.imresize(image,[h,w,c])
            if i==22:
                print("???")
            if len(image.shape)==2:
                temp=np.zeros([h,w,c])
                temp[:,:,0]=image
                temp[:,:,1]=image
                temp[:,:,2]=image
                image=temp.astype(np.uint8)

            if image.shape[0]!=128 or image.shape[1]!=128 or image.shape[2]!=3:
                print("fucke")
            if image is None:
                print('Error image:' + images[i])
            else:
                image_raw = image.tostring()
                if len(image_raw)!=49152:
                    print("fucl")
                # imagetemp=bytes_feature(image_raw)
                # imagetemp = tf.decode_raw(imagetemp, tf.uint8)
                # sess=tf.Session()
                # imagetemp2=sess.run(imagetemp)
                # if imagetemp2[1] !=49152:
                #     print("fuck")

            label = int(labels[i])
            ftrs = tf.train.Features(
                feature={'Label': int64_feature(label),
                         'image_raw': bytes_feature(image_raw)}
            )
            example = tf.train.Example(features=ftrs)
            TFWriter.write(example.SerializeToString())
        except IOError as e:
            print('Skip!\n')
    TFWriter.close()
    print('Transform done!')

def parse_exmp(serial_exmp):
    img_features = tf.parse_single_example(
        serial_exmp,
        features={'Label': tf.FixedLenFeature([], tf.int64),
                  'image_raw': tf.FixedLenFeature([], tf.string), })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [128, 128, 3])
    image = tf.cast(image, tf.float32)
    label = img_features['Label']
    # shape = tf.cast(feats['shape'], tf.int32)
    return image, label
def read_and_decode(filename, batch_size,h=128,w=128,c=3):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_exmp)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    image_batch, label_batch = iterator.get_next()
    return image_batch, label_batch
# def read_and_decode(filename, batch_size,h=128,w=128,c=3):
#     filename_queue = tf.train.string_input_producer([filename],
#                                                     num_epochs=None)
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     img_features = tf.parse_single_example(
#         serialized_example,
#         features={'Label': tf.FixedLenFeature([], tf.int64),
#                   'image_raw': tf.FixedLenFeature([], tf.string), })
#     image = tf.decode_raw(img_features['image_raw'], tf.uint8)
#     image = tf.reshape(image, [h, w, c])
#     image = tf.cast(image, tf.float32)
#     label = tf.cast(img_features['Label'], tf.int64)
#     # dataset
#     # image_batch, label_batch = tf.data.Dataset.
#     image_batch, label_batch = tf.train.shuffle_batch(
#         [image, label],
#         batch_size=batch_size,
#         capacity= 10000,
#         num_threads=4,
#         min_after_dequeue=1000)
#     return image_batch, label_batch
def generate_TFRecord(dirName,testdirName,fileName):
    image_list, label_list = get_File(dirName, shuffle=False)
    timage_list,tlabel_list= get_File(testdirName,shuffle=False)
    label2digit, digit2label, digit_label = toDigitLabel(label_list)
    tdigit_label=[label2digit[label] for label in tlabel_list]
    convert_to_TFRecord(image_list, digit_label, "cifar101.tfrecords")
    convert_to_TFRecord(timage_list,tdigit_label,"cifar101test.tfrecords")
    with open("labeltranslate.pickle", 'wb') as f:
        pickle.dump([label2digit, digit2label, digit_label],f)
# prepocess=1
#
# if prepocess==1:
#     image_list,label_list=get_File("train",shuffle=True)
#     label2digit, digit2label, digit_label=toDigitLabel(label_list)
#     convert_to_TFRecord(image_list,digit_label,"cifar101.tfrecords")
#     with open("labeltranslate.pickle", 'wb') as f:
#         pickle.dump([label2digit, digit2label, digit_label],f)
#
# else:
#     filename = "cifar101.tfrecords"
#     batch_size = 1
#     Label_size = 101
#     image_batch, label_batch = read_and_decode(filename, batch_size)
#     # image_batch_train = tf.reshape(image_batch, [-1, 3*42*42])
#     image_batch_train=image_batch
#     label_batch_train = tf.one_hot(label_batch, Label_size)
#     with tf.Session() as sess:
#         sess.run(tf.local_variables_initializer())
#         sess.run(tf.global_variables_initializer())
#         # sess.run(tf.group(tf.global_variables_initializer(),tf.global_variables_initializer()))
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord,sess=sess)
#         image_data, label_data = sess.run([image_batch_train, label_batch_train])
#         coord.request_stop()
#         coord.join(threads)
#
# print("")