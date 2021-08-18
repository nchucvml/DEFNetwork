from nets import vgg
from nets import resnet_v2
from nets import densenet
from nets import inception_v3
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import input_data
import argparse
import time
import os

slim = tf.contrib.slim

parser = argparse.ArgumentParser(description='DEF Processing')
parser.add_argument('--mode', default='train', help='select your action, type \'train\' or \'teat\'')
# input param
parser.add_argument('--source_data_dir', default='./tfrecord', help='the location of your dataset')
parser.add_argument('--num_classes', default=9, help='the number of categories in your dataset')
parser.add_argument('--num_train', default=1637, help='the number of images in your training data')
parser.add_argument('--num_test', default=333, help='the number of images in your testing data')
# output param
parser.add_argument('--weights_dir', default='./result', help='the location of your training weights')
parser.add_argument('--logs_dir', default='./logs', help='the location of your logs')
parser.add_argument('--logits_dir', default='./logits', help='the location of your logits')
# training param
parser.add_argument('--training_batch_size', default=5, help='training param, your batch size')
parser.add_argument('--training_learning_rate', default=1e-5, help='training param, your learning rate')
parser.add_argument('--training_epochs', default=500, help='training param, your training epochs')
# testing param
parser.add_argument('--testing_your_pretrained_weights_path', default='./result/fine_tune.ckpt-500',
                    help='testing param, your pretrained weights path. If you have no pretrained weights, type \'None\'')
parser.add_argument('--testing_your_dataset_labesls', default=['1', '2', '5', '10', '20', '50', '100', '500', '1000'],
                    help='testing param, your dataset labels')

args = parser.parse_args()

# dataset parameters
NUM_CLASSES = int(args.num_classes)
NUM_TRAIN = int(args.num_train)
NUM_VAL = int(args.num_test)
DATA_DIR = args.source_data_dir
IMAGE_SIZE = 224

weights_dir = args.weights_dir
trained_file = 'fine_tune.ckpt'
logs_dir = args.logs_dir
logits_dir = args.logits_dir

# training parameters
batch_size = int(args.training_batch_size)
learning_rate = float(args.training_learning_rate)
training_epochs = int(args.training_epochs)
display_epoch = 1
train_num_batch = int(np.ceil(NUM_TRAIN / batch_size))
val_num_batch = int(np.ceil(NUM_VAL / batch_size))

# testing parameters
checkpoint_file_myown = args.testing_your_pretrained_weights_path
data_label = args.testing_your_dataset_labesls

# pretrained parameters
checkpoint_file_vgg = './pre_train/vgg_19.ckpt'
checkpoint_file_googlenet = './pre_train/inception_v3.ckpt'
checkpoint_file_densenet = './pre_train/tf-densenet121.ckpt'
checkpoint_file_resnet = './pre_train/resnet_v2_50.ckpt'


def my_fus(inputs, num_classes, is_training):
    # obtain the feature maps of CNNs: vgg, googlenet, resnet, densenet
    net_vgg = vgg.vgg_19(inputs, num_classes, is_training)
    net_google, _ = inception_v3.inception_v3(inputs, num_classes, is_training)
    net_resnet, _ = resnet_v2.resnet_v2_50(inputs, num_classes, is_training)
    net_densenet, _ = densenet.densenet121(inputs, num_classes, is_training)

    net = tf.concat([net_vgg, net_google, net_resnet, net_densenet], -1)

    # for the ensemble feature networks
    net_vgg = slim.conv2d(net_vgg, NUM_CLASSES, [1, 1], activation_fn=None, normalizer_fn=None, scope='vgg')
    net_vgg = tf.squeeze(net_vgg, [1, 2], name='SpatialSqueeze')
    net_vgg = slim.softmax(net_vgg, scope='konet')

    net_google = slim.conv2d(net_google, NUM_CLASSES, [1, 1], activation_fn=None, normalizer_fn=None, scope='google')
    net_google = tf.squeeze(net_google, [1, 2], name='SpatialSqueeze')
    net_google = slim.softmax(net_google, scope='konet')

    net_resnet = slim.conv2d(net_resnet, NUM_CLASSES, [1, 1], activation_fn=None, normalizer_fn=None, scope='resnet')
    net_resnet = tf.squeeze(net_resnet, [1, 2], name='SpatialSqueeze')
    net_resnet = slim.softmax(net_resnet, scope='konet')

    net_densenet = slim.conv2d(net_densenet, NUM_CLASSES, [1, 1], activation_fn=None, normalizer_fn=None, scope='densenet')
    net_densenet = tf.squeeze(net_densenet, [1, 2], name='SpatialSqueeze')
    net_densenet = slim.softmax(net_densenet, scope='konet')

    # for the decision networks
    net = slim.conv2d(net, 512, [1, 1], scope='ko1')
    net = slim.conv2d(net, 1024, [1, 1], scope='ko2')
    net = slim.conv2d(net, 1024, [1, 1], scope='ko3')
    net = slim.conv2d(net, NUM_CLASSES, [1, 1], activation_fn=None, normalizer_fn=None, scope='ko4')
    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
    net = slim.softmax(net, scope='konet')

    return net, net_vgg, net_google, net_resnet, net_densenet


def Train():
    if not tf.gfile.Exists(weights_dir):
        tf.gfile.MakeDirs(weights_dir)
    if not tf.gfile.Exists(logs_dir):
        tf.gfile.MakeDirs(logs_dir)
    if not tf.gfile.Exists(logits_dir):
        tf.gfile.MakeDirs(logits_dir)
    # load preprocessed data
    train_images, train_labels = input_data.load_batch_inception(DATA_DIR, batch_size, NUM_CLASSES, True, IMAGE_SIZE,
                                                                 IMAGE_SIZE)
    test_images, test_labels = input_data.load_batch_inception_test(DATA_DIR, batch_size, NUM_CLASSES, False,
                                                                    IMAGE_SIZE, IMAGE_SIZE)
    # declare the placeholders
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    input_labels = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES])

    is_training = tf.placeholder(dtype=tf.bool)

    logits, logits_vgg, logits_google, logits_resnet, logits_densenet = my_fus(input_images,
                                                                               num_classes=NUM_CLASSES,
                                                                               is_training=is_training)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=logits))

    cost_vgg = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=logits_vgg))
    cost_google = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=logits_google))
    cost_resnet = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=logits_resnet))
    cost_densenet = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=logits_densenet))

    # deep network loss + decision loss
    cost = (cost + cost_vgg + cost_google + cost_resnet + cost_densenet) / 5
    loss_summary = tf.summary.scalar('loss', cost)

    model_variables = slim.get_model_variables()
    model_vgg_variables = tf.train.list_variables(checkpoint_file_vgg)
    model_google_variables = tf.train.list_variables(checkpoint_file_googlenet)
    model_resnet_variables = tf.train.list_variables(checkpoint_file_resnet)
    model_densenet_variables = tf.train.list_variables(checkpoint_file_densenet)

    # for the pretrained weights loading
    global_variables = [v.name[:-2] for v in model_variables]

    vgg_variables = [v[0] for v in model_vgg_variables]
    google_variables = [v[0] for v in model_google_variables]
    resnet_variables = [v[0] for v in model_resnet_variables]
    densenet_variables = [v[0] for v in model_densenet_variables]

    variables_name_can_be_restored_vgg = list(set(global_variables).intersection(set(vgg_variables)))
    variables_name_can_be_restored_google = list(set(global_variables).intersection(set(google_variables)))
    variables_name_can_be_restored_resnet = list(set(global_variables).intersection(set(resnet_variables)))
    variables_name_can_be_restored_densenet = list(set(global_variables).intersection(set(densenet_variables)))

    variables_can_be_restored_vgg = [var for var in model_variables if
                                     var.name[:-2] in variables_name_can_be_restored_vgg]
    variables_can_be_restored_google = [var for var in model_variables if
                                        var.name[:-2] in variables_name_can_be_restored_google]
    variables_can_be_restored_resnet = [var for var in model_variables if
                                        var.name[:-2] in variables_name_can_be_restored_resnet]
    variables_can_be_restored_densenet = [var for var in model_variables if
                                          var.name[:-2] in variables_name_can_be_restored_densenet]

    restorer_vgg = tf.train.Saver(variables_can_be_restored_vgg)
    restorer_google = tf.train.Saver(variables_can_be_restored_google)
    restorer_resnet = tf.train.Saver(variables_can_be_restored_resnet)
    restorer_densenet = tf.train.Saver(variables_can_be_restored_densenet)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    logits = (logits + logits_vgg + logits_google + logits_resnet + logits_densenet) / 5
    pred = tf.argmax(logits, axis=1)
    correct = tf.equal(pred, tf.argmax(input_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    acc_summary = tf.summary.scalar('accuracy', accuracy)

    # save 5 weights at most
    save = tf.train.Saver(max_to_keep=5)

    config = tf.ConfigProto(allow_soft_placement=False)
    config.gpu_options.allow_growth = True

    merged = tf.summary.merge([loss_summary, acc_summary])

    # start to flow into the data
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(logs_dir, sess.graph)

        # custom pretrained weights
        # ckpt = tf.train.latest_checkpoint(save_dir)
        # ckpt = './result/fine_tune.ckpt-175'
        # save.restore(sess, ckpt)

        # 4 pretrained weights of the deep networks
        print('Loading checkpoint %s' % checkpoint_file_vgg)
        restorer_vgg.restore(sess, checkpoint_file_vgg)
        print('Loading checkpoint %s' % checkpoint_file_googlenet)
        restorer_google.restore(sess, checkpoint_file_googlenet)
        print('Loading checkpoint %s' % checkpoint_file_resnet)
        restorer_resnet.restore(sess, checkpoint_file_resnet)
        print('Loading checkpoint %s' % checkpoint_file_densenet)
        restorer_densenet.restore(sess, checkpoint_file_densenet)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('Start！')
        fileCnt_logits = 0
        for epoch in range(training_epochs):
            fileCnt_logits += 1
            train_loss = 0.0
            train_acc = 0.0
            for i in range(train_num_batch):
                imgs, labs, = sess.run([train_images, train_labels])
                _, loss, train_accuracy, train_summaries = sess.run([optimizer, cost, accuracy, merged],
                                                                    feed_dict={input_images: imgs,
                                                                               input_labels: labs,
                                                                               is_training: True})
                train_loss += loss
                train_acc += train_accuracy

            if epoch % display_epoch == 0:
                train_writer.add_summary(train_summaries, epoch)
                train_accuracy = sess.run(accuracy,
                                          feed_dict={input_images: imgs, input_labels: labs, is_training: False})

            print('Epoch {}/{}  average cost {:.4f}  train accuracy {:.4f}'.format(epoch + 1,
                                                                                   training_epochs,
                                                                                   train_loss / train_num_batch,
                                                                                   train_acc / train_num_batch))

            val_accuracy = 0.0
            val_loss = 0.0
            feat_softmax = [[]]
            pre_lab = []
            gt_lab = []
            for j in range(val_num_batch):
                imgs, labs = sess.run([test_images, test_labels])
                pred_value = sess.run(pred, feed_dict={input_images: imgs, is_training: False})
                if j == 0:
                    feat_softmax = sess.run(logits, feed_dict={input_images: imgs, is_training: False})
                else:
                    feat_softmax = np.concatenate(
                        (feat_softmax, sess.run(logits, feed_dict={input_images: imgs, is_training: False})),
                        axis=0)
                cost_values, accuracy_values = sess.run([cost, accuracy],
                                                        feed_dict={input_images: imgs, input_labels: labs,
                                                                   is_training: False})
                val_accuracy += accuracy_values
                val_loss += cost_values
                pre_lab = np.concatenate((pre_lab, pred_value), axis=-1)
                gt_lab = np.concatenate((gt_lab, np.argmax(labs, 1)), axis=-1)
            pre_lab = pre_lab.tolist()
            gt_lab = gt_lab.tolist()
            print('Epoch {}/{}  Test cost {:.4f} Test accuracy {:.4f}'.format(epoch + 1, training_epochs,
                                                                              val_loss / val_num_batch,
                                                                              val_accuracy / val_num_batch))
            cm = confusion_matrix(gt_lab, pre_lab)
            print(cm)
            np.savetxt(logits_dir + '/' + str(fileCnt_logits) + '_label.csv', gt_lab, delimiter=',')
            np.savetxt(logits_dir + '/' + str(fileCnt_logits) + '_logits.csv', feat_softmax, delimiter=',')

            if (epoch + 1) % 100 == 0:
                save.save(sess, os.path.join(weights_dir, trained_file), global_step=epoch + 1)
                print('Epoch {}/{}  save model successfully!'.format(epoch + 1, training_epochs))

        print('train finish!')
        coord.request_stop()
        coord.join(threads)


def Test():
    # load data
    test_images, test_labels = input_data.load_batch_inception_test(DATA_DIR, batch_size, NUM_CLASSES, False,
                                                                    IMAGE_SIZE, IMAGE_SIZE)
    # declare the placeholders
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    is_training = tf.placeholder(dtype=tf.bool)

    logits, logits_vgg, logits_google, logits_resnet, logits_densenet = my_fus(input_images,
                                                                               is_training=is_training,
                                                                               num_classes=NUM_CLASSES)
    logits = (logits + logits_vgg + logits_google + logits_resnet + logits_densenet) / 5

    pred = tf.argmax(logits, axis=1)
    restorer = tf.train.Saver()

    # start to flow into the data
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = checkpoint_file_myown

        if ckpt != 'None':
            restorer.restore(sess, ckpt)
            print("Model restored.")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        pd_lab = []
        gt_lab = []
        start = time.time()

        for i in range(val_num_batch):
            imgs, labs = sess.run([test_images, test_labels])
            pred_value = sess.run(pred, feed_dict={input_images: imgs, is_training: False})
            pd_lab = np.concatenate((pd_lab, pred_value), axis=-1)
            gt_lab = np.concatenate((gt_lab, np.argmax(labs, 1)), axis=-1)
        end = time.time()
        pd_lab = pd_lab.tolist()
        gt_lab = gt_lab.tolist()
        pd_lab = [int(i) for i in pd_lab]
        gt_lab = [int(i) for i in gt_lab]
        print('pre length:', len(pd_lab))
        print('gt_lab length:', len(gt_lab))
        print('pd_lab: ', pd_lab)
        print('gt_lab: ', gt_lab)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plot_confusion_matrix(gt_lab, pd_lab, classes=np.array(data_label), title='Confusion matrix')
        plt.show()
        print(classification_report(gt_lab, pd_lab, target_names=np.array(data_label)))
        print('Accuracy: ', metrics.accuracy_score(gt_lab, pd_lab))
        print('=============================================================')
        print('Precision macro: ', metrics.precision_score(gt_lab, pd_lab, average='macro'))
        print('Precision weighted: ', metrics.precision_score(gt_lab, pd_lab, average='weighted'))
        print('=============================================================')
        print('Recall macro: ', metrics.recall_score(gt_lab, pd_lab, average='macro'))
        print('Recall weighted: ', metrics.recall_score(gt_lab, pd_lab, average='weighted'))
        print('=============================================================')
        print('f-measure macro: ', metrics.f1_score(gt_lab, pd_lab, average='macro'))
        print('f-measure weighted: ', metrics.f1_score(gt_lab, pd_lab, average='weighted'))
        print('testing time', (end - start) / NUM_VAL)
        coord.request_stop()
        coord.join(threads)
    tf.reset_default_graph()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Ground Truth Label',
           xlabel='Predicted Label')
    plt.xlabel('Predicted Label', fontname="Times New Roman", fontsize=10)
    plt.ylabel('Ground Truth Label', fontname="Times New Roman", fontsize=10)
    plt.title('Confusion Matrix', fontname="Times New Roman", fontsize=10)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family("Times New Roman")
        l.set_size(10)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontname="Times New Roman",
             fontsize=10)
    plt.setp(ax.get_yticklabels(), fontname="Times New Roman", fontsize=10)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), fontname="Times New Roman", fontweight="bold", fontsize=10,
                    ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return ax


if __name__ == '__main__':
    tf.reset_default_graph()

    if args.mode == 'train':
        Train()
    elif args.mode == 'test':
        Test()
    else:
        print('select your action, type --mode \'train\' or \'test\'')