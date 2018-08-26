import os
import cv2
import csv
import pickle
import random
import numpy as np
import sklearn.utils
import opencl4py as cl
import tensorflow as tf
import matplotlib.pyplot as plt


# %matplotlib inline


def load_data(load_model):
    data_folder = 'TrafficSignsData'
    training_file = os.path.join(data_folder, 'train.p')
    validation_file = os.path.join(data_folder, 'valid.p')
    testing_file = os.path.join(data_folder, 'test.p')

    if load_model:
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)
        return None, None, test
    else:
        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        return train, valid, None


def lenet(x, keep_prob_conv, keep_prob_fc, n_classes, img_channels):
    mu = 0
    sigma = 0.1

    # Layer 1: Convolution. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, img_channels, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.dropout(conv1, keep_prob_conv)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolution. Input = 14x14x6. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.dropout(conv2, keep_prob_conv)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = tf.contrib.layers.flatten(conv2)

    # Layer 3: Fully connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob_fc)

    # Layer 4: Fully connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob_fc)

    # TODO Make 43 (number of classes) an input variable
    # Layer 5: Fully connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    probs = tf.nn.softmax(logits)

    return logits, probs


def evaluate(X_data, y_data, batch_size, accuracy_operation, x, y, keep_prob_conv, keep_prob_fc):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        end = offset + batch_size
        batch_x = X_data[offset:end]
        batch_y = y_data[offset:end]
        accuracy = sess.run(accuracy_operation,
                            feed_dict={x: batch_x, y: batch_y, keep_prob_conv: 1.0, keep_prob_fc: 1.0})
        total_accuracy += accuracy * len(batch_x)

    return total_accuracy / num_examples


cl_queue = None
cl_context = None
cl_image_processing_kernel = None


def load_kernels():
    global cl_image_processing_kernel, cl_queue, cl_context

    platforms = cl.Platforms()
    cuda_platform = None

    for p in platforms:
        # It is hard to determine the device with the most power.
        # As my machines only have nvidia graphics cards, just filter for the nvidia CUDA platform
        if 'cuda' in p.name.lower():
            cuda_platform = p
            break

    if cuda_platform is None:
        print('No suitable device found. Exiting.')
        exit(0)

    device = cuda_platform.devices[0]
    cl_context = cuda_platform.create_context([device])

    cl_queue = cl_context.create_queue(device)
    program = cl_context.create_program(
        """
        __kernel void imageProcessing(__global const uchar* rgbImage, __global const float* parameters, __global float* grayImage) {
            size_t imgIdx = get_global_id(0);
            size_t rgbIdx = 3072 * imgIdx; 
            size_t grayIdx = 1024 * imgIdx;
            size_t paramIdx = 2 * imgIdx;
            for (int y = 0; y < 32; y++) {
                for (int x = 0; x < 32; x++) {
                    float gray = 0.21 * rgbImage[rgbIdx] + 0.72 * rgbImage[rgbIdx + 1] + 0.07 * rgbImage[rgbIdx + 2];
                    float grayEqualized = (gray - parameters[paramIdx]) * 255.0 / parameters[paramIdx + 1];
                    grayImage[grayIdx] = (grayEqualized - 128.0) / 128.0;
                    grayIdx++;
                    rgbIdx += 3;
                }
            }
        }
        """)

    cl_image_processing_kernel = program.get_kernel('imageProcessing')


def image_processing(images):
    n_images = len(images)
    param_array = np.empty((n_images, 2), dtype=np.float32)

    for idx, img in enumerate(images):
        min = np.ndarray.min(img)
        max = np.ndarray.max(img)
        param_array[idx][0] = min
        param_array[idx][1] = max - min

    image_array = np.ndarray.flatten(images)
    output = np.empty(int(images.size / 3), dtype=np.float32)

    input_buffer = cl_context.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, image_array)
    param_buffer = cl_context.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, param_array)
    output_buffer = cl_context.create_buffer(cl.CL_MEM_WRITE_ONLY | cl.CL_MEM_ALLOC_HOST_PTR, size=output.nbytes)

    cl_image_processing_kernel.set_arg(0, input_buffer)
    cl_image_processing_kernel.set_arg(1, param_buffer)
    cl_image_processing_kernel.set_arg(2, output_buffer)
    cl_queue.execute_kernel(cl_image_processing_kernel, [n_images], None)
    cl_queue.read_buffer(output_buffer, output)

    output_images = output.reshape((n_images, 32, 32, 1))
    return output_images


def rotate_img(img, angle):
    # 16 = 32 / 2, center of image
    M = cv2.getRotationMatrix2D((16, 16), angle, 1)
    rotated = cv2.warpAffine(img, M, (32, 32)).reshape((32, 32, 1))
    return rotated


def skew_img(img, shift):
    pts1 = np.float32([[0, 0], [32, 0], [0, 32], [32, 32]])
    pts2 = np.float32([[shift, shift], [32 - shift, shift], [0, 32], [32, 32]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    skewed = cv2.warpPerspective(img, M, (32, 32)).reshape((32, 32, 1))
    return skewed


class Operations:
    ROTATE = 0
    NOISE = 1
    SKEW = 2
    NUM_OPERATIONS = 3
    names = ['Rotate', 'Noise', 'Skew']


def show_image(img0, img1, img2, operation):
    print('Showing for operation: ' + Operations.names[operation])
    fig = plt.figure(figsize=(1, 3))
    fig.add_subplot(1, 3, 1)
    plt.imshow(img0)
    fig.add_subplot(1, 3, 2)
    plt.imshow(img1.reshape((32, 32)) * 128.0 + 128.0, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(1, 3, 3)
    plt.imshow(img2.reshape((32, 32)) * 128.0 + 128.0, cmap='gray', vmin=0, vmax=255)
    plt.waitforbuttonpress()
    plt.close(fig)


def show_image_new(img0, img1):
    fig = plt.figure(figsize=(1, 2))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img0)
    if img1 is not None:
        fig.add_subplot(1, 2, 2)
        plt.imshow(img1.reshape((32, 32)) * 128.0 + 128.0, cmap='gray', vmin=0, vmax=255)
    plt.waitforbuttonpress()
    plt.close(fig)


def generate_fake_data(orig_images, images, show_images):
    new_images = np.empty(images.shape)

    for i in range(len(new_images)):
        new_image = None

        operation = random.randrange(0, Operations.NUM_OPERATIONS)
        if operation == Operations.ROTATE:
            angle = random.randrange(-20, 20)
            new_image = rotate_img(images[i], angle)

        elif operation == Operations.NOISE:
            noise = np.random.randn(32, 32, 1).astype(np.float32) / 30.0
            new_image = images[i] + noise

        elif operation == Operations.SKEW:
            shift = random.randrange(-5, 5)
            new_image = skew_img(images[i], shift)

        else:
            print('Invalid operation')
            exit(0)

        if show_images and i % 100 == 0:
            show_image(orig_images[i], images[i], new_image, operation)
        new_images[i] = new_image

    return new_images


def create_sign_dict():
    sign_dict = {}
    with open('TrafficSignsData/trafficsignnames.csv') as file:
        reader = csv.reader(file, delimiter=',')
        # Skip the header
        next(reader)
        for row in reader:
            sign_dict[int(row[0])] = row[1]
    return sign_dict


def load_and_prepare_test_image(file_name):
    img = cv2.imread(file_name)
    img = np.ndarray.reshape(img, (1, 32, 32, 3))
    img = image_processing(img)
    # show_image_new(img, None)
    return img


def main():
    load_kernels()
    load_model = False
    train, valid, test = load_data(load_model)

    # TODO: Make batch_size a global constant
    # TODO: Put grayscale and normalization into one single OpenCL kernel?
    # TODO: Remove normalization option? Actually this should be done together with grayscale, should not have impact on histogram equalization
    if load_model:
        X_test, y_test = test['features'], test['labels']
        #X_test = grayscale(X_test)
        X_test = image_processing(X_test)
        #for idx, img in enumerate(X_test):
        #    show_image_new(img, X_test_p[idx])
    else:
        X_train, y_train = train['features'], train['labels']
        #X_train_orig = np.copy(X_train)
        X_train = image_processing(X_train)
        new_images = generate_fake_data(None, X_train, False)
        X_train = np.concatenate((X_train, new_images))
        y_train = np.concatenate((y_train, y_train))

        X_valid, y_valid = valid['features'], valid['labels']
        X_valid = image_processing(X_valid)

    img_width = 32
    img_height = 32
    img_channels = 1
    rate = 0.001
    batch_size = 128
    n_classes = 43

    x = tf.placeholder(tf.float32, (None, img_width, img_height, img_channels))
    y = tf.placeholder(tf.int32)
    keep_prob_conv = tf.placeholder(tf.float32)
    keep_prob_fc = tf.placeholder(tf.float32)
    one_hot_y = tf.one_hot(y, n_classes)

    logits, probs = lenet(x, keep_prob_conv, keep_prob_fc, n_classes, img_channels)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    session_path = './TrafficSignTF/Session.ckpt'

    with tf.Session() as sess:
        if load_model:
            sign_dict = create_sign_dict()
            saver.restore(sess, session_path)
            test_accuracy = evaluate(X_test, y_test, batch_size, accuracy_operation, x, y, keep_prob_conv, keep_prob_fc)
            print('Test accuracy = {:.3f}'.format(test_accuracy))

            imgs = { \
                17: load_and_prepare_test_image('TestImages/Test01_cropped.jpg'), \
                27: load_and_prepare_test_image('TestImages/Test02_cropped.jpg'), \
                5: load_and_prepare_test_image('TestImages/Test03_cropped.jpg'), \
                13: load_and_prepare_test_image('TestImages/Test04_cropped.jpg'), \
                23: load_and_prepare_test_image('TestImages/Test05_cropped.jpg') \
            }

            for sign_id, img in imgs.items():
                #sign_accuracy = sess.run(accuracy_operation, feed_dict={x: img, y: [sign_id], keep_prob_conv: 1.0, keep_prob_fc: 1.0})
                softmax_probs = sess.run(probs, feed_dict={x: img, y: [sign_id], keep_prob_conv: 1.0, keep_prob_fc: 1.0})[0]
                softmax_probs_idx = sorted(range(len(softmax_probs)), key=lambda k: softmax_probs[k], reverse=True)

                print('----------------------')
                print('Input sign: "{0}"'.format(sign_dict[sign_id]))
                print('Predicted sign: "{0}"'.format(sign_dict[softmax_probs_idx[0]]))
                #print('Sign accuracy = {:.3f}'.format(sign_accuracy))
                print('Softmax probablilites:')
                for idx in softmax_probs_idx[0:5]:
                    print('{0}: {1}'.format(sign_dict[idx], softmax_probs[idx]))
                print('----------------------')
                print()
        else:
            sess.run(tf.global_variables_initializer())
            print('Training...')
            print()
            epoch = 1
            validation_accuracy = 0

            while validation_accuracy < 0.97:
                X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
                for offset in range(0, len(X_train), batch_size):
                    end = offset + batch_size
                    batch_x = X_train[offset:end]
                    batch_y = y_train[offset:end]
                    sess.run(training_operation,
                             feed_dict={x: batch_x, y: batch_y, keep_prob_conv: 0.75, keep_prob_fc: 0.5})

                validation_accuracy = evaluate(X_valid, y_valid, batch_size, accuracy_operation, x, y, keep_prob_conv,
                                               keep_prob_fc)
                # test_accuracy = evaluate(X_train, y_train, batch_size, accuracy_operation, x, y)
                print('EPOCH {0} ...'.format(epoch))
                print('Validation accuracy = {:.3f}'.format(validation_accuracy))
                print()
                epoch += 1

            saver.save(sess, session_path)


if __name__ == '__main__':
    main()