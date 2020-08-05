import numpy as np 
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2 
import os, sys
 
import tensorflow as tf

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from model_data import ResNet50

### Constants for our model:
WIDTH_NORM = 224 
GRID_NUM = 11
X_SPAN = WIDTH_NORM/GRID_NUM
Y_SPAN = WIDTH_NORM/GRID_NUM
X_NORM = WIDTH_NORM/GRID_NUM
Y_NORM = WIDTH_NORM/GRID_NUM

weights_path = 'imagenet'
save_prefix = 'run224_'
learning_rate = 1e-2

#-----------------------------------------------------------------------#
def loop_body(t_true, t_pred, i, ta):
    '''
    This funtion is the main body of the custom_loss() definition, called from within the tf.while_loop()
    The loss funtion implemented here is as decsribed in the original YOLO paper: https://arxiv.org/abs/1506.02640

    # Arguments
    t_true: the ground truth tensor; shape: (batch_size, 1573)
    t_pred: the predicted tensor; shape: (batch_size, 1573)
    i: iteration cound of the while_loop
    ta: TensorArray that stores loss
    '''

    ### Get the current iteration's tru and predicted tensor
    c_true = t_true[i]
    c_pred = t_pred[i]
    ### Apply sigmoid to the coordinates part of the tensor to scale it between 0 and 1 as expected
    c_pred = tf.concat((c_pred[:605], tf.sigmoid(c_pred[-968:])), axis=0)

    ### Reshape to GRIDxGRIDxBBOXES blocks for simpler coorespondence of
    ### values across grid cell and bounding boxes
    xywh_true = tf.reshape(c_true[-968:], (11,11,2,4))
    xywh_pred = tf.reshape(c_pred[-968:], (11,11,2,4))

    ### Convert normalized values to actual ones (still relative to grid cell size)
    x_true = xywh_true[:,:,:,0] * X_NORM
    x_pred = xywh_pred[:,:,:,0] * X_NORM

    y_true = xywh_true[:,:,:,1] * Y_NORM
    y_pred = xywh_pred[:,:,:,1] * Y_NORM

    w_true = xywh_true[:,:,:,2] * WIDTH_NORM
    w_pred = xywh_pred[:,:,:,2] * WIDTH_NORM

    h_true = xywh_true[:,:,:,3] * WIDTH_NORM
    h_pred = xywh_pred[:,:,:,3] * WIDTH_NORM

    ### The below is a different approach on calculating IOU between
    ### predicted bounding boxes and ground truth
    ### See README.md for explanation for the formula
    x_dist = tf.abs(tf.subtract(x_true, x_pred))
    y_dist = tf.abs(tf.subtract(y_true, y_pred))

    ### (w1/2 +w2/2 -d) > 0 => intersection, else no intersection
    ### (h1/2 +h2/2 -d) > 0 => intersection, else no intersection
    wwd = tf.nn.relu(w_true/2 + w_pred/2 - x_dist)
    hhd = tf.nn.relu(h_true/2 + h_pred/2 - y_dist)

    area_true = tf.multiply(w_true, h_true)
    area_pred = tf.multiply(w_pred, h_pred)
    area_intersection = tf.multiply(wwd, hhd)

    iou = area_intersection / (area_true + area_pred - area_intersection + 1e-4)
    confidence_true = tf.reshape(iou, (-1,))

    ### Masks for grids that do contain an object, from ground truth
    ### The class probability block from the ground truth is used as an indicator for all grid cells that
    ### actually have an object present in itself.
    grid_true = tf.reshape(c_true[:363], (11,11,3))
    grid_true_sum = tf.reduce_sum(grid_true, axis=2)
    grid_true_exp = tf.stack((grid_true_sum, grid_true_sum), axis=2)
    grid_true_exp3 = tf.stack((grid_true_sum, grid_true_sum, grid_true_sum), axis=2)
    grid_true_exp4 = tf.stack((grid_true_sum, grid_true_sum, grid_true_sum, grid_true_sum), axis=2)

    coord_mask = tf.reshape(grid_true_exp4, (-1,))
    confidence_mask = tf.reshape(grid_true_exp, (-1,))
    confidence_true = confidence_true * confidence_mask


    ### Revised ground truth tensor, based on calculated confidence values and with non-object grids suppressed
    c_true_new = tf.concat([c_true[:363], confidence_true, c_true[-968:]], axis=0)

    ### Create masks for 'responsible' bounding box in a grid cell for loss calculation
    confidence_true_matrix = tf.reshape(confidence_true, (11,11,2))
    confidence_true_argmax = tf.argmax(confidence_true_matrix, axis=2)
    confidence_true_argmax = tf.cast(confidence_true_argmax, tf.int32)
    ind_i, ind_j = tf.meshgrid(tf.range(11), tf.range(11), indexing='ij')
    ind_argmax = tf.stack((ind_i, ind_j, confidence_true_argmax), axis=2)
    ind_argmax = tf.reshape(ind_argmax, (121,3))

    responsible_mask_2 = tf.scatter_nd(ind_argmax, tf.ones((121)), [11,11,2])
    responsible_mask_2 = tf.reshape(responsible_mask_2, (-1,))
    responsible_mask_2 = responsible_mask_2 * confidence_mask

    responsible_mask_4 = tf.scatter_nd(ind_argmax, tf.ones((121,2)), [11,11,2,2])
    responsible_mask_4 = tf.reshape(responsible_mask_4, (-1,))
    responsible_mask_4 = responsible_mask_4 * coord_mask

    ### Masks for rest of the bounding boxes
    inv_responsible_mask_2 = tf.cast(tf.logical_not(tf.cast(responsible_mask_2, tf.bool)), tf.float32)
    inv_responsible_mask_4 = tf.cast(tf.logical_not(tf.cast(responsible_mask_4, tf.bool)), tf.float32)

    ### lambda values
    lambda_coord = 5.0
    lambda_noobj = 0.5

    ### loss from dimensions ###
    dims_true = tf.reshape(c_true_new[-968:], (11,11,2,4))
    dims_pred = tf.reshape(c_pred[-968:], (11,11,2,4))

    xy_true = tf.reshape(dims_true[:,:,:,:2], (-1,))
    xy_pred = tf.reshape(dims_pred[:,:,:,:2], (-1,))

    wh_true = tf.reshape(dims_true[:,:,:,2:], (-1,))
    wh_pred = tf.reshape(dims_pred[:,:,:,2:], (-1,))

    #### XY difference loss
    xy_loss = (xy_true - xy_pred) * responsible_mask_4
    xy_loss = tf.square(xy_loss)
    xy_loss = lambda_coord * tf.reduce_sum(xy_loss)


    #### WH sqrt diff loss
    wh_loss = (tf.sqrt(wh_true) - tf.sqrt(tf.abs(wh_pred))) * responsible_mask_4
    wh_loss = tf.square(wh_loss)
    wh_loss = lambda_coord * tf.reduce_sum(wh_loss)


    ### Conf losses
    conf_true = c_true_new[363:605]
    conf_pred = c_pred[363:605]

    conf_loss_obj = (conf_true - conf_pred) * responsible_mask_2
    conf_loss_obj = tf.square(conf_loss_obj)
    conf_loss_obj = tf.reduce_sum(conf_loss_obj)


    conf_loss_noobj = (conf_true - conf_pred) * inv_responsible_mask_2
    conf_loss_noobj = tf.square(conf_loss_noobj)
    conf_loss_noobj = lambda_noobj * tf.reduce_sum(conf_loss_noobj)


    #### Class Prediction Loss
    class_true = tf.reshape(c_true_new[:363], (11,11,3))
    class_pred = tf.reshape(c_pred[:363], (11,11,3))
    class_pred_softmax = class_pred #tf.nn.softmax(class_pred)

    classification_loss = class_true - class_pred_softmax
    classification_loss = classification_loss * grid_true_exp3
    classification_loss = tf.square(classification_loss)
    classification_loss = tf.reduce_sum(classification_loss)


    ## Total loss = xy-loss + wh-loss + Confidence_loss_obj + Confidence_loss_noobj + classification_loss
    total_loss = xy_loss + wh_loss + conf_loss_obj + conf_loss_noobj + classification_loss

    #debug
    #ta_debug = ta_debug.write(0, total_loss)
    #ta_debug = ta_debug.write(1, xy_loss)
    #ta_debug = ta_debug.write(2, wh_loss)
    #ta_debug = ta_debug.write(3, conf_loss_obj)
    #ta_debug = ta_debug.write(4, conf_loss_noobj)
    #ta_debug = ta_debug.write(5, classification_loss)

    ta = ta.write(i, total_loss)
    i = i+1
    return t_true, t_pred, i, ta

def custom_loss(y_true, y_pred):
    '''
    custom loss function as per the YOLO paper, since there are no default
    loss functions in TF or Keras that fit
    '''
    c = lambda t, p, i, ta : tf.less(i, tf.shape(t)[0])
    ta = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
    #ta_debug = tf.TensorArray(tf.float32, size=1, dynamic_size=True)

    ### tf.while_loop creates a Tensorflow map with our loss function calculation (in loop_body())
    t, p, i, ta = tf.while_loop(c, loop_body, [y_true, y_pred, 0, ta])

    ### convert TensorArray into a tensor and calculate mean loss
    loss_tensor = ta.stack()
    loss_mean = tf.reduce_mean(loss_tensor)

    return loss_mean #, ta_debug.pack()
#-----------------------------------------------------------------------#

### Helper funtions for data augumentation for training the network ###
def coord_translate(bboxes, tr_x, tr_y):
    '''
    Takes a single frame's bounding box list with confidence scores and
    applies translation (addition) to the coordinates specified by 'tr'

    parameters:
    bboxes: list with element of the form ((x1,y1), (x2,y2)), (c1,c2,c3)
    tr_x, tr_y: translation factor to add the coordinates to, for x and y respectively

    returns: new list with translated coordinates and same conf scores; same shape as bboxes
    '''
    new_list = []
    for box in bboxes:
        coords = np.array(box[0])
        coords[:,0] = coords[:,0] + tr_x
        coords[:,1] = coords[:,1] + tr_y
        coords = coords.astype(np.int64)
        out_of_bound_indices = np.average(coords, axis=0) >= WIDTH_NORM
        if out_of_bound_indices.any():
            continue
        coords = coords.tolist()
        new_list.append((coords, box[1]))
    return new_list
def coord_scale(bboxes, sc):
    '''
    Takes a singl frame's bounding box list with confidence scores and
    applies scaling to the coordinates specified by sc

    parameters:
    bboxes: list with element of the form ((x1,y1), (x2,y2)), (c1,c2,c3)
    sc: scaling factor to multiply the coordinates with

    returns: new list with scaled coordinates and same conf scores; same shape as bboxes
    '''
    new_list = []
    for box in bboxes:
        coords = np.array(box[0])
        coords = coords * sc
        coords = coords.astype(np.int64)
        out_of_bound_indices = np.average(coords, axis=0) >= WIDTH_NORM
        if out_of_bound_indices.any():
            continue
        coords = coords.tolist()
        new_list.append((coords, box[1]))
    return new_list
def label_to_tensor(frame, imgsize=(WIDTH_NORM, WIDTH_NORM), gridsize=(11,11), classes=3, bboxes=2):
    '''
    This function takes in the frame (rows corresponding to a single image in the labels.csv)
    and converts it into the format our network expects (coord conversion and normalization)

    '''
    grid = np.zeros(gridsize)

    y_span = imgsize[0]/gridsize[0]
    x_span = imgsize[1]/gridsize[1]

    class_prob = np.zeros((gridsize[0], gridsize[1], classes))
    confidence = np.zeros((gridsize[0], gridsize[1], bboxes))
    dims = np.zeros((gridsize[0], gridsize[1], bboxes, 4))

    for box in frame:
        ((x1,y1), (x2,y2)), (c1,c2,c3) = box
        x_grid = int(((x1+x2)/2)//x_span)
        y_grid = int(((y1+y2)/2)//y_span)

        class_prob[y_grid, x_grid] = (c1,c2,c3)

        x_center = ((x1+x2)/2)
        y_center = ((y1+y2)/2)

        x_center_norm = (x_center-x_grid*x_span)/(x_span)
        y_center_norm = (y_center-y_grid*y_span)/(y_span)

        w = x2-x1
        h = y2-y1

        w_norm = w/imgsize[1]
        h_norm = h/imgsize[0]

        dims[y_grid, x_grid, :, :] = (x_center_norm, y_center_norm, w_norm, h_norm)

        grid[y_grid, x_grid] += 1

    tensor = np.concatenate((class_prob.ravel(), confidence.ravel(), dims.ravel()))
    return tensor
def augument_data(label, frame, imgsize=(WIDTH_NORM, WIDTH_NORM), folder='object-detection-crowdai/'):
    '''
    Takes the image file name and the frame (rows corresponding to a single image in the labels.csv)
    and randomly scales, translates, adjusts SV values in HSV space for the image,
    and adjusts the coordinates in the 'frame' accordingly, to match bounding boxes in the new image
    '''
    img = cv2.imread(folder+label)
    img = cv2.resize(img, imgsize)
    rows, cols = img.shape[:2]

    #translate_factor
    tr = np.random.random() * 0.2 + 0.01
    tr_y = np.random.randint(rows*-tr, rows*tr)
    tr_x = np.random.randint(cols*-tr, cols*tr)
    #scale_factor
    sc = np.random.random() * 0.4 + 0.8

    # flip coin to adjust image saturation
    r = np.random.rand()
    if r < 0.5:
        #randomly adjust the S and V values in HSV representation
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        fs = np.random.random() + 0.7
        fv = np.random.random() + 0.2
        img[:,:,1] *= fs
        img[:,:,2] *= fv
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # new random factor for scaling and translating
    r = np.random.rand()

    if r < 0.3:
        #translate image
        M = np.float32([[1,0,tr_x], [0,1,tr_y]])
        img = cv2.warpAffine(img, M, (cols,rows))
        frame = coord_translate(frame, tr_x, tr_y)
    elif r < 0.6:
        #scale image keeping the same size
        placeholder = np.zeros_like(img)
        meta = cv2.resize(img, (0,0), fx=sc, fy=sc)
        if sc < 1:
            placeholder[:meta.shape[0], :meta.shape[1]] = meta
        else:
            placeholder = meta[:placeholder.shape[0], :placeholder.shape[1]]
        img = placeholder
        frame = coord_scale(frame, sc)

    return img, frame
#-----------------------------------------------------------------------#

### Define generator and Import dataset (do test/train split)
def generator(label_keys, label_frames, batch_size=4, folder='object-detection-crowdai/'):
    '''
    Generator function
    # Arguments
    label_keys: image names, that are keys of the label_frames Arguments
    label_frames: array of frames (rows corresponding to a single image in the labels.csv)
    batch_size: batch size
    '''
    num_samples = len(label_keys)
    indx = label_keys

    while 1:
        shuffle(indx)
        for offset in range(0, num_samples, batch_size):
            batch_samples = indx[offset:offset+batch_size]

            images = []
            gt = []
            for batch_sample in batch_samples:
                im, frame = augument_data(batch_sample, label_frames[batch_sample])
                im = im.astype(np.float32)
                im -= 128
                images.append(im)
                frame_tensor = label_to_tensor(frame)
                gt.append(frame_tensor)

            X_train = np.array(images)
            y_train = np.array(gt)
            yield shuffle(X_train, y_train)

def plot_history(history_object):
    np.savetxt(os.path.join('../','loss.txt'),history.history['loss'])
    np.savetxt(os.path.join('../','val_loss.txt'),history.history['val_loss'])
   # np.savetxt(os.path.join('../','acc.txt'),history.history['acc'])
  #  np.savetxt(os.path.join('../','val_acc.txt'),history.history['val_acc'])
    print(history_object.history.keys())
    ### plot the training and validation loss for each epoch
    loss = history_object.history['loss'][120:]
    valloss = history_object.history['val_loss'][120:]
    plt.plot(loss) 
    plt.plot(valloss)
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    
 #   plt.plot(history_object.history['acc']) 
  #  plt.plot(history_object.history['val_acc'])
  #  plt.title('model mean squared acc')
 #   plt.ylabel('mean squared acc')
 #   plt.xlabel('epoch')
 #   plt.legend(['training set', 'validation set'], loc='upper right')
 #   plt.show()

 
#if __name__ == "__main__":
if len(sys.argv) > 3:
    weights_path = sys.argv[1]
    save_prefix = sys.argv[2]
    learning_rate = float(sys.argv[3])
elif len(sys.argv) > 2:
    weights_path = sys.argv[1]
    save_prefix = sys.argv[2]
elif len(sys.argv) > 1:
    weights_path = sys.argv[1]

model = ResNet50(include_top=False, input_shape=(WIDTH_NORM,WIDTH_NORM,3),
                load_weight=True, weights=weights_path)

with open('label_frames.p', 'rb') as f:
    label_frames = pickle.load(f)

label_keys = list(label_frames.keys())
lbl_train, lbl_validn = train_test_split(label_keys, test_size=0.2)

### Intialize generator
train_generator = generator(lbl_train, label_frames)
validation_generator = generator(lbl_validn, label_frames)
### Compile model
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])
model.summary()

model_checkpoint = ModelCheckpoint(filepath='models/' + save_prefix + str(learning_rate) + '_weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True, mode='auto', save_weights_only=True, period=1, verbose=1)
history = model.fit_generator(train_generator, validation_data=validation_generator,
                                steps_per_epoch=4, epochs=1000,
                                validation_steps=1,
                                callbacks=[model_checkpoint],verbose=1)
model.save_weights('models/'+save_prefix+str(learning_rate))

plot_history(history)
 
