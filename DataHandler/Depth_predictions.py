
input_shape = (512, 384)
DATASET = 'BDD10k'
input_path = '/MLDatasetsStorage/exjobb/' + DATASET + '/images/'
output_path = '/MLDatasetsStorage/exjobb/' + DATASET + '/depth/'
sets = ['train', 'val', 'test']
PYDNET_SAVED_WEIGHTS = '/WeightModels/exjobb/OldStuff/pydnet_weights/pydnet_org/pydnet'

# Create list of directory
for set in sets:
    img_list = os.listdir(input_path + set)
    nbr_img = len(img_list)

    # Predict disparity-map from pydnet
    with tf.Graph().as_default():
        placeholders = {'im0': tf.placeholder(tf.float32, [None, None, None, 3], name='im0')}

        with tf.variable_scope("model") as scope:
            model = pydnet(placeholders)

        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        loader = tf.train.Saver()


            with tf.Session() as sess:
                sess.run(init)
                loader.restore(sess, PYDNET_SAVED_WEIGHTS)
                pyd_image = cv2.resize(image, (input_shape[1], input_shape[0])).astype(np.float32)
                pyd_image = np.expand_dims(pyd_image, 0)
                disp = sess.run(model.results[0], feed_dict={placeholders['im0']: pyd_image})
                left_disp = disp[0, :, :, 0]*3.33
                left_disp = np.expand_dims(left_disp, 3)




