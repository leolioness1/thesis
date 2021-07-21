import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
# Clear any logs from previous runs
#
# HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd','nadam']))

METRIC_BCE = 'binary_crossentropy'
HP_DROPOUT_LOSS= hp.HParam('loss', hp.RealInterval(0.1, 0.2))

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_BCE, display_name='Binary CE')],
  )


smooth = 1e-12
def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


def dice_coef(y_true, y_pred, smooth=1):
    """
    Arguments:
        y_true: (string) ground truth image mask
        y_pred : (int) predicted image mask

    Returns:
        Calculated Dice coeffecient
    """
    y_true_f = K.flatten(y_true)
    y_true_f = K.cast(y_true_f, 'float32')
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    """
    Arguments:
        y_true: (string) ground truth image mask
        y_pred : (int) predicted image mask

    Returns:
        Calculated Dice coeffecient loss
    """
    return 1 - dice_coef(y_true, y_pred)

def dice_score(y_true, y_pred, smooth=1, threshold = 0.6):
    """
    Arguments:
        y_true: (string) ground truth image mask
        y_pred : (int) predicted image mask
        smooth : (float) smoothening to prevent divison by 0
        threshold : (float) threshold over which pixel is conidered positive

    Returns:
        Calculated Dice coeffecient for evaluation metric
    """
    y_true_f = K.flatten(y_true)
    y_true_f = K.cast(y_true_f, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), threshold), 'float32')
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth)/ (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score
#

def conv_block(input_tensor, num_filters):
	encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
	encoder = layers.Activation('relu')(encoder)
	encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
	encoder = layers.Activation('relu')(encoder)
	return encoder

def encoder_block(input_tensor, num_filters):
	encoder = conv_block(input_tensor, num_filters)
	encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
	return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
	decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
	decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
	decoder = layers.Activation('relu')(decoder)
	decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
	decoder = layers.BatchNormalization()(decoder)
	decoder = layers.Activation('relu')(decoder)
	decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
	decoder = layers.Activation('relu')(decoder)
	return decoder

def get_model():
	inputs = layers.Input(shape=(256, 256, n_channels), name="MSI")# 256
	encoder0_pool, encoder0 = encoder_block(inputs, 32) # 128
	encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64) # 64
	encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128) # 32
	encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256) # 16
	encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512) # 8
	center = conv_block(encoder4_pool, 1024) # center
	decoder4 = decoder_block(center, encoder4, 512) # 16
	decoder3 = decoder_block(decoder4, encoder3, 256) # 32
	decoder2 = decoder_block(decoder3, encoder2, 128) # 64
	decoder1 = decoder_block(decoder2, encoder1, 64) # 128
	decoder0 = decoder_block(decoder1, encoder0, 32) # 256
	outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

	model = models.Model(inputs=[inputs], outputs=[outputs])

	# model.compile(
	# 	optimizer=optimizers.get(OPTIMIZER),
	# 	loss=losses.get(LOSS),
	# 	metrics=[metrics.get(metric) for metric in METRICS])
	# model.compile(
	# 	optimizer=tf.keras.optimizers.Adam(
	# 		learning_rate=1e-3
	# 	),  # this LR is overriden by base cycle LR if CyclicLR callback used
	# 	loss=dice_coef_loss,
	# 	metrics=dice_score,
	# )
	return model






def train_test_model(hparams):
    model = get_model()
    model.compile(
      optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-2),  # this LR is overriden by base cycle LR if CyclicLR callback used
      # loss=dice_coef_loss,
      # metrics=dice_score,
      # loss="binary_crossentropy",
      # metrics=metrics
      loss=jaccard_coef_loss,
      metrics=['binary_crossentropy', jaccard_coef_int]
    )

    history= model.fit(train_generator,
                    steps_per_epoch=250 // 3, shuffle=True,
                    epochs=30,
                    verbose=1,
                    validation_data=test_generator,
                    callbacks=[reduce_lr,
        tf.keras.callbacks.TensorBoard(logdir),  # log metrics
        hp.KerasCallback(logdir, hparams),  # log hparams
      ],
    )



    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('history.csv')
    # ### Save model
    export_path = tf.saved_model.save(m, 'keras_export')
    print("Model exported to: ", export_path)


    # Removing the first value of the loss
    losses = history.history['loss']
    val_losses = history.history['val_loss']

    # Looking at the loss curve
    plt.plot(losses)
    plt.plot(val_losses)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # Log as MLflow artifact
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir, "loss_curve.png")
        plt.savefig(image_path)
        mlflow.log_artifact(image_path)

    def plot_metric(history, metric):
        train_metrics = history.history[metric]
        val_metrics = history.history['val_' + metric]
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics, 'bo--')
        plt.plot(epochs, val_metrics, 'ro-')
        plt.title('Training and validation ' + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_" + metric, 'val_' + metric])
        plt.show()