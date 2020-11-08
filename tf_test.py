import tensorflow as tf
print(tf.__version__)
tf.config.experimental.list_physical_devices('GPU')
# Set up GPU:
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

print(gpu_devices)
print(str(len(gpu_devices)) + " GPU(s) available" if len(gpu_devices) > 0 else "Warning: no GPU available.")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

tf.test.is_built_with_cuda()



