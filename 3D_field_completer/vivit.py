#!/usr/bin/env python
# coding: utf-8

# # Video Vision Transformer
# 
# **Author:** [Aritra Roy Gosthipaty](https://twitter.com/ariG23498), [Ayush Thakur](https://twitter.com/ayushthakur0) (equal contribution)<br>
# **Description:** A Transformer-based architecture for video classification.
# **Modified:** [Zhenze Yang]

# ## Introduction
# 
# Videos are sequences of images. Let's assume you have an image
# representation model (CNN, ViT, etc.) and a sequence model
# (RNN, LSTM, etc.) at hand. We ask you to tweak the model for video
# classification. The simplest approach would be to apply the image
# model to individual frames, use the sequence model to learn
# sequences of image features, then apply a classification head on
# the learned sequence representation.
# The Keras example
# [Video Classification with a CNN-RNN Architecture](https://keras.io/examples/vision/video_classification/)
# explains this approach in detail. Alernatively, you can also
# build a hybrid Transformer-based model for video classification as shown in the Keras example
# [Video Classification with Transformers](https://keras.io/examples/vision/video_transformers/).
# 
# In this example, we minimally implement
# [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691)
# by Arnab et al., a **pure Transformer-based** model
# for video classification. The authors propose a novel embedding scheme
# and a number of Transformer variants to model video clips. We implement
# the embedding scheme and one of the variants of the Transformer
# architecture, for simplicity.
# 
# This example requires TensorFlow 2.6 or higher, and the `medmnist`
# package, which can be installed by running the code cell below.

# In[ ]:


#get_ipython().system('pip install -qq medmnist')


# ## Imports

# In[33]:


import os
import io
import imageio
# import medmnist
import ipywidgets
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setting seed for reproducibility
SEED = 42
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
keras.utils.set_random_seed(SEED)


# ## Hyperparameters
# 
# The hyperparameters are chosen via hyperparameter
# search. You can learn more about the process in the "conclusion" section.

# In[164]:


# DATA
DATASET_NAME = "S113d"
BATCH_SIZE = 64
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (8, 32, 32, 3)
OUTPUT_SHAPE = (8, 32, 32, 3)

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 100

# TUBELET EMBEDDING
PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 8


# ## Dataset
# 
# For our example we use the
# [MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification](https://medmnist.com/)
# dataset. The videos are lightweight and easy to train on.

# In[141]:



def download_and_prepare_dataset(data_path):
    """Utility function to download the dataset.

    Arguments:
        data_info (dict): Dataset metadata.
    """
    

    data = np.load(data_path) #(Num_data, num_frames, img_size, img_size, channel)
    # Get videos
    indexes = np.arange(data.shape[0])
    #np.random.shuffle(indexes)
    train_index = indexes[: int(0.7 * data.shape[0])]
    val_index = indexes[int(0.7 * data.shape[0]) : int(0.9 * data.shape[0])]
    test_index = indexes[int(0.9 * data.shape[0]) :]
    train_videos = data[train_index]
    valid_videos = data[val_index]
    test_videos = data[test_index]

    return (
        (train_videos[:, :INPUT_SHAPE[0], :, :, :], train_videos[:, -OUTPUT_SHAPE[0]:, :, :, :] / 255),
        (valid_videos[:, :INPUT_SHAPE[0], :, :, :], valid_videos[:, -OUTPUT_SHAPE[0]:, :, :, :] / 255),
        (test_videos[:, :INPUT_SHAPE[0], :, :, :], test_videos[:, -OUTPUT_SHAPE[0]:, :, :, :] / 255),
    )



# Get the dataset
data_path = "./datasets/S11.npy"
prepared_dataset = download_and_prepare_dataset(data_path)
(train_videos, train_labels) = prepared_dataset[0]
(valid_videos, valid_labels) = prepared_dataset[1]
(test_videos, test_labels) = prepared_dataset[2]
print(train_videos.shape, train_labels.shape)


# ### `tf.data` pipeline

# In[142]:



@tf.function
def preprocess(frames: tf.Tensor, label: tf.Tensor):
    """Preprocess the frames tensors and parse the labels."""
    # Preprocess images
    frames = tf.image.convert_image_dtype(
        frames[
            ..., tf.newaxis
        ],  # The new axis is to help for further processing with Conv3D layers
        tf.float32,
    )
    # Parse label
    label = tf.cast(label, tf.float32)
    return frames, label


def prepare_dataloader(
    videos: np.ndarray,
    labels: np.ndarray,
    loader_type: str = "train",
    batch_size: int = BATCH_SIZE,
):
    """Utility function to prepare the dataloader."""
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

    if loader_type == "train":
        dataset = dataset.shuffle(BATCH_SIZE * 2)

    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataloader


trainloader = prepare_dataloader(train_videos, train_labels, "train")
validloader = prepare_dataloader(valid_videos, valid_labels, "valid")
testloader = prepare_dataloader(test_videos, test_labels, "test")


# ## Tubelet Embedding
# 
# In ViTs, an image is divided into patches, which are then spatially
# flattened, a process known as tokenization. For a video, one can
# repeat this process for individual frames. **Uniform frame sampling**
# as suggested by the authors is a tokenization scheme in which we
# sample frames from the video clip and perform simple ViT tokenization.
# 
# | ![uniform frame sampling](https://i.imgur.com/aaPyLPX.png) |
# | :--: |
# | Uniform Frame Sampling [Source](https://arxiv.org/abs/2103.15691) |
# 
# **Tubelet Embedding** is different in terms of capturing temporal
# information from the video.
# First, we extract volumes from the video -- these volumes contain
# patches of the frame and the temporal information as well. The volumes
# are then flattened to build video tokens.
# 
# | ![tubelet embedding](https://i.imgur.com/9G7QTfV.png) |
# | :--: |
# | Tubelet Embedding [Source](https://arxiv.org/abs/2103.15691) |

# In[181]:



class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        print(videos.shape)
        projected_patches = self.projection(videos)
        print(projected_patches.shape)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

class Decoder(layers.Layer):
    def __init__(self, input_shape, output_shape, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.depth = int(input_shape[0] / patch_size[0])
        self.height = int(input_shape[1] / patch_size[1])
        self.width = int(input_shape[2] / patch_size[2])
        print(self.depth, self.height, self.width)
        self.wrap = layers.Reshape(target_shape=(self.depth, self.height, self.width, embed_dim))
        self.decode = layers.Conv3DTranspose(
            filters=64,
            kernel_size=patch_size,
            strides=patch_size,
            #activation="sigmoid",
            data_format='channels_last',
            padding="VALID",
        )
        self.conv3d_1 = layers.Conv3D(
            filters=16,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding="same",
        )
        self.conv3d_2 = layers.Conv3D(
            filters=input_shape[-1],
            kernel_size=(1, 1, 1),
            activation="sigmoid",
            strides=(1, 1, 1),
            padding="same",
        )
        

    def call(self, flattened_patches):
        projected_patches = self.wrap(flattened_patches)
        decoded_videos = self.decode(projected_patches)
        decoded_videos = self.conv3d_1(decoded_videos)
        decoded_videos = self.conv3d_2(decoded_videos)
        return decoded_videos


# ## Positional Embedding
# 
# This layer adds positional information to the encoded video tokens.

# In[182]:



class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens
    

    


# ## Video Vision Transformer
# 
# The authors suggest 4 variants of Vision Transformer:
# 
# - Spatio-temporal attention
# - Factorized encoder
# - Factorized self-attention
# - Factorized dot-product attention
# 
# In this example, we will implement the **Spatio-temporal attention**
# model for simplicity. The following code snippet is heavily inspired from
# [Image classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer/).
# One can also refer to the
# [official repository of ViViT](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit)
# which contains all the variants, implemented in JAX.

# In[183]:



def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    decoder,
    input_shape=INPUT_SHAPE,
    output_shape=OUTPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    num_patches=NUM_PATCHES,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    #print(inputs.shape[0])
    patches = tubelet_embedder(inputs)
    # Encode patches.
    #print(patches.shape)
    encoded_patches = positional_encoder(patches)
    #print(encoded_patches.shape)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])
    decoded_videos = decoder(encoded_patches)


    # Classify outputs.
    outputs = decoded_videos

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# ## Train

# In[ ]:



def run_experiment():
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
    checkpoint_path = "ckpts/my_model"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False,
                                                 save_freq=125, 
                                                 verbose=1)
    # Initialize model
    model = create_vivit_classifier(
        tubelet_embedder=TubeletEmbedding(
            embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
        ),
        positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
        decoder=Decoder(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE)
    )

    # Compile the model with the optimizer, loss function
    # and the metrics.
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.binary_crossentropy,
        #metrics=[
        #    keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        #    keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        #],
    )

    # Train the model.
    _ = model.fit(trainloader, epochs=EPOCHS, verbose=2, validation_data=validloader, callbacks=[early_stopping, reduce_lr, cp_callback])

#     _, accuracy, top_5_accuracy = model.evaluate(testloader)
#     print(f"Test accuracy: {round(accuracy * 100, 2)}%")
#     print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return model


model = run_experiment()


# In[153]:



testsamples, labels = next(iter(testloader))
testsamples, labels = testsamples[:1], labels[:1]
print(testsamples.shape, labels.shape)
output = model.predict(testsamples)[0]
fig, axs = plt.subplots(3, 8, figsize=(16, 6))
print(np.squeeze(testsamples)[0].shape)
for i in range(8):
    axs[0, i].imshow(np.squeeze(testsamples)[i])
    axs[0, i].set_title(f"Frame {i + 1}")
    axs[0, i].axis("off")

for i in range(8):
    axs[1, i].imshow(np.squeeze(labels)[i])
    axs[1, i].axis("off")
for i in range(8):

    axs[2, i].imshow(np.squeeze(output)[i] )
    axs[2, i].axis("off")

plt.savefig("example_results.png", dpi=300)
plt.show()
    



# ## Final thoughts
# 
# With a vanilla implementation, we achieve ~79-80% Top-1 accuracy on the
# test dataset.
# 
# The hyperparameters used in this tutorial were finalized by running a
# hyperparameter search using
# [W&B Sweeps](https://docs.wandb.ai/guides/sweeps).
# You can find out our sweeps result
# [here](https://wandb.ai/minimal-implementations/vivit/sweeps/66fp0lhz)
# and our quick analysis of the results
# [here](https://wandb.ai/minimal-implementations/vivit/reports/Hyperparameter-Tuning-Analysis--VmlldzoxNDEwNzcx).
# 
# For further improvement, you could look into the following:
# 
# - Using data augmentation for videos.
# - Using a better regularization scheme for training.
# - Apply different variants of the transformer model as in the paper.
# 
# We would like to thank [Anurag Arnab](https://anuragarnab.github.io/)
# (first author of ViViT) for helpful discussion. We are grateful to
# [Weights and Biases](https://wandb.ai/site) program for helping with
# GPU credits.
# 
# You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/video-vision-transformer) and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/video-vision-transformer-CT).
