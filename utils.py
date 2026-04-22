# IMPORTS 
import numpy as np
import hashlib
import os
import cv2 
import matplotlib.pyplot as plt
import seaborn as sns
from keras import ops
from keras.models import Sequential 
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import kerastuner as kt
import tensorflow as tf
from tensorflow.keras.metrics import AUC
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from sklearn.metrics import confusion_matrix, classification_report



# IMAGE PREPROCESSING 

def load_images_and_labels(metadata, method=None, image_size=(50, 50)):
    '''
    Import and preprocess the images and their lables into two arrays.

    Args:
        metadata (pandas.DataFrame): Data about the images, containing the path directories.
        method (str): Color normalization or transformation to be applied to the images.
        image_size (tuple): Target size to resize all images.

    Returns:
        images (np.array): Array of preprocessed images.
        binary_labels (np.array): Array of binary labels corresponding to the images.
        multiclass_labels (np.array): Array of multi-class labels corresponding to the images.
    '''
    images = []
    binary_labels = []
    multiclass_labels = []
    for _, row in metadata.iterrows():
        image = cv2.imread(row['path_to_image'])
        # resizing
        image = cv2.resize(image, image_size)

        if method == 'gray scaling':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if method == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if method == 'laplacian over rgb':
            image = cv2.Laplacian(image, cv2.CV_8UC3)
        if method == 'laplacian over gray':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image =  cv2.Laplacian(gray, cv2.CV_8UC3)
        if method == 'contrast adjustment':
            image = cv2.convertScaleAbs(image, alpha=1.2, beta=20)

        # rescaling
        image = image / 255.0

        images.append(image)
        binary_labels.append(row['binary_class'])
        multiclass_labels.append(row['multi_class'])

    return np.array(images), np.array(binary_labels), np.array(multiclass_labels)


def hash_image(image):
    '''
    Compute a hash for a NumPy image array.
    
    Args:
        image (np.array): NumPy array representing the image.

    Returns:
        str: A hexadecimal SHA-256 hash of the image's byte representation.
    '''
    return hashlib.sha256(image.tobytes()).hexdigest()


def get_duplicated_indixes(train, test):
    '''
    Identify duplicate images between two datasets based on their hashes.

    Args:
        train (np.array): Array of training images.
        test (np.array): Array of testing images.

    Returns:
        duplicates_train (list): List of indices of duplicate images in the training dataset.
        duplicates_test (list): List of indices of duplicate images in the testing dataset.
    '''
# Create sets of hashes for train and test datasets
    train_hashes = {hash_image(img) for img in train}
    test_hashes = {hash_image(img) for img in test}

    # Find duplicates
    duplicates = train_hashes.intersection(test_hashes)

    duplicates_train = [i for i, img in enumerate(train) if hash_image(img) in duplicates]
    duplicates_test = [i for i, img in enumerate(test) if hash_image(img) in duplicates]

    return duplicates_train, duplicates_test   


def plot_duplicates(train, test, k=6):
    '''
    Plot duplicate images from training and testing datasets.

    This function identifies duplicate images between the train and test datasets 
    and displays a side-by-side comparison of the duplicate pairs. The first `k` duplicate pairs are plotted.

    Args:
        train (np.array): Array of training images.
        test (np.array): Array of testing images.
        k (int): Number of duplicate pairs to display (default is 6).
    '''
    train_duplicates, test_duplicates = get_duplicated_indixes(train, test)
    print(f"Number of duplicate images: {len(train_duplicates)}")

    duplicate_pairs= []
    for i in train_duplicates:
        for j in test_duplicates:
            if hash_image(train[i]) == hash_image(test[j]):
                duplicate_pairs.append([i,j])

    plt.figure(figsize=(12, 6))

    for i, [itrain, itest] in enumerate(duplicate_pairs[:k]):
        plt.subplot(2, k, i + 1)
        plt.imshow(train[itrain]) 
        plt.title("train image")
        plt.axis("off")
        
        plt.subplot(2, k, i + 1 + k)
        plt.imshow(test[itest]) 
        plt.title("test image")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def drop_duplicates(X_train, y_train, w_train, X_test, y_test, w_test):
    '''
    Remove duplicate images from the training and testing datasets.

    Args:
        X_train (np.array): Array of training images.
        y_train (np.array): Array of labels for the training images.
        w_train (np.array): Array of sample weights for the training images.
        X_test (np.array): Array of testing images.
        y_test (np.array): Array of labels for the testing images.
        w_test (np.array): Array of sample weights for the testing images.

    Returns:
        X_train (np.array): Updated training images after removing duplicates.
        y_train (np.array): Updated labels for the training images.
        w_train (np.array): Updated sample weights for the training images.
        X_test (np.array): Updated testing images after removing duplicates.
        y_test (np.array): Updated labels for the testing images.
        w_test (np.array): Updated sample weights for the testing images.
    '''
    train_duplicates, test_duplicates = get_duplicated_indixes(X_train, X_test)

    X_train = np.delete(X_train, train_duplicates, axis=0)
    y_train = np.delete(y_train, train_duplicates, axis=0)
    w_train = np.delete(w_train, train_duplicates, axis=0)

    X_test = np.delete(X_test, test_duplicates, axis=0)
    y_test = np.delete(y_test, test_duplicates, axis=0)
    w_test = np.delete(w_test, test_duplicates, axis=0)

    print(f"Updated train images shape: {X_train.shape}")
    print(f"Updated test images shape: {X_test.shape}")

    return X_train, y_train, w_train, X_test, y_test, w_test


def data_augmentation(train_data, train_labels):
    '''
    Applies data augmentation to the training data and generates augmented data for training and validation.

    Args:
        train_data (np.array): Array of input images to be augmented.
        train_labels (np.array): Array of labels corresponding to the input images.

    Returns:
        train (tensorflow.keras.preprocessing.image.NumpyArrayIterator): Augmented training data generator.
        val (tensorflow.keras.preprocessing.image.NumpyArrayIterator): Augmented validation data generator.
    '''
    train_datagen = ImageDataGenerator(rotation_range=10, 
                                       zoom_range = 0.2,
                                       width_shift_range=0.2,  
                                       height_shift_range=0.2,  
                                       horizontal_flip=True,  
                                       vertical_flip=True,
                                       fill_mode='nearest',
                                       validation_split=0.2)
    
    train_datagen.fit(train_data)

    train = train_datagen.flow(train_data, train_labels, batch_size=32, subset="training")
    val = train_datagen.flow(train_data, train_labels, batch_size=8, subset="validation")

    return train, val



# VISUALIZATIONS

# Function to display images with labels
def plot_images(images, labels, class_names, num_images=20):
    '''
    Plots a grid of images with their corresponding labels.

    Args:
        images (np.array): Array of image data to plot.
        labels (np.array): Array of labels corresponding to the images.
        class_names (list): List of class names where the index corresponds to the class label.
        num_images (int): Number of images to display (default is 20).
    '''
    plt.figure(figsize=(20, 15))
    for i in range(num_images):
        plt.subplot(5, 4, i + 1)
        plt.imshow(images[i]) 
        plt.axis('off')  
        label_index = labels[i]  
        plt.title(f"Label: {class_names[label_index]}")
    plt.show()


def plot_images_transformed(sample_images, method):
    '''
    Plots a comparison of original images and their transformed versions with the defined method.

    Args:
        sample_images (list): List of file paths to the images to be processed and displayed.
        method (str): Transformation method to apply to the images. 
    '''
    plt.figure(figsize=(12, 6))

    for i, image_path in enumerate(sample_images):
        input_image = cv2.imread(image_path)
        
        if method == 'gray scaling':
            gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        if method == 'rgb':
            rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        ax = plt.subplot(2, len(sample_images), i + 1)
        plt.imshow(input_image)
        plt.axis("off")
        plt.title("Original")
        
        ax = plt.subplot(2, len(sample_images), i + 1 + len(sample_images))
        if method == 'gray scaling':
            plt.imshow(gray_image, cmap='gray')
            plt.axis("off")
            plt.title("Grayscale")
        if method == 'rgb':
            plt.imshow(rgb_image)
            plt.axis("off")
            plt.title("RGB")

    plt.tight_layout()
    plt.show()


def plot_image_laplacian_transformation(sample_image):
    '''
    Plots a comparison of original images and their versions with laplacian transformation.

    Args:
        sample_images (list): File paths to the image to be processed and displayed.
    '''
    input_image = cv2.imread(sample_image)

    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    laplacian = cv2.Laplacian(input_image, cv2.CV_8UC3)
    laplacian_gray =   cv2.Laplacian(gray, cv2.CV_8UC3)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(input_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(laplacian)
    plt.title("Adjusted Image OVER THE RGB ")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(laplacian_gray)
    plt.title("Adjusted Image OVER THE GRAY SCALE ")
    plt.axis("off")


def plot_image_contrast_adjustment(sample_image, alpha=1.2, beta=20):
    '''
    Adjust the contrast and brightness of an image and display a comparison of the original and adjusted images.

    Args:
        sample_image (str): File path to the input image.
        alpha (float): Contrast adjustment factor (default is 1.2). 
                       Values > 1 increase contrast, values < 1 decrease it.
        beta (int): Brightness adjustment factor (default is 20). 
                    Positive values make the image brighter, negative values make it darker.
    '''
    input_image = cv2.imread(sample_image)

    adjusted_image = cv2.convertScaleAbs(input_image, alpha=alpha, beta=beta)

    # Display the original and adjusted images
    plt.figure(figsize=(8, 4))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(input_image) 
    plt.title("Original Image")
    plt.axis("off")

    # Adjusted image
    plt.subplot(1, 2, 2)
    plt.imshow(adjusted_image) 
    plt.title(f"Adjusted Image (alpha={alpha}, beta={beta})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



# BINARY CLASSIFICATION - HYPERTUNING 

def build_model_from_scratch(hp):
    '''
    Builds a CNN model from scratch and sets up hyperparameter space to search.

    Args:
        hp (HyperParameter): Configures hyperparameters to tune.

    Returns:
        model (keras model): Compiled model with hyperparameters to tune.
    '''
    model = Sequential()

    # first convolutional layer
    model.add(Conv2D(filters=hp.Choice("filters_block1", [16, 32, 64]),
              kernel_size=hp.Choice("kernel_size_block1", [3, 5]),
              activation='relu',
              input_shape=(50,50, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # additional convolutional layers
    for i in range(hp.Int("num_conv_blocks", 0, 3)):  
        model.add(Conv2D(filters=hp.Int(f"filters_block{i+2}", min_value=32, max_value=128, step=32),
                  kernel_size=hp.Choice(f"kernel_size_block{i+2}", [3, 5]),
                  activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # flatten layer
    model.add(Flatten())

    # dense layers
    for i in range(hp.Int("num_dense_layers", 1, 5)):  
        model.add(Dense(units=hp.Int(f"dense_units_{i+1}", min_value=32, max_value=256, step=32),
                  activation='relu'))
        model.add(Dropout(hp.Float(f"dropout_dense_{i+1}", min_value=0.2, max_value=0.6, step=0.1)))

    # output layer
    model.add(Dense(1, activation='sigmoid'))

    # learning rate
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    # compile model
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='binary_crossentropy', metrics=[AUC(name='auc', curve='PR')])

    return model


def build_model_from_scratch_gray_scaling(hp):
    '''
    Builds a CNN model from scratch and sets up hyperparameter space to search.

    Args:
        hp (HyperParameter): Configures hyperparameters to tune.

    Returns:
        model (keras model): Compiled model with hyperparameters to tune.
    '''
    model = Sequential()

    # first convolutional layer
    model.add(Conv2D(filters=hp.Choice("filters_block1", [16, 32, 64]),
              kernel_size=hp.Choice("kernel_size_block1", [3, 5]),
              activation='relu',
              input_shape=(50,50, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # additional convolutional layers
    for i in range(hp.Int("num_conv_blocks", 1, 3)):  
        model.add(Conv2D(filters=hp.Int(f"filters_block{i+2}", min_value=32, max_value=128, step=32),
                  kernel_size=hp.Choice(f"kernel_size_block{i+2}", [3, 5]),
                  activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # flatten layer
    model.add(Flatten())

    # dense layers
    for i in range(hp.Int("num_dense_layers", 1, 3)):  
        model.add(Dense(units=hp.Int(f"dense_units_{i+1}", min_value=32, max_value=256, step=32),
                  activation='relu'))
        model.add(Dropout(hp.Float(f"dropout_dense_{i+1}", min_value=0.2, max_value=0.6, step=0.1)))

    # output layer
    model.add(Dense(1, activation='sigmoid'))

    # learning rate
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    # compile model
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='binary_crossentropy', metrics=[AUC(name='auc', curve='PR')])

    return model



def build_on_base_model(hp, base_model):
    '''
    Builds model using a pre-trained base model and sets up hyperparameter space to search.

    Args:
        hp (HyperParameter): Configures hyperparameters to tune.
        base_model (keras model): Pre-trained model to use as the base.

    Returns:
        model (keras model): Compiled model with hyperparameters to tune.
    '''
    # initializa model and add base model
    model = Sequential()
    model.add(base_model)  
    model.add(Flatten())

    # dense layers
    for i in range(1, hp.Int("num_layers", 2, 4)): 
        model.add(Dense(units=hp.Int("units_" + str(i), min_value=64, max_value=768, step=64),
                  activation="relu") )
        model.add(Dropout(hp.Float("dropout_" + str(i), 0.2, 0.6, step=0.1)))

    # output layer
    model.add(Dense(1, activation='sigmoid'))

    # learning rate
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    # compile model
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='binary_crossentropy', metrics=[AUC(name='auc', curve='PR')])

    return model


def run_tuner(model_builder, project_name, train_data, val_data, callbacks):
    '''
    Runs a Keras Tuner search to find the best hyperparameters for a model.

    Args:
        model_builder (function): A function that builds the model to be tuned.
        project_name (str): The name of the project for logging and saving tuner results.
        train_data (generator): Data generator for training data.
        val_data (generator): Data generator for validation data.
        callbacks (list): List of callbacks to use during training.

    Returns:
        best_model (keras model): The model built using the best hyperparameters found by the tuner.
        best_hp (keras_tuner.HyperParameters): The best hyperparameters found by the tuner.
    '''
    tuner = kt.Hyperband(model_builder,
                         objective='val_auc',
                         max_epochs=20,
                         factor=3,
                         directory='log',
                         project_name=project_name)
    
    tuner.search(train_data, validation_data=val_data, 
                 epochs=20, callbacks=[callbacks])

    best_hp = tuner.get_best_hyperparameters()[0]
    best_model = tuner.hypermodel.build(best_hp)

    return best_model, best_hp



# MULTICLASS CLASSIFICATION - HYPERTUNING 

def build_multi_model_from_scratch(hp):
    '''
    Builds a CNN model from scratch and sets up hyperparameter space to search.

    Args:
        hp (HyperParameter): Configures hyperparameters to tune.

    Returns:
        model (keras model): Compiled model with hyperparameters to tune.
    '''
    model = Sequential()

    # first convolutional layer
    model.add(Conv2D(filters=hp.Choice("filters_block1", [16, 32, 64]),
                    kernel_size=hp.Choice("kernel_size_block1", [3,5]),
                     padding="same",
                    activation='relu',
                    input_shape=(50,50, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # additional convolutional layers
    for i in range(hp.Int("num_conv_blocks", 1, 4)):  
        model.add(Conv2D(filters=hp.Choice(f"filters_block{i+2}", [64, 92, 128]),
                  kernel_size=hp.Choice(f"kernel_size_block{i+2}", [3,5]),
                  activation='relu',
                  padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # flatten layer
    model.add(Flatten())

    # dense layers
    for i in range(hp.Int("num_dense_layers", 1, 3)):  
        model.add(Dense(units=hp.Int(f"dense_units_{i+1}", min_value=32, max_value=256, step=32),
                  activation='relu'))
        model.add(Dropout(hp.Float(f"dropout_dense_{i+1}", min_value=0.2, max_value=0.6, step=0.1)))

    # output layer
    model.add(Dense(8, activation="softmax"))

    # learning rate
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    # compile model
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])   

    return model


def build_on_base_model_multi(hp, base_model):
    '''
    Builds model using a pre-trained base model and sets up hyperparameter space to search.

    Args:
        hp (HyperParameter): Configures hyperparameters to tune.
        base_model (keras model): Pre-trained model to use as the base.

    Returns:
        model (keras model): Compiled model with hyperparameters to tune.
    '''
    # initializa model and add base model
    model = Sequential()
    model.add(base_model)  
    model.add(Flatten())

    # dense layers
    for i in range(1, hp.Int("num_layers", 2, 5)): 
        model.add(Dense(units=hp.Int("units_" + str(i), min_value=64, max_value=512, step=64),
                        activation="relu") )
        model.add(Dropout(hp.Float("dropout_" + str(i), 0.2, 0.6, step=0.1)))

    # output layer
    model.add(Dense(8, activation="softmax"))

    # learning rate
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    # compile model
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy']) 

    return model


def build_functional_api(hp):
    '''
    Builds a multi-input CNN model and sets up hyperparameter space to search.

    Args:
        hp (HyperParameter): Configures hyperparameters to tune.

    Returns:
        model (keras Model): Compiled model with hyperparameters to tune.
    '''
    # Image input
    image_input = Input(shape=(50, 50, 3), name='image_input')
    
    # Initial convolutional block for image input
    x = Conv2D(filters=hp.Choice("filters_block1", [16, 32]),
               kernel_size=hp.Choice("kernel_size_block1", [3]),
               padding="same",
               activation='relu')(image_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Additional convolutional blocks (hyperparameterized)
    for i in range(hp.Int("num_conv_blocks", 1, 4)):  
        x = Conv2D(filters=hp.Choice(f"filters_block{i+2}", [32, 64, 128]),
                   kernel_size=hp.Choice(f"kernel_size_block{i+2}", [3]),
                   padding="same",
                   activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Flatten image features
    x = Flatten()(x)

    # Binary label input
    binary_input = Input(shape=(1,), name='binary_input')
    y = Dense(units=hp.Int("binary_dense_units", min_value=8, max_value=64, step=8), activation='relu')(binary_input)

    # Combine image features and binary input
    combined = Concatenate()([x, y])

    # Fully connected layers (hyperparameterized)
    for i in range(hp.Int("num_dense_layers", 1, 3)):  
        combined = Dense(units=hp.Int(f"dense_units_{i+1}", min_value=32, max_value=256, step=32),
                         activation='relu')(combined)
        combined = Dropout(rate=hp.Float(f"dropout_dense_{i+1}", min_value=0.2, max_value=0.6, step=0.1))(combined)

    # Output layer
    output = Dense(8, activation="softmax", name="output")(combined)

    # Create the model
    model = Model(inputs=[image_input, binary_input], outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model


def run_tuner_multi(model_builder, project_name, train_data, val_data, callbacks):
    '''
    Runs a Keras Tuner search to find the best hyperparameters for a model.

    Args:
        model_builder (function): A function that builds the model to be tuned.
        project_name (str): The name of the project for logging and saving tuner results.
        train_data (generator): Data generator for training data.
        val_data (generator): Data generator for validation data.
        callbacks (list): List of callbacks to use during training.

    Returns:
        best_model (keras model): The model built using the best hyperparameters found by the tuner.
        best_hp (keras_tuner.HyperParameters): The best hyperparameters found by the tuner.
    '''
    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=20,
                         factor=3,
                         directory='log',
                         project_name=project_name)
    
    tuner.search(train_data, validation_data=val_data, epochs=20, callbacks=[callbacks])

    best_hp = tuner.get_best_hyperparameters()[0]
    best_model = tuner.hypermodel.build(best_hp)

    return best_model, best_hp


def run_tuner_api(model_builder, project_name, train_data, train_labels, callbacks):
    '''
    Runs a Keras Tuner search to find the best hyperparameters for a model.

    Args:
        model_builder (function): A function that builds the model to be tuned.
        project_name (str): The name of the project for logging and saving tuner results.
        train_data (generator): Data generator for training data.
        val_data (generator): Data generator for validation data.
        callbacks (list): List of callbacks to use during training.

    Returns:
        best_model (keras model): The model built using the best hyperparameters found by the tuner.
        best_hp (keras_tuner.HyperParameters): The best hyperparameters found by the tuner.
    '''
    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=20,
                         factor=3,
                         directory='log',
                         project_name=project_name)
    
    tuner.search(train_data, train_labels, validation_split=0.2,epochs=20, callbacks=[callbacks])

    best_hp = tuner.get_best_hyperparameters()[0]
    best_model = tuner.hypermodel.build(best_hp)

    return best_model, best_hp



# MODEL EVALUATION 

def plot_val_scores(histories, model_names, binary=True):
    '''
    Plot the final training and validation metrics (AUC or accuracy) and losses for multiple models.

    Args:
        histories (list): List of Keras `History` objects containing training history for each model.
        model_names (list): List of names corresponding to the models in `histories`.
        binary (bool): If True, plots AUC metrics for binary classification; if False, plots accuracy for multiclass classification.
    '''
    # Extract metrics
    if binary:
        final_scores = [history.history['auc'][-1] for history in histories]
        final_val_scores = [history.history['val_auc'][-1] for history in histories]
        title = 'AUC'
    else:
        final_scores = [history.history['accuracy'][-1] for history in histories]
        final_val_scores = [history.history['val_accuracy'][-1] for history in histories]
        title = 'Accuracy'

    final_losses = [history.history['loss'][-1] for history in histories]
    final_val_losses = [history.history['val_loss'][-1] for history in histories]

    # Plot the metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharey=False)

    # Plot F1 Scores
    axes[0,0].bar(model_names, final_scores, color='lightpink', edgecolor='black')
    axes[0,0].set_title('Final '+title, fontsize=14)
    axes[0,0].set_ylabel(title, fontsize=12)
    axes[0,0].set_xticklabels(model_names, rotation=80, fontsize=10)
    axes[0,0].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot Validation F1 Scores
    axes[0,1].bar(model_names, final_val_scores, color='skyblue', edgecolor='black')
    axes[0,1].set_title('Final Validation '+title, fontsize=14)
    axes[0,1].set_xticklabels(model_names, rotation=80, fontsize=10)
    axes[0,1].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot Losses
    axes[1,0].bar(model_names, final_losses, color='lightgreen', edgecolor='black')
    axes[1,0].set_title('Final Losses', fontsize=14)
    axes[1,0].set_ylabel('Loss', fontsize=12)
    axes[1,0].set_xticklabels(model_names, rotation=80, fontsize=10)
    axes[1,0].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot Validation Losses
    axes[1,1].bar(model_names, final_val_losses, color='lightcoral', edgecolor='black')
    axes[1,1].set_title('Final Validation Losses', fontsize=14)
    axes[1,1].set_xticklabels(model_names, rotation=80, fontsize=10)
    axes[1,1].grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.show()


def plot_test_scores(models, model_names, binary=True):
    '''
    Plot the F1 scores, precision, and recall for multiple models based on their test set performance.

    Args:
        models (dict): Dictionary where keys are model names and values are tuples containing 
                       test images (np.array) and test labels (np.array).
        model_names (list): List of names corresponding to the models in `models`.
        binary (bool): If True, evaluates metrics for binary classification; if False, evaluates metrics for multiclass classification.
    '''
    precisions = []
    recalls = []
    f1_scores = []

    for model, (test_images, test_labels) in models.items(): 
        predictions = model.predict(test_images)
        if binary:
            predictions = (predictions > 0.5).astype(int)
        else:
            predictions = np.argmax(predictions,axis=1)

        report = classification_report(test_labels, predictions, output_dict=True)

        weighted_avg = report['weighted avg']
        precisions.append(weighted_avg['precision'])
        recalls.append(weighted_avg['recall'])
        f1_scores.append(weighted_avg['f1-score'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Plot F1 Scores
    axes[0].bar(model_names, f1_scores, color='lightpink', edgecolor='black')
    axes[0].set_title('Test F1 Scores')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=80)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot Recalls
    axes[1].bar(model_names, recalls, color='lightgreen', edgecolor='black')
    axes[1].set_title('Test Recalls')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=80)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot Precisions
    axes[2].bar(model_names, precisions, color='skyblue', edgecolor='black')
    axes[2].set_title('Test Precisions')
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis='x', rotation=80)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.show()


def plot_history(history, binary=True):
    '''
    Plot the training and validation metrics (loss and AUC/accuracy) over epochs.

    Args:
        history (keras.callbacks.History): The history object returned by the `fit` method of a Keras model, 
                                           containing training and validation metrics for each epoch.
        binary (bool): If True, plots AUC for binary classification; if False, plots accuracy for multiclass classification.
    '''
    # training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])

    if binary:
        # training and validation auc
        plt.subplot(1, 2, 2)
        plt.plot(history.history['auc'])
        plt.plot(history.history['val_auc'])
        plt.title('Model AUC')
        plt.ylabel('AUC')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'])
        plt.show()
    else:
        # training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'])
        plt.show()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, test_images, test_labels, class_names, binary=True):
    '''
    Generate and plot a confusion matrix for a model's predictions on test data.

    Args:
        model (keras.Model): The trained model to evaluate.
        test_images (np.array): Array of test images used as input to the model.
        test_labels (np.array): Array of true labels corresponding to the test images.
        class_names (list): List of class names corresponding to the labels.
        binary (bool): If True, evaluates binary classification; if False, evaluates multiclass classification.
    '''
    
    predictions = model.predict(test_images)
    if binary:
        predictions = (predictions > 0.5).astype(int)
    else:
        predictions = np.argmax(predictions,axis=1)

    cm = confusion_matrix(test_labels, predictions)

    sns.light_palette("lightpink", as_cmap=True)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d',cmap='RdPu', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print(classification_report(test_labels, predictions))