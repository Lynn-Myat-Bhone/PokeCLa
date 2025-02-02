# Pokemon Image Classification

This project is a deep learning model for classifying images of Pokemon using TensorFlow and Keras. The model is built on top of a pre-trained ResNet50 architecture and fine-tuned for the specific task of Pokemon classification.

## Project Structure

- `model.ipynb`: Jupyter notebook containing the code for data preprocessing, model building, training, and evaluation.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## Setup

1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Download the dataset:
    ```sh
    # !kaggle datasets download -d thedagger/pokemon-generation-one
    # !unzip /content/pokemon-generation-one.zip
    ```

## Data Preprocessing

- The dataset is loaded using `image_dataset_from_directory` with a validation split of 20%.
- Data augmentation techniques such as random flipping, rotation, zoom, contrast, brightness, and translation are applied to the training dataset.
- The images are resized to 224x224 pixels and preprocessed using the `preprocess_input` function from `tensorflow.keras.applications.resnet50`.

## Model Architecture

- The base model is a pre-trained ResNet50 without the top classification layers.
- The base model's layers are frozen to prevent them from being updated during training.
- A new model is created on top of the base model with additional layers:
    - GlobalAveragePooling2D
    - Dense layer with 128 units and ReLU activation
    - Dropout layer with a rate of 0.5
    - Dense output layer with softmax activation for classification

## Training

- The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the metric.
- Early stopping is used to monitor the validation loss and restore the best weights.
- Class weights are computed to handle class imbalance.
- The model is trained for 10 epochs with the specified class weights and early stopping callback.

## Evaluation

- The training and validation loss are plotted to visualize the model's performance.
- The model's predictions can be evaluated using metrics such as classification report (commented out in the notebook).

## Usage

1. Run the Jupyter notebook `model.ipynb` to execute the entire workflow from data preprocessing to model evaluation.
2. Modify the notebook as needed to experiment with different hyperparameters, data augmentation techniques, or model architectures.

## License

This project is licensed under the MIT License.