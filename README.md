# Person Re-Identification with Siamese Networks

## Project Overview

This project implements a Person Re-Identification system using Siamese Networks and Deep Learning with PyTorch. The system is designed to identify and match individuals across different images, which is crucial for various applications in surveillance and security.

## Features

- Utilizes a Siamese Network architecture with EfficientNet as the backbone
- Implements Triplet Loss for training the network
- Includes data preprocessing and augmentation
- Provides functions for training, evaluation, and inference
- Generates embeddings for quick similarity comparisons

## Requirements

- Python 3.x
- PyTorch
- torchvision
- timm
- pandas
- numpy
- matplotlib
- scikit-image
- scikit-learn
- tqdm

## Dataset

The project uses a Person Re-Identification dataset. The dataset should be structured as follows:

- A CSV file (`train.csv`) containing columns: Anchor, Positive, Negative
- An image directory containing all the images referenced in the CSV file

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/person-re-identification.git
   cd person-re-identification
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Update the `DATA_DIR` and `CSV_FILE` variables in the script to point to your dataset.

## Usage

1. Train the model:
   ```
   python person_reid_train.py
   ```

2. Generate embeddings for the database:
   ```
   python generate_embeddings.py
   ```

3. Run inference:
   ```
   python person_reid_inference.py
   ```

## Model Architecture

The model uses an EfficientNet-B0 backbone, pretrained on ImageNet, with a custom classifier layer to generate embeddings of size 512.

## Training

The model is trained using Triplet Margin Loss. The training process involves:
- Loading images in triplets (Anchor, Positive, Negative)
- Generating embeddings for each image
- Computing the triplet loss
- Backpropagation and optimization

## Inference

For inference:
1. Load a query image
2. Generate its embedding using the trained model
3. Compare the embedding with the pre-computed database embeddings
4. Retrieve the closest matches based on Euclidean distance

## Results

The script includes functionality to visualize the closest matches for a given query image.

## Future Improvements

- Implement hard triplet mining for more efficient training
- Experiment with different backbone architectures
- Add data augmentation techniques to improve robustness
- Implement a GUI for easier interaction with the system

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

[MIT License](LICENSE)

## Acknowledgments

- Thanks to the creators of the Person Re-Identification dataset used in this project.
- This project uses the `timm` library for efficient model implementations.
