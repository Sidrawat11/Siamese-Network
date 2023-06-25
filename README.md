# Siamese-Network
This is a Deep Learning Facial Recognition model that uses Siamese-Network

## Dataset

The code utilizes the [Labelled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) dataset for training and testing. 

### Dataset Download

Please download the LFW dataset from the [official website](http://vis-www.cs.umass.edu/lfw/) and extract the files.

### Dataset Directory

Once the dataset is downloaded and extracted, place the dataset files in the `lfw` directory. The `lfw` directory should be created in the same directory as the code files.

The directory structure should look like this:



# Image Collection and Verification

This code provides functionality for collecting images from a webcam and performing verification using a siamese neural network model. It allows the user to collect anchor, positive, and negative images, preprocess them, build a siamese model, train the model, evaluate its performance, and perform real-time verification using the trained model.

## Dependencies

The following dependencies are required to run the code:
- OpenCV (cv2)
- NumPy
- Matplotlib
- TensorFlow (tf)

Please make sure to install these dependencies before running the code.

## Usage

1. Set up the required directories:
   - The code assumes the existence of the following directories:
     - `data/positive` - Directory to store positive images
     - `data/negative` - Directory to store negative images
     - `data/anchor` - Directory to store anchor images
     - `lfw` - Directory containing the Labelled Faces in the Wild dataset
     - `application_data/input_image` - Directory to store the input image for verification
     - `application_data/verification_images` - Directory to store verification images

2. Uncompress the Labelled Faces in the Wild dataset:
   - Run the command `tar -xf lfw.tgz` to extract the dataset.

3. Collecting Images:
   - Run the code to start the webcam.
   - Press the 'a' key to collect an anchor image.
   - Press the 'p' key to collect a positive image.
   - The collected images will be stored in the respective directories mentioned above.
   - Press the 'q' key to exit the image collection process.

4. Building the Siamese Model:
   - The code defines a siamese model with L1 distance as the similarity metric.
   - The model architecture and summary are provided in the code.

5. Training the Siamese Model:
   - The collected images are split into training and testing partitions.
   - The siamese model is trained using the training data.
   - The training process saves checkpoints every 10 epochs.

6. Evaluating Model Performance:
   - Model metrics such as Precision and Recall are calculated using the test data.

7. Saving and Loading the Model:
   - The trained model can be saved using `siamese_model.save('NewModel.h5')`.
   - The saved model can be loaded using `tf.keras.models.load_model('NewModel.h5')`.

8. Verification:
   - Real-time verification can be performed using the webcam and the trained model.
   - Point the webcam towards the person to be verified.
   - Press the 'v' key to capture the input image.
   - The input image will be saved in `application_data/input_image` folder.
   - The model will compare the input image with verification images in `application_data/verification_images` folder.
   - The verification result will be printed.

## Note

- Please ensure that the required directories are set up before running the code.
- Adjust the parameters such as batch size, image size, and model architecture as needed for your specific use case.
