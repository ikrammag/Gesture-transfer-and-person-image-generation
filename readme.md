
# Posture-Guided Image Synthesis of a Person

This project demonstrates motion transfer from a source video to a target person using PyTorch. Designed as an introduction to Generative Adversarial Networks (GANs), this project guides you through each stage required to generate realistic images from a skeleton-based representation of the target person. By the end, you'll have a working model capable of synthesizing movements from one individual and applying them to another in a visually coherent manner.

## Objectives

- Extract poses from a source video and apply them to a target video to create a new video.
- Use machine learning techniques to generate images of the target person in the extracted poses.
- Implement a simple neural network and a GAN to improve image generation quality.

## Installation and Requirements

1. Install the required libraries:
   - `numpy`
   - `pytorch`
   - `mediapipe` (Mediapipe automatically includes OpenCV, so make sure no version of OpenCV is installed to avoid conflicts).
2. Place the data in the appropriate folder, typically `data/....`.

## Code Structure

- **VideoReader**: Functions for video playback and image retrieval.
- **Vec3**: Representation of 3D points.
- **Skeleton**: Class that manages 3D skeleton positions, with an option to reduce to 13 joints in 2D.
- **VideoSkeleton**: Associates each video frame with a skeleton and stores the images of a video.
- **GenNearest**: A generator that creates an image by finding the nearest skeleton from a video dataset using Euclidean distance, returning the corresponding image.
- **GenVanillaNN**: A neural network-based generator that creates images from skeleton data by transforming skeletons into images using a deep convolutional network. The model is trained with a dataset of skeletons and corresponding images and can generate realistic images from new skeleton poses.
- **GenGAN**: A Generative Adversarial Network (GAN) that generates images from skeleton data. It consists of a generator (GenVanillaNN) that creates images from skeleton poses and a discriminator that distinguishes real from fake images. During training, the generator improves by fooling the discriminator, and the model is optimized using binary cross-entropy loss with Adam. Once trained, it can generate images from new skeleton inputs.
- **DanceDemo**: Main class for the dance demo.

## Approach

### Step 1: Skeleton Extraction

- **Implementation**:
  - Use Mediapipe to extract skeleton data from videos. The code uses `VideoSkeleton` and `VideoReader` to process and extract skeletons from video frames. The video is split into individual frames to process them separately. This matches the code setup where frames are processed to extract skeleton data for training.

### Step 2: Nearest Neighbor Generation

- **Implementation**:
  - This method searches for the most similar skeleton posture in a dataset and retrieves the corresponding image, This would likely involve comparing skeleton features in a feature space and selecting the closest match.

### Step 3: Direct Neural Network

- **Implementation**:
  - The `GenVanillaNN` class implements a simple neural network to generate images from skeletons. The code uses a neural network to convert skeleton data (input as vectors) into images. This aligns with the approach of training a network to directly map skeletons to images.

### Step 4: Neural Network with Stick Figure Image

- **Implementation**:
  
   - The `GenVanillaNN2` integrate an intermediate stick figure representation to improve the training process. This is achieved through the (`draw_reduced`) method in the `Skeleton` class, which simplifies the skeleton to a stick figure. This stick figure serves as an intermediate visual step, helping the network learn key pose features before generating the final image. By training on this representation, the model can better capture the structure of each pose, making it easier to produce realistic images in the final output.
- 
### Step 5: GAN Implementation

- **Implementation**:
  - This is well implemented in the code with the `GenGAN` class, which adds a discriminator to improve the quality of generated images. The generator (`GenVanillaNN`) creates images from skeletons, while the discriminator differentiates real images from generated ones. This GAN setup allows the model to improve its image generation capability over time.

## Execution

To run this project, use the following commands in your terminal:

1.  **Run the Nearest Neighbor Generation**:

    ```bash
    "python GenNearest.py" modify the DanceDemo.py file to set GEN_TYPE = 1
     **Run the Dance Demo**
    "python DanceDemo.py"
    ```

2.  **Run the GenVanillaNN**:
    ```bash
    "python GenVanillaNN.py"
    Run the Dance Demo: After training, modify the DanceDemo.py file to set GEN_TYPE = 2 so it uses the trained model. Then, run the command below to generate new images from the skeletons in the target video

         "python DanceDemo.py"

    This command will generate images by applying poses from the source video (taichi2.mp4) to a target person.

      3.  **Run the GenVanillaNN**:




3.  **Run the GenVanillaNN**
     ```bash
    "python GenVanillaNN2.py"
    Run the Dance Demo: After training, modify the DanceDemo.py file to set GEN_TYPE = 3 so it uses the trained model. Then, run the command below to generate new images from the skeletons in the target video

         "python DanceDemo.py"

    This command will generate images by applying poses from the source video (taichi2.mp4) to a target person.
4.  **Run the GenGAN**:
    
     ```bash

      Run the Dance Demo: After training, modify the DanceDemo.py file to set GEN_TYPE = 4 so it uses the trained model. Then, run the command below to generate new images from the skeletons in the target video

         "python DanceDemo.py"

    This command will generate images by applying poses from the source video (taichi2.mp4) to a target person.
         "python GenGAN.py"

## Conclusion

This project demonstrates the application of GANs for motion transfer between individuals, a promising approach for creating realistic animated content from simple skeleton representations. 


