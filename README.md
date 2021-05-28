# Digit OCR
Python program that performs digit recognition using Logistic Regression. The model is trained on the MNIST dataset and uses the Pytorch ML library.
<br><br>
## Requirements:
* anaconda
* pytorch
* matplotlib
* numpy
* tqdm
* cv2
* PIL

Once you have installed `anaconda` and made a virtual env named `pytorch` (for example), run:
1. `conda activate pytorch`
2. `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
3. `pip install matplotlib numpy tqdm opencv-python Pillow`
<br><br>
## Usage
Enter your pytorch virtual environment in `anaconda` using terminal/cmd, then navigate to the code directory and run:

`python digit_detect.py`
<br><br>
## Links
To use a Google Colaboratory hosted version, <a href="https://colab.research.google.com/drive/1uaWPn638dNEn9BpXKAWxLj5iXDWmWdjr?usp=sharing">click here</a>. The files required to use it are in the `MNIST_ocr/` folder under `For-GDrive/`.
<br><br>
## Examples (from the Google Colab version)
Here is an image of the model that is trained for 5 epochs and then used to identify an image of a figure 8:

![black-tul1p](/readme_images/training.png)

And here, pre-trained parameters from the previous training session are being used to identify another image of a 3:

![black-tul1p](/readme_images/pre-trained.png)
