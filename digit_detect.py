from image_edit import convert_bnw, resize_img, to_float_tensor
from MNIST_model import MNIST_Model, get_gradient, predict
from torchvision import torch, datasets, transforms
import torch.nn as nn
import torch.nn.functional as fnc
import cv2
import os

# Global variables
batch = 100
img_num = 3

def main():
	choice = input("Train the model? (Y/N): ").lower()
	if choice == "y":
		num_epochs = int(input("Enter number of training Epochs: "))

	image_name = input("Enter image name (under /test_images/): ")
	if image_name == "":
		print("\nNo image specified. Defaulting to image_{}.jpg...\n".format(img_num))

	# Get MNIST training data
	mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
	mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)

	# Load and shuffle dataset
	train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch, shuffle=True)
	test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch, shuffle=False)

	# Define model
	model = MNIST_Model()

	# saving the model parameters
	if choice == "y":
		get_gradient(num_epochs, model, 0.4, train_loader, test_loader)
		torch.save(model.state_dict(), 'mnist-digit-ocr.pth')

	# Read image and convert to greyscale
	if image_name == "":
		in_path =  os.path.join(".", "test_images", "image_"+str(img_num)+".png")
		out_path = os.path.join(".", "bnw_images", "outfile"+str(img_num)+".png")
	else:
		in_path =  os.path.join(".", "test_images", image_name)
		out_path = os.path.join(".", "bnw_images", image_name)
	convert_bnw(in_path, out_path)

	# Resize and convert image to Tensor for analysis
	input_im = resize_img(out_path)
	input_im = to_float_tensor(input_im)

	# Load existing model if requested
	if choice != "y":
		model_new = MNIST_Model()
		model_new.load_state_dict(torch.load('mnist-digit-ocr.pth'))
		# print('Label:', label, ', Predicted:', predict(input_im, model_new))
		print('Predicted:', predict(input_im, model_new))
	else:
		# print('Label:', label, ', Predicted:', predict(input_im, model))
		print('Predicted:', predict(input_im, model))
		# input("Save? (ctrl-C to save)")

# Calling main
if __name__=="__main__":
	main()

