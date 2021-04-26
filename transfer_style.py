# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# Run on GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Establish the model and features
def model_selection():
    # Import VGG-19 network
    vgg19 = models.vgg19(pretrained=True).features.to(device).eval()

    # Adjust modules
    for name, module in vgg19.named_modules():
        # Convert max pooling to average
        if isinstance(module, nn.MaxPool2d):
            vgg19[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)

    # Prevent gradient change
    for param in vgg19.parameters():
        param.requires_grad = False

    # Return modified vgg19
    return vgg19


# Obtain style and content features
def get_features(image, model, layers=None):
    # Set feature dictionary
    features = {}
    x = image

    # Add outputs of layer if they are content or style layers
    for name, layer in enumerate(model):
        x = layer(x)
        if str(name) in layers:
            features[str(name)] = x

    # Return list of features
    return list(features.values())


# Create gram matrix
def gram_matrix(tensor):
    # Obtain tensor dimensions (batch, channels, height, width)
    b, c, h, w = tensor.size()

    # Reshape the tensor to obtain feature space
    feature_space = tensor.view(b, c, h * w)

    # Compute gram matrix
    gram = torch.bmm(feature_space, torch.transpose(feature_space, 1, 2))

    # Return normalized gram matrix
    return gram.div_(h * w)


# Load images
def load_images(style_name, content_name):
    style_img = Image.open("Style_Images/" + style_name + ".jpg")
    content_img = Image.open("Content_Images/" + content_name + ".jpg")

    return style_img, content_img


# Create function to convert the image to a tensor
def img_to_tensor(image, image_size):
    # Create resize and tensor transformation
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    # Transform image
    tensor = transform(image)
    # Add batch dimension and return tensor
    return tensor.unsqueeze(0).to(device)


# Convert tensor to image to confirm resizing
def tensor_to_img(tensor):
    # Clone tensor
    tensor_clone = tensor.cpu().clone()
    # Remove batch dimension
    tensor_clone = tensor_clone.squeeze(0)
    # Convert to image
    image = transforms.ToPILImage()(tensor_clone)

    # Display image
    plt.imshow(image)


# Save image resulting from NST
def save_nst_image(tensor, content_img=None, style_img=None):
    # Create filename
    filename = content_img + "_" + style_img + ".jpg"

    # Create directory for saving generated images using content image name
    directory = "Generated_Images/" + content_img
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        pass

    # Save image and print confirmation
    save_image(tensor, os.path.join(directory, filename), normalize=True)
    print("Transferred Image Saved!")


# Run NST algorithm
def run_nst(
    iters,
    alpha,
    beta,
    img_size,
    model,
    content_img,
    style_img,
    input_img,
    content_layers,
    style_layers,
    style_weights,
):

    # Load images
    style, content = load_images(style_img, content_img)

    # Convert images to tensors
    style_tensor = img_to_tensor(style, img_size)
    content_tensor = img_to_tensor(content, img_size)

    # Obtain style representation
    style_rep = get_features(style_tensor, model, style_layers)

    # Convert style features to gram matrices
    style_gram = [gram_matrix(style).detach() for style in style_rep]

    # Obtain content representation
    content_rep = get_features(content_tensor, model, content_layers)[0].detach()

    # Create input image tensor
    if input_img == "content":
        noise = content_tensor.clone().requires_grad_().to(device)
    elif input_img == "style":
        noise = style_tensor.clone().requires_grad_().to(device)
    else:
        noise = torch.randn(content_tensor.size(), requires_grad=True, device=device)

    # Set optimizer
    optimizer = optim.LBFGS([noise])

    # Run model
    for i in tqdm(range(iters)):

        def closure():
            # Zero the gradient
            optimizer.zero_grad()

            # Initialize content and style loss
            content_loss = 0
            style_loss = 0

            # Calculute content features of noise
            noise_content = get_features(noise, model, content_layers)[0]

            # Calculate content loss
            content_loss = nn.MSELoss()(noise_content, content_rep)

            # Calculate style features of noise
            noise_style = get_features(noise, model, style_layers)

            # Convert style features to gram matrices
            noise_gram = [gram_matrix(style) for style in noise_style]

            # Calculate loss for each gram matrix
            for j in range(len(noise_gram)):
                style_loss += (
                    nn.MSELoss()(noise_gram[j], style_gram[j]) * style_weights[j]
                )

            # Calculate total loss
            total_loss = alpha * content_loss + beta * style_loss
            total_loss.backward()

            # Backpropagate
            return total_loss

        # Update
        optimizer.step(closure)

    return noise
