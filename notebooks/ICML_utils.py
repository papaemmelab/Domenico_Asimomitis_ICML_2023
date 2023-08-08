from PIL import Image
import torchvision.transforms as T
import cv2
import torch
import numpy as np
from Model_class import MyGradCAM
import matplotlib.pyplot as plt
import torchvision.transforms as T

    
def read_png_as_tensor(filename, height, width):
    
    """Read a jpg and convert it to a normalized 4d tensor (1, #channels, H, W)
    Args:
        filename (string): full path of the image file
        height (int): height of the tensor
        width (int): width of the tensor
    Returns:
        input_image (torch.Tensor): 4d tensor (1, #channels, H, W)
    """
    original_image = cv2.imread(filename)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (width, height))
    input_image = np.transpose(original_image, (2, 0, 1))
    input_image = input_image.astype(np.float32) / 255.0
    input_image = torch.from_numpy(input_image).unsqueeze(0)
    
    return original_image,input_image


def custom_generate_cam(input_tens, net, target_class):
    """Compute GradCAM feature map using custom implementation
    Args:
        input_tens (torch.Tensor): 4d input tensor (N,#channels, H, W)
        net (MyModel): model
        model_type (string): type of model used
    Returns:
        attr (numpy.ndarray): 3d numpy.ndarray (N, x, y) - x and y depend on the H and W of the input_tens. For 221x221, x=y=5
    """
    net.eval()
    
    grad_cam = MyGradCAM(net, 'Model.layer4')

    # Run the forward pass
    model_output = grad_cam.model(input_tens)

    # Compute the gradient with respect to the target class
    grad_cam.model.zero_grad()
    
    one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
    one_hot_output[0][target_class] = 1
        
    model_output.backward(gradient=one_hot_output, retain_graph=True)

    # Get the feature maps and gradients
    feature_maps = grad_cam.feature_maps.detach().cpu().numpy()
    gradients = grad_cam.gradient.detach().cpu().numpy()
    
    # Compute the weights using the gradients
    weights = np.mean(gradients, axis=(2,3))
    attr = np.ones(([feature_maps.shape[i] for i in [0,2,3]]), dtype=np.float64)

    for j in range(0,weights.shape[0]):
        for i, w in enumerate(weights[j,:]):
            attr[j] += w * feature_maps[j,i, :, :]

    # Apply ReLU to the CAM
    attr = np.maximum(attr, 0)

    # Normalize the CAM
    attr = (attr - np.min(attr)) / (np.max(attr) - np.min(attr))
    attr = np.uint8(attr * 255)
    
    attr_resized = np.zeros([input_tens.shape[i] for i in [0,2,3]])
    for j in range(0,attr_resized.shape[0]):
        attr_resized[j] = np.uint8(Image.fromarray(attr[j]).resize((input_tens.shape[2],input_tens.shape[3]), Image.ANTIALIAS))

    return attr,attr_resized


def guided_grad_cam(grad_cam_mask, guided_backprop_mask): 
    """
        Guided grad cam computation
    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb
    
    
def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    
    return grayscale_im


def normalize_gradient_image(gradient):
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    
    return gradient


def classify_res(actual,prediction):
    
    ret = 'examine'
    if actual==prediction==1:
        ret = 'TP'
    elif prediction==1 and actual!=prediction:
        ret = 'FP'
    elif actual==prediction==0:
        ret = 'TN'
    elif prediction==0 and actual!=prediction:
        ret = 'FN'
        
    return ret
    