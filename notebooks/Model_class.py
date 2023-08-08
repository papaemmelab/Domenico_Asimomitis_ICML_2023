from torchvision import models
import torch
import torch.nn as nn
from torch.nn import ReLU


class MyModel(nn.Module):
    def __init__(self, pretrained=True, n_features=2048, n_classes=2, n_channels=3, drop_rate=0., aux_logits=True):

        super(MyModel,self).__init__()
            
        self.Model = models.resnet50(pretrained=pretrained)
        self.Model.fc = torch.nn.Linear(in_features=n_features, out_features=n_classes)    
        self.n_classes = n_classes
            
    def forward(self, x):

        x = self.Model(x)
        return x
    
    
class MyGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradient = None

        self.model.eval()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradient = grad_out[0]

        def forward_function(module, input, output):
            self.feature_maps = output

        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_layer = module
                break

        target_layer.register_forward_hook(forward_function)
        target_layer.register_backward_hook(hook_function)

        
class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model,first_layer_str):
        self.model = model
        self.gradients = None
        self.first_layer = first_layer_str
        self.forward_relu_outputs = []
        
        # Put model in evaluation mode
        self.model.eval()
        
        self.hooksfor = {}
        self.hooksback = {}
        self.update_relus()
        self.hook_layers()


    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            
        # Register hook to the first layer
        
        first_layer = None
        for name, module in self.model.named_modules():
            if name == self.first_layer:
                first_layer = module
                break
        
        self.hookbacksingle = first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output            
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.named_modules():    
            if isinstance(module, nn.ReLU): 
                self.hooksback[pos] = module.register_backward_hook(relu_backward_hook_function)
                self.hooksfor[pos] = module.register_forward_hook(relu_forward_hook_function)


    def generate_gradients(self, input_tens, target_class=None):
        # Forward pass
        model_output = self.model(input_tens)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        if target_class is None:
            one_hot_output = torch.ones_like(model_output)
        else:
            one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
            one_hot_output[0][target_class] = 1
            
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        gradients_as_arr = self.gradients.data.numpy()
        return gradients_as_arr
