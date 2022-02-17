import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.nn import ReLU, BatchNorm2d, Conv2d, ConvTranspose2d
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        modules = []
        in_channels, hight, width = in_size
        out_channels = 1024

        # CNN layers of discriminator (Encoder):
        from hw4.autoencoder import EncoderCNN
        self.encoder_cnn = EncoderCNN(in_channels, out_channels)

        # classification layer (FC):
        # first find out and store shape of cnn output to FC layer:
        with torch.no_grad():
            sample = torch.zeros(1, *in_size)
            self.cnn_features_shape = self.encoder_cnn(sample).shape
            self.cnn_out_flattened_size = self.encoder_cnn(sample).flatten().shape[0]

        # now add classification layer
        self.classification_fc = nn.Linear(self.cnn_out_flattened_size, 1, bias=True)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        cnn_output = self.encoder_cnn(x)
        cnn_output = cnn_output.view(x.shape[0], -1)
        y = self.classification_fc(cnn_output)
        # ========================
        return y

class SNEncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        modules = [] 
        convnet_arch = [(in_channels, 64, 5, 2, 2, False, 0.9), (64, 128, 5, 2, 2, False, 0.9), (128, 256, 5, 2, 2, False, 0.9), (256, out_channels, 5, 2, 2, False, 0.9)]
        num_conv_layers = len(convnet_arch)

        for i, layer in enumerate(convnet_arch): 
            in_chan, out_chan, kernel_size, padding, stride, bias, momentum = layer

            modules.append(spectral_norm(Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)))
            if i < num_conv_layers - 1:
                modules.append(BatchNorm2d(num_features=out_chan, momentum=momentum))
                modules.append(ReLU())
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)

class SNDiscriminator(Discriminator):
    def __init__(self, in_size):
        super().__init__(in_size)
        in_channels, hight, width = in_size
        out_channels = 1024
        self.encoder_cnn =  SNEncoderCNN(in_channels, out_channels)

class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        #hint (you dont have to use....)
        
        modules = []
        
        #save latent feature map size for future use
        self.cnn_in_channels = 1024      
        self.cnn_featurmap_size = featuremap_size

        # First layers to convert latent vector z to cnn feature map as required
        self.z_to_features_fc = nn.Linear(z_dim, featuremap_size*featuremap_size*self.cnn_in_channels, bias=True)

        # now conv layers for image decoder
        # each layer params: (in_channels, out_channels, kernel_size, padding, output_padding, stride, bias, normalization momentum)
        convnet_arch = [(self.cnn_in_channels, 512, 5, 2, 1, 2, False, 0.9), (512, 256, 5, 2, 1, 2, False, 0.9), (256, 128, 5, 2, 1, 2, False, 0.9), (128, out_channels, 5, 2, 1, 2, False, 0.9)]
        num_conv_layers = len(convnet_arch)

        for i, layer in enumerate(convnet_arch): 
            in_chan, out_chan, kernel_size, padding, output_padding, stride, bias, momentum = layer

            modules.append(ConvTranspose2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel_size, padding=padding, output_padding=output_padding, stride=stride, bias=bias))
        
            if i < num_conv_layers - 1:
                modules.append(BatchNorm2d(num_features=out_chan, momentum=momentum))
                modules.append(ReLU())
                
        self.cnn_decoder = nn.Sequential(*modules)   #maybe add some stuff later, better keep flexible

        # #TEST DELETE YANIV
        # z = torch.zeros([1, 128])
        # print('original vector shape:', z.shape)

        # sample = self.z_to_features_fc(z).reshape(z.shape[0], self.cnn_in_channels, self.cnn_featurmap_size, self.cnn_featurmap_size)

        # print(sample.shape)

        # for module in modules:
        #     sample = module(sample)
        #     print(sample.shape)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        samples = torch.randn(n, self.z_dim, device=device)
        if with_grad:
            samples = self.forward(samples)
        else:
            with torch.no_grad():
                samples = self.forward(samples)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        features = self.z_to_features_fc(z)
        features = features.reshape(z.shape[0], self.cnn_in_channels, self.cnn_featurmap_size, self.cnn_featurmap_size)
        x = self.cnn_decoder(features)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======

    # Add random noise per label from [+-label_noise / 2]
    y_data_noise = torch.rand(y_data.shape, device=y_data.device) * label_noise - label_noise / 2
    y_generated_noise = torch.rand(y_generated.shape, device=y_data.device) * label_noise - label_noise / 2

    data_labels = data_label + y_data_noise
    generated_labels = (1-data_label)+y_generated_noise

    loss_func = torch.nn.BCEWithLogitsLoss()
    loss_data = loss_func(y_data, data_labels)
    loss_generated = loss_func(y_generated, generated_labels)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    loss_func = torch.nn.BCEWithLogitsLoss()
    data_labels = torch.ones(y_generated.shape, device=y_generated.device) * float(data_label)
    loss = loss_func(y_generated, data_labels)
    # ========================
    return loss

def wgan_discriminator_loss_fn(y_data, y_generated):
    loss = -torch.mean(y_data) + torch.mean(y_generated)
    return loss

def wgan_generator_loss_fn(y_generated):
    loss = -torch.mean(y_generated)
    return loss

def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()
    generated_samples = gen_model.sample(x_data.shape[0], with_grad=False)
    generated_samples_score = dsc_model(generated_samples)

    y_samples_score = dsc_model(x_data)
    dsc_loss = dsc_loss_fn(y_samples_score, generated_samples_score)

    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()
    generated_samples = gen_model.sample(x_data.shape[0], with_grad=True)
    generated_samples_score = dsc_model(generated_samples)

    gen_loss = gen_loss_fn(generated_samples_score)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    torch.save(gen_model, checkpoint_file)
    saved=True
    # ========================

    return saved