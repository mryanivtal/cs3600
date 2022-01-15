r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""



# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=27, 
        h_dim=256, 
        z_dim=128, 
        x_sigma2=0.001, 
        learn_rate=0.0001, 
        betas=(0.9, 0.999))
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
Sigma^2 controls the width of the gaussian.
Making it larger will make the distribution wider, and the sampling gets more "freedom" to vary fro the original training data (because in avg. z will be further to Mu)
making it smaller will make the distribution narrower, making the samples more like the train data (because in avg. z will be closer to Mu)
"""

part2_q2 = r"""
**Your answer:**
Reconstruction loss - measures (MSE) how much of the data that is required to reconstruct the original sample was lost in encode-decode process.    
Bigger loss here  will signify that we lost important "Essence" data on the sample.   

KL loss - estimates the distance between the posterior distribution and the approximated one, so that sampling from that distribution will generate data close the real one.
Also makes sure the resulting distribution is not a point but a gaussian
"""

part2_q3 = r"""
**Your answer:**
Because we do not know the actual prior distribution (only assume it exists and that it can be approximated by a Gaussian), we need to find it given the evidence.
Once we have the approx. distribution we can use it to generate new samples
"""

part2_q4 = r"""
**Your answer:**
We use log for numerical stability.
Since sigma is typically small, log expands its range for sigma in [0,1] significantly.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    # hypers = dict(
    #     batch_size=0,
    #     z_dim=0,
    #     data_label=0,
    #     label_noise=0.0,
    #     discriminator_optimizer=dict(
    #         type="",  # Any name in nn.optim like SGD, Adam
    #         lr=0.0,
    #         # You an add extra args for the optimizer here
    #     ),
    #     generator_optimizer=dict(
    #         type="",  # Any name in nn.optim like SGD, Adam
    #         lr=0.0,
    #         # You an add extra args for the optimizer here
    #     ),
    # )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=53,
        z_dim=100,
        data_label=1,
        label_noise=0.1,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=2e-4,
            betas=(0.5, 0.999),
            #weight_decay=2e-3
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=2e-4,
            betas=(0.5, 0.999)
            # You an add extra args for the optimizer here
        )
    )
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
In GAN, we actually train two different models: the Generator, which generates the data, and the discriminator, which classifies it.
Those models have different loss functions and we do not want Torch to sum gradients on one when we are training the other.

When we train the discriminator, we use generator-generated images to create samples, but do not want the discriminator backprop to impact the generator weights, so we set gradients to false.
When we train the generator, we set thits grads to true so it can update wiights and improve.
"""

part3_q2 = r"""
**Your answer:**
No, we should not.
In Gan, it is almost irrelevant to set stopping criteria based on the loss functions.
In a good scenario, Discriminator loss should stay around 50% (Meaning it cannot tell between the generated samples and real ones) and the generator loss should 
typically stay around an arbitrary number.      Because both should improve in parallel, there is no visible loss changes if noone "wins".

If the discriminator is far from 50%, or gets stuck in 1/0, this means the equilibrium is not good, and we should probably fix the hyperparams.


"""

part3_q3 = r"""
**Your answer:**
Our VAE samples are much nicer and less blurry, probably because of two reasons:
1. The GAN is harder to calibrate and does not perform amazingly well in our case
2. Generally, in GAN the generator never "sees" the real samples, but creates them based on kind of a "hot/Cold" game with the discriminator.
   It doesnt have a reconstructin loss etc. so it "makes up" data based on True / False inputs only.   This creates photos that are not strictly like the originals.
   

"""

# ==============
