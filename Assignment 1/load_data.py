import numpy as np
import os
import random
import torch


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    import imageio
    image = imageio.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}, device = torch.device('cpu')):
        """
        Args:
            num_classes: int
                Number of classes for classification (N-way)
            
            num_samples_per_class: int
                Number of samples per class in the support set (K-shot).
                Will generate additional sample for the querry set.
                
            device: cuda.device: 
                Device to allocate tensors to.
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]
        self.device = device

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: str
                train/val/test set to sample from
                
            batch_size: int:
                Size of batch of tasks to sample
                
        Returns:
            images: tensor
                A tensor of images of size [B, K+1, N, 784]
                where B is batch size, K is number of samples per class, 
                N is number of classes
                
            labels: tensor
                A tensor of images of size [B, K+1, N, N] 
                where B is batch size, K is number of samples per class, 
                N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        #############################
        #### YOUR CODE GOES HERE ####
        #############################

        # SOLUTION:

        classes = self.num_classes
        shots = self.num_samples_per_class

        images = torch.zeros(batch_size, shots + 1, classes, 784)
        labels = torch.zeros(batch_size, shots + 1, classes, classes)

        for task in range(batch_size):

            # diagonal matrix N*N matrix with diag = 0 ,..., N.
            # in particular, each row is the one-hot vector wrt its class label.
            one_hots = np.fill_diagonal(
                np.zeros(classes, classes), 
                np.arange(0, classes, dtype=float)
            )

            # randomly select character sets (the N)
            chars = np.random.choice(folders)
            # randomly select shots for each N (the K)
            dta_train = get_images(chars, one_hots, nb_samples=shots)
            # randomly select test, with mixed up label
            dta_test = get_images(chars, one_hots, nb_samples=1, shuffle=True)

            # control flow for populating query data in loop
            flag = False
            for shot in range(shots):
                for clss in range(classes):
                    # fix a shot, iterate through the class representatives 
                    # for that shot and populate the training data.
                    label, img = dta_train[clss*shots + shot]
                    img = image_file_to_array(img, 28 * 28)
                    images[task, shot, clss] = torch.from_numpy(img)
                    labels[task, shot, clss] = torch.from_numpy(label)

                    if not flag:
                        # populate the query data.
                        label, img = dta_test[clss]
                        img = image_file_to_array(img, 28*28)
                        images[task, shots + 1, clss] = torch.from_numpy(img)
                        labels[task, shots + 1, clss] = torch.from_numpy(label)
                
                # after first iteration of outer loop, 
                # query data will be fully populated.
                flag = True


