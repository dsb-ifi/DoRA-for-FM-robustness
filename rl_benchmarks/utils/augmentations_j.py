import numpy as np
import tomli
import tools
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from skimage.color import rgb2hsv, hsv2rgb
from PIL import Image

def get_test_slide():
    slide="/mnt/c/Users/vilde/Documents/Figures/image_eg/original_patch.png"
    return slide

def preprocess(image_bag, conf=None, scan_mean=None, scan_stddev=None):
    """Preprocess image

    Args:
      image_bag: float32/uint8 tensor [bag_size, num_channels, height, width], values in [0, 255].
                 Shape: [batch_size, bag_size, num_channels, height, width]
      conf: Configuration dictionary.
      scan_mean: float32 tensor [3], if present
      scan_stddev: float32 tensor [3], if present

    Returns:
      image: float32 tensor [bag_size, target_height, target_width, num_channels],
             values in approx [-3, 3]
    """
    if not conf:
        conf_path = "/home/vilde/code/Phikon/HistoSSLscaling/j_config.toml"
        with open(conf_path, "rb") as f:
            conf = tomli.load(f)
    if conf['input']['random_background_fraction'][1] > 0:
        ## print("Applying background augmentation")
        min_frac = conf['input']['random_background_fraction'][0]
        max_frac = conf['input']['random_background_fraction'][1]
        print("Not implemented")
        #image_bag = tf.map_fn(lambda tile: add_random_background_padding(tile, min_frac, max_frac), image_bag)

    #print("Input has type", type(image_bag))#, type(image_bag[0]))
    #print("type is PIL?", type(image_bag)==Image.Image) # Gives True
    tensor_trans = transforms.Compose([transforms.PILToTensor()])
    image_bag = tensor_trans(image_bag)
    #print("Image_bag size", image_bag.size(), len(image_bag.size()))
    if len(image_bag.size())<4:
        image_bag = image_bag.unsqueeze(0)

    corr_tiles = True if conf['input']['input_data_format'] == 'tf_data_corresponding_pretiled' else False
    if (conf['input']['resize_crop'] or conf['input']['random_crop'] or
            conf['input']['central_crop'] or conf['input']['distort_orientation']):
        ## print("Distort geometry")
        image_bag = geometric_distortions(image_bag, conf['input']['resize_crop'],
                                          conf['input']['random_crop'],
                                          conf['input']['central_crop'],
                                          conf['input']['distort_orientation'],
                                          conf['input']['target_height'],
                                          conf['input']['target_width'],
                                          conf['input']['num_channels'],
                                          corresponding_tiles=corr_tiles,
                                          corr_equal_rot=conf['input']['corr_tiles_equal_rot'])

    image_bag = image_bag.type(torch.FloatTensor)
    image_bag = torch.div(image_bag, 255.0)

    diff = np.sum(([torch.max(image_bag[:,j,:,:]).tolist()-torch.min(image_bag[:,j,:,:]).tolist() for j in [0,1,2]]))
    if diff < 1e-4:
        print("Image variation is distroyed!", diff, " 0")

    if conf['input']['distort_colorspace'] is not None and conf['input']['num_channels'] == 3:
        ## print("Distort colors")
        img_bag_shape = image_bag.size()  # [num_images, num_channels, height, width]
        if corr_tiles and not conf['input']['distort_corr_tiles_equal']:
            ## print("Different color distortion for paired tiles")
            image_bag = torch.reshape(image_bag, 
                                   [2, img_bag_shape[0] // 2, img_bag_shape[1], img_bag_shape[2], img_bag_shape[3]])
        else:
            image_bag = torch.reshape(image_bag, 
                                   [1, img_bag_shape[0], img_bag_shape[1], img_bag_shape[2], img_bag_shape[3]])
        for i in range(len(image_bag)):
            image_bag[i] = color_distortion(image_bag[i],
                                     conf['input']['colorspace'],
                                     conf['input']['distort_colorspace'],
                                     conf['input']['shift_interval_size'],
                                     conf['input']['scale_interval_size'],
                                     conf['input']['distort_channel_per_bag'], 
                                     conf['input']['exponential_scales'])

        image_bag = torch.reshape(image_bag, [img_bag_shape[0], img_bag_shape[1], img_bag_shape[2], img_bag_shape[3]])
    elif conf['input']['distort_colorspace'] == 'gl' and conf['input']['num_channels'] == 1:
        ## print("Distort grayscale")
        print("Not implemented")
        # image_bag = grayscale_distortion(image_bag,
        #                                  conf['input']['contrast_interval_size'],
        #                                  conf['input']['brightness_interval_size'],
        #                                  conf['input']['distort_channel_per_bag'])
    else:
        if conf['input']['colorspace'] == 'hed':
            print("hed colorspace not implemented")
            # image_bag = rgb_to_hed(image_bag)
            # image_bag = scale_hed(image_bag, 0.0, 1.0)


    diff = np.sum(([torch.max(image_bag[:,j,:,:]).tolist()-torch.min(image_bag[:,j,:,:]).tolist() for j in [0,1,2]]))
    if diff < 1e-4:
        print("Image variation is distroyed!", diff, " 1")
        
    if conf['input']['blur_distortion'][0] > 0 and conf['input']['blur_distortion'][1] > 0:
        ## print('Applying Gaussian blur')
        kernel_size = conf['input']['blur_distortion'][0]   #.type(torch.FloatTensor)
        sigma_max = conf['input']['blur_distortion'][1]     #.type(torch.FloatTensor)
        
        if corr_tiles and not conf['input']['distort_corr_tiles_equal']:
            img_bag_shape = image_bag.size()
            image_bag = torch.reshape(image_bag, 
                                [2, img_bag_shape[0] // 2, img_bag_shape[1], img_bag_shape[2], img_bag_shape[3]])
            image_bag = torch.unbind(image_bag)
            assert len(image_bag) == 2, 'Expected two image bags (AP and XR), got {}'.format(len(image_bag))
            for i in range(len(image_bag)):
                image_bag[i] = random_gaussian_blur(image_bag[i], kernel_size, sigma_max)
            image_bag = torch.stack(image_bag)
            image_bag = torch.reshape(image_bag, [img_bag_shape[0], img_bag_shape[1], img_bag_shape[2], img_bag_shape[3]])
        else:
            image_bag = random_gaussian_blur(image_bag, kernel_size, sigma_max)
    
    diff = np.sum(([torch.max(image_bag[:,j,:,:]).tolist()-torch.min(image_bag[:,j,:,:]).tolist() for j in [0,1,2]]))
    if diff < 1e-4:
        print("Image variation is distroyed!", diff, " 2")

    if conf['input']['gaussian_noise'] > 0:
        ## print('Applying Gaussian Noise')
        if not conf['input']['distort_channel_per_bag']:
            ## print('Gaussian Noise only implemented for per bag augmentation')
            raise NotImplementedError('Gaussian Noise only implemented for per bag augmentation')
        variance_max = torch.tensor(conf['input']['gaussian_noise'], dtype=torch.float32)
        if corr_tiles and not conf['input']['distort_corr_tiles_equal']:
            img_bag_shape = image_bag.size()
            image_bag = torch.reshape(image_bag, 
                                   [2, img_bag_shape[0] // 2, img_bag_shape[1], img_bag_shape[2], img_bag_shape[3]])
            #image_bag = tf.map_fn(lambda bag: random_gaussian_noise(bag, variance_max), image_bag)
            for i in range(len(image_bag)):
                image_bag[i] = random_gaussian_noise(image_bag[i], variance_max)
            image_bag = torch.reshape(image_bag, [img_bag_shape[0], img_bag_shape[1], img_bag_shape[2], img_bag_shape[3]])
        else:
            image_bag = random_gaussian_noise(image_bag, variance_max)
    

    if conf['input']['periodic_noise_amplitude'] > 0:
        print("Periodic noise is not implemented")
        # assert isinstance(conf['input']['periodic_noise_period'], list)
        # if not conf['input']['distort_channel_per_bag']:
        #     log.error('Periodic Noise only implemented for per bag augmentation')
        #     raise NotImplementedError('Periodic Noise only implemented for per bag augmentation')
        # amplitude_max = tf.cast(conf['input']['periodic_noise_amplitude'], tf.float32)
        # if corr_tiles and not conf['input']['distort_corr_tiles_equal']:
        #     img_bag_shape = image_bag.get_shape().as_list()
        #     image_bag = tf.reshape(image_bag, 
        #                            [2, img_bag_shape[0] // 2, img_bag_shape[1], img_bag_shape[2], img_bag_shape[3]])
        #     image_bag = tf.map_fn(lambda bag: random_periodic_noise(bag, amplitude_max, conf['input']['periodic_noise_period']), 
        #                           image_bag)
        #     image_bag = tf.reshape(image_bag, [img_bag_shape[0], img_bag_shape[1], img_bag_shape[2], img_bag_shape[3]])
        # else:
        #     image_bag = random_periodic_noise(image_bag, amplitude_max, conf['input']['periodic_noise_period'])

    # This should be done in both training and eval (if it is done at all).
    if conf['input']['per_image_standardization']:
        ## print("Per tile standardization")
        # zero mean and unit variance
        #image_bag = tf.map_fn(lambda tile: tf.image.per_image_standardization(tile), image_bag)
        for i in range(len(image_bag)):
            m = torch.mean(image_bag[i])
            adjusted_s = np.max([torch.std(image_bag[i]), 1/torch.sqrt(torch.tensor(image_bag[i].numel()))])
            image_bag[i] = torch.divide( torch.subtract(image_bag[i],m), adjusted_s)
    if conf['input']['per_scan_standardization']:
        ## print("Per scan standardization")
        print("Not implemented")
        # assert scan_mean is not None, "Attempting per_scan_standardization with scan_mean=None"
        # assert scan_stddev is not None, "Attempting per_scan_standardization with scan_stddev=None"
        # # zero mean and unit variance
        # image_bag = tf.map_fn(lambda tile:
        #                       tf.divide(tf.subtract(tile, scan_mean), scan_stddev),
        #                       image_bag)
    if conf['input']['whole_batch_standardization']:
        ## print('Whole batch standardization')
        mean = torch.divide(tf.constant(conf['input']['train_mean']), 255.0)
        std = torch.divide(tf.constant(conf['input']['train_std']), 255.0)
        for i in range(len(image_bag)):
            image_bag[i] = torch.divide( torch.subtract(image_bag[i],mean), std)

    num_images = conf['input']['bag_size'] if not conf['input']['input_data_format'] == 'tf_data_corresponding_pretiled' \
        else conf['input']['bag_size'] * 2
    image_bag.reshape([num_images,
                         conf['input']['num_channels'],
                         conf['input']['target_height'],
                         conf['input']['target_width']])

    diff = np.sum(([torch.max(image_bag[:,j,:,:]).tolist()-torch.min(image_bag[:,j,:,:]).tolist() for j in [0,1,2]]))
    if diff < 1e-4:
        print("Image variation is distroyed!", diff, " last")

    return image_bag


# Functions for processing
def scale_hed(hed, new_min, new_max):
    """Input hed: [?, H, W, C]"""
    #color_system_matrix = np.array([[0.65, 0.70, 0.29],
    #                                [0.07, 0.99, 0.11],
    #                                [0.27, 0.57, 0.78]])
    inv_color_system_matrix = np.array([[ 1.87798274, -1.00767869, -0.55611582],
                                        [-0.06590806,  1.13473037, -0.1355218 ],
                                        [-0.60190736, -0.48041419,  1.57358807]])

    min_values = -np.log([3, 2, 2]) @ inv_color_system_matrix
    max_values = -np.log([2, 3, 3]) @ inv_color_system_matrix

    h_min = min_values[0]
    e_min = min_values[1]
    d_min = min_values[2]
    h_max = max_values[0]
    e_max = max_values[1]
    d_max = max_values[2]

    old_h = hed[:,:,:,:]
    old_e = hed[:,:,:,1:]
    old_d = hed[:,:,:,2:]

    new_h = tools.scale_tensor(old_h, 0.0, 1.0, h_min, h_max)
    new_e = tools.scale_tensor(old_e, 0.0, 1.0, e_min, e_max)
    new_d = tools.scale_tensor(old_d, 0.0, 1.0, d_min, d_max)

    out_hed = torch.cat([new_h, new_e, new_d], dim=3)

def geometric_distortions(image_bag, resize_crop, random_crop, central_crop, distort_orientation,
                          target_height, target_width, num_channels, name=None, corresponding_tiles=False,
                          corr_equal_rot=True):
    """Apply geometric manipulations on image.

    Args:
      image_bag: Tensor of shape [bag_size, height, width, num_channels]
      bag_size, channels, height, width

    Returns:
      mod_image_bag: Tensor of shape [bag_size, height, width, num_channels]
    """

    bag_size = image_bag.size(0)    #.get_shape().as_list()[0]
    if resize_crop:
        print("Resize and random crop to original size")
        # Padding is done automatically if needed
        # Crop the image to the target size (target_height+4, target_width+4)
        transform_center_crop = transforms.CenterCrop((target_height+4, target_width+4))
        image_bag = torch.stack([transform_center_crop(img) for img in image_bag])

        # Define the random crop transformation for the given target height and width
        transform_random_crop = transforms.RandomCrop((target_height, target_width))
        # Apply
        image_bag = torch.stack([transform_random_crop(img) for img in image_bag])
        

    if random_crop:
        ## print("Random crop")
        # Define the random crop transformation for the given target height and width
        transform_random_crop = transforms.RandomCrop((target_height, target_width))
        # Apply
        image_bag = torch.stack([transform_random_crop(img) for img in image_bag])

    if central_crop:
        ## print("Central crop")
        transform_center_crop = transforms.CenterCrop((target_height, target_width))
        image_bag = torch.stack([transform_center_crop(img) for img in image_bag])

    if distort_orientation:
        ## print("Distort orientation")
        if corresponding_tiles:
            img_bag_shape = image_bag.size()
            num_bags = 1 if corr_equal_rot else 2
            image_bag = image_bag.view(num_bags, bag_size // num_bags, img_bag_shape[1], img_bag_shape[2], img_bag_shape[3])
            
            image_bag_list = torch.unbind(image_bag, dim=0)
            for i in range(len(image_bag_list)):
                # Horizontal flip
                h_flip = transforms.RandomHorizontalFlip(p=0.5)
                image_bag[i] = h_flip(image_bag[i])

                # Vertical flip
                v_flip = transforms.RandomVerticalFlip(p=0.5)
                image_bag[i] = v_flip(image_bag[i])

                # Random rotation of 0/90/180/270 degrees
                rotation = torch.randint(low=0, high=4, size=[1], dtype=torch.int32)
                image_bag[i] = torch.rot90(image_bag[i], rotation[0], dims=[-2,-1])
            image_bag = torch.stack(image_bag)
            image_bag = torch.reshape(image_bag, [bag_size, img_bag_shape[1], img_bag_shape[2], img_bag_shape[3]])
        else:
            for i in range(len(image_bag)):
                # Flip left-right and up-down
                h_flip = transforms.RandomHorizontalFlip(p=0.5)
                image_bag[i] = h_flip(image_bag[i])
                v_flip = transforms.RandomVerticalFlip(p=0.5)
                image_bag[i] = v_flip(image_bag[i])

            rotation = torch.randint(low=0, high=4, size=[1], dtype=torch.int32)
            # Rotate all tiles in the same way
            h_flip = transforms.RandomHorizontalFlip(p=0.5)
            v_flip = transforms.RandomVerticalFlip(p=0.5)
            for i in range(image_bag.size(0)):
                image_bag[i] = h_flip(image_bag[i])
                image_bag[i] = v_flip(image_bag[i])
                image_bag[i] = torch.rot90(image_bag[i], rotation[0], dims=[-2,-1])

        image_bag.reshape([bag_size, num_channels, target_height, target_width])

        return image_bag

    return image_bag

def color_distortion(image_bag,
                     input_colorspace,
                     distort_colorspace,
                     shift_interval_size,
                     scale_interval_size,
                     distort_channel_per_bag,
                     exponential_scale,
                     name=None):
    """Perform color distortion on each channel individually.

    Args:
      image_bag: float32 tensor of RGB images with values in [0, 1]
      distort_colorspace: {'rgb', 'hsv', 'he'} in what colorspace to perform the distortion
      shift_interval_size: A list of four numbers, representing the shift interval size in
            [channel 1, channel 2, channel 3, contrast]
      scale_interval_size: A list of four numbers, representing the scale interval size in
            [channel 1, channel 2, channel 3, contrast]
      distort_channel_per_bag: boolean indicating if color distortion should be be applied the same for the bag
      exponential_scale: If True, will randomly select scaling_factor from [10**(-scale_interval_size), 10**(scale_interval_size)]
    Returns:
      image_bag: float32 tensor. The altered image bag, where each image is in the RGB space
    """

    ## print("Distortion shift intervals: {}".format(shift_interval_size))
    ## print("Distortion scale intervals: {}".format(scale_interval_size))
    ## print("Perform color distortion per bag: {}".format(distort_channel_per_bag))

    # Randomize contrast and hsv distortion
    #rand_0_1 = np.random.randint(0, 2)
    # TODO Contrast adjustment is transforms the batch from RGB to float representation.
    # This is redundant. The actual operation is, for each channel:
    #
    #  x <- (x - mean) * contrast_factor + mean
    # TODO: Configurations
    if distort_colorspace == 'hsv':
        image_bag = distort_hsv(image_bag,
                                input_colorspace,
                                shift_interval_size,
                                scale_interval_size,
                                distort_channel_per_bag,
                                exponential_scale)

        diff = np.sum(([torch.max(image_bag[:,j,:,:]).tolist()-torch.min(image_bag[:,j,:,:]).tolist() for j in [0,1,2]]))
        if diff < 1e-4:
            print("Image variation is distroyed!", diff, " afterHSVDistortion")

    # elif distort_colorspace == 'he':
    #     image_bag = distort_he(image_bag,
    #                             input_colorspace,
    #                             shift_interval_size,
    #                             scale_interval_size,
    #                             distort_channel_per_bag)
    # elif distort_colorspace == 'hed':
    #     image_bag = distort_hed(image_bag,
    #                             input_colorspace,
    #                             shift_interval_size,
    #                             scale_interval_size,
    #                             distort_channel_per_bag)
    else:
        print('Color distortion in colorspace {} is not implemented'.format(distort_colorspace))
    
    if shift_interval_size[3] > 0 or scale_interval_size[3] > 0:
        ## print("Distort contrast")
        if distort_channel_per_bag:
            # TODO: random_contrast needs lower, upper (not shift scale).
            contrast_factor = (scale_interval_size[3]-shift_interval_size[3]) * torch.rand(size=[1]).item() + shift_interval_size[3]
            if exponential_scale:
                contrast_factor = torch.pow(10.0, contrast_factor)
            # Adjust contrast independently on all channels
            means = image_bag.mean(dim=(-2,-1), keepdim=True)
            image_bag = (image_bag-means) * contrast_factor + means

        else:
            if not exponential_scale:
                # Random contrast for individual tiles
                low=shift_interval_size[3]
                high=scale_interval_size[3]
                for i in range(len(image_bag)):
                    contrast_factor = (high-low) * torch.rand(1).item() + low
                    means = image_bag[i].mean(dim=(-2,-1), keep_dim=True)
                    image_bag[i] = (image_bag-means) * contrast_factor + means
                # image_bag = tf.map_fn(lambda tile: tf.image.random_contrast(tile,
                #                                                         shift_interval_size[3],
                #                                                         scale_interval_size[3]),
                #                     image_bag)
            else:
                ## print('Contrast distortion not implemented for exponential_scale and not per_bag augmentation')
                raise NotImplementedError('Contrast distortion not implemented for exponential_scale and not per_bag augmentation')
        image_bag = torch.clip(image_bag, min=0.0, max=1.0)

    return image_bag


def random_gaussian_blur(image_bag,
                         kernel_size,
                         sigma):
    """Perform gaussian blur on each channel individually.

    https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing/random_gaussian_blur.py

    Args:
      image_bag: float32 tensor of RGB images with values in [0, 1], shape [bag_size, num_channels, height, width]
      kernel_size: Size of filter, constant
      sigma: Filter sigma is randomly sampled from [0, sigma)
    Returns:
      image_bag: float32 tensor. The altered image bag, where each image is in the RGB space
    """
    sigma = (sigma-0)*torch.rand(1).item() + 0

    for i in range(len(image_bag)):
        img = image_bag[i]
        channels = [torch.select(img, 0, 0), torch.select(img, 0, 1), torch.select(img, 0, 2)]
        blur_transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        blurred_channels = []
        for channel in channels:
            blurred_channel = blur_transform(channel.unsqueeze(0)).squeeze()
            blurred_channels.append(blurred_channel)

        blur_img = torch.stack(blurred_channels, dim=0)
        ## print("Blur image has shape", blur_img.size(), flush=True)

        image_bag[i] = blur_img

    return image_bag

def random_gaussian_noise(image_bag, variance_max):
    """Add random Gaussian noise to the value channel

    Args:
      image_bag: float32 tensor of RGB images with values in [0, 1], shape [bag_size, num_channels, height, width]
                 Can also be a single image with shape [num_channels, height, width]
      variance_max: Noise is generated from normal distribution with mean=0 and variance randomly selected from [0, variance_max)
    Returns:
      image_bag: float32 tensor. The altered image bag, where each image is in the RGB space
    """
    if len(image_bag.size()) < 4:
        image_bag = image_bag.unsqueeze(0)

    hsv_bag = rgb2hsv(torch.swapaxes(image_bag, 1, -1))
    hsv_bag = torch.from_numpy(hsv_bag)
    hue_bag, saturation_bag, value_bag = hsv_bag[:, :, :, 0], hsv_bag[:, :, :, 1], hsv_bag[:, :, :, 2]

    variance = variance_max * torch.rand(1).item()
    std = torch.sqrt(variance)
    noise = torch.empty(1).normal_(mean=0, std=0.0036).item() #torch.normal(mean=0.0, std=std)
    # print("Adding noise", noise, "From std", std, "prev max", torch.max(value_bag))
    noisy_value_bag = torch.add(value_bag, noise)
    if torch.max(noisy_value_bag).item() < 0:
        noisy_value_bag = torch.add(noisy_value_bag, -torch.min(noisy_value_bag).item()*1.01)
        print("Having to shift the random gaussian noise on value so it's not 0 everywhere")
    noisy_value_bag = torch.clip(noisy_value_bag, min=0.0, max=1.0)
    
    noisy_hsv = torch.stack([hue_bag, saturation_bag, noisy_value_bag], -1)
    ## print("Hsv image in random_gaussian_noise has size", noisy_hsv.size(), flush=True)
    image_bag = hsv2rgb(noisy_hsv)
    image_bag = torch.swapaxes(torch.from_numpy(image_bag), 1, -1)
    ## print("Image size after return to rgb", image_bag.size(), flush=True)
    
    diff = np.sum(([torch.max(image_bag[:,j,:,:]).tolist()-torch.min(image_bag[:,j,:,:]).tolist() for j in [0,1,2]]))
    if diff < 1e-4:
        print("Image variation is distroyed!", diff, " R-G-Noise")

    return image_bag

def distort_hsv(image_bag, input_colorspace, shift_interval_size, scale_interval_size, 
                distort_channel_per_bag, exponential_scale):
    """Randomize channels in hsv.

    Input image is RGB in [0, 1] with shape [bag_size, 3, height, width]
    """
    ## print("Distort colors in hsv space")
    #print(f"Diff at input {torch.max(image_bag)-torch.min(image_bag)}, max is {torch.max(image_bag)}")
    if len(image_bag.size()) < 4:
        image_bag = image_bag.unsqueeze(0)

    hsv_bag = rgb2hsv(torch.swapaxes(image_bag, 1, -1))
    hsv_bag = torch.from_numpy(hsv_bag)
    hue_bag, saturation_bag, value_bag = hsv_bag[:, :, :, 0], hsv_bag[:, :, :, 1], hsv_bag[:, :, :, 2]

    hue_bag = distort_single_channel(hue_bag,
                                     shift_delta=shift_interval_size[0],
                                     scale_delta=scale_interval_size[0],
                                     distort_channel_per_bag=distort_channel_per_bag,
                                     exponential_scale=exponential_scale,
                                     name="hue")
    saturation_bag = distort_single_channel(saturation_bag,
                                            shift_delta=shift_interval_size[1],
                                            scale_delta=scale_interval_size[1],
                                            distort_channel_per_bag=distort_channel_per_bag,
                                            exponential_scale=exponential_scale,
                                            name="saturation")
    value_bag = distort_single_channel(value_bag,
                                       shift_delta=shift_interval_size[2],
                                       scale_delta=scale_interval_size[2],
                                       distort_channel_per_bag=distort_channel_per_bag,
                                       exponential_scale=exponential_scale,
                                       name="value")

    hue_bag = torch.remainder(hue_bag, 1.0)
    #print("hm", hue_bag.mean(), hue_bag.size())
    hsv_bag = torch.stack([hue_bag, saturation_bag, value_bag], -1)
    hsv_bag = torch.clamp(hsv_bag, min=0., max=1.)
    rgb_bag = hsv2rgb(hsv_bag)

    if input_colorspace == 'hed':
        print("hed colorspace not implemented")
    #     hed_bag = rgb_to_hed(image_bag)
    #     hed_bag = scale_hed(hed_bag, 0.0, 1.0)
    #     return hed_bag
    if input_colorspace == 'rgb':
        return torch.swapaxes(torch.from_numpy(rgb_bag), 1, -1)

    return torch.swapaxes(torch.from_numpy(rgb_bag), 1, -1)

def distort_single_channel(single_channel_bag, shift_delta=-1.0, scale_delta=-1.0, distort_channel_per_bag=False,
                           exponential_scale=False, name=None):
    """Distorts a tensor of a single channel

    Note that this distorts every tile independently.
    """

    if distort_channel_per_bag:
        shape = [1]
    else:
        bag_size = single_channel_bag.size(0)
        shape = [bag_size, 1, 1, 1]

    if shift_delta > 0:
        # Random (uniform) between -shift_delta, shift_delta
        shift_deltas = 2*shift_delta * torch.rand(shape) + shift_delta

    if scale_delta > 0:
        if exponential_scale:
            base = torch.ones(shape)*10
            #base = tf.constant(10.0, shape=shape)
            exponentials = 2*scale_delta * torch.rand(shape) + scale_delta
            scale_deltas = torch.pow(base, exponentials)
        else:
            scale_deltas = (1+scale_delta - 1/(1+scale_delta)) * torch.rand(shape) + 1/(1+scale_delta)

    single_channel_bag = single_channel_bag.type(torch.FloatTensor)

    # if name=="hue":
    #     import random
    #     intervals = [[0, 0.1], [0.6, 0.99]]
    #     r_mean = random.uniform(*random.choices(intervals,weights=[r[1]-r[0] for r in intervals])[0])
    #     curr_mean = single_channel_bag.mean()
    #     return single_channel_bag.add(-curr_mean+r_mean)

    if scale_delta > 0 and shift_delta > 0:
        return single_channel_bag * scale_deltas + shift_deltas
    elif scale_delta > 0 and shift_delta <= 0:
        return single_channel_bag * scale_deltas
    elif scale_delta <= 0 and shift_delta > 0:
        return single_channel_bag + shift_deltas
    else:
        return single_channel_bag


if __name__ == "__main__":
    import toml
    with open('j_config.toml', 'r') as f:
        conf = toml.load(f)
    aug_p = conf['input']