import torch
from torchvision import transforms
from transformers import AutoImageProcessor
from PIL import Image


def preprocess(image_bag, local_crops_nr=8, global_crops_nr=2, org=False, aug_strength="weak", dinotype=2):
        aug_strength="heavy" # {"weak" // "strong" //"heavy" // "max"}
        dinotype=1
        if aug_strength=="strong" or aug_strength=="heavy" or aug_strength=="max":
            global_scale=(0.32, 1.)
            local_scale=(0.05, 0.32)
        elif aug_strength=="weak":
            global_scale = (0.9, 1)
            local_scale = (0.7, 0.9)
        else:
            print("You must choose augmentation strength weak, strong, or heavy, not", aug_strength)
            import sys
            sys.exit()     

        # Phikon, Phikon2, Virchow, Virchow2 all normalize their input in this way.
        # Check for any new model. We apply this at the end, as it replaces the feate_extractor.transform that would notmally be used.
        normalizer = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        norm = normalizer
        #norm = transforms.Identity()

        totensor = transforms.PILToTensor()
        flips = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)])
        
        if aug_strength=="weak":
            blur = transforms.GaussianBlur(kernel_size=5, sigma=(1e-4, 0.4))
            jitter = transforms.ColorJitter(brightness=0.1, contrast=(0.7,1.3), saturation=(0.5,1.3), hue=(-0.5, 0.5))
        elif aug_strength=="strong":
            blur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5)
            jitter = transforms.Compose(
                [
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=(-0.5, 0.5))],
                        p=0.8,
                    ),
                    transforms.RandomGrayscale(p=0.2),
                ]
            )
        elif aug_strength=="heavy":
            blur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.6)
            jitter = transforms.Compose(
                [
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=(-0.5, 0.5))],
                        p=0.9,
                    ),
                    transforms.RandomGrayscale(p=0.3),
                ]
            )
        elif aug_strength=="max":
            blur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.8)
            jitter = transforms.Compose(
                [
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=(-0.5, 0.5))],
                        p=0.9,
                    ),
                    transforms.RandomGrayscale(p=0.3),
                ]
            )

        if aug_strength=="weak":
            # Don't stretch height/width differently -> ratio aspect = 1.
            global_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, global_scale, ratio=(1.0,1.0)),
                flips, 
                jitter,
                blur
            ])
            local_transform = transforms.Compose([
                transforms.RandomResizedCrop(98, local_scale, ratio=(1.0,1.0)), #Used 98 for dinov2 # Used 224 previously, dinoV2 uses 96. For Virchow2, it must be divisible by 14, so use 98. Try a scale of 0.75 of global->168
                flips,
                jitter,
                blur
            ])

        elif aug_strength=="strong" or aug_strength=="heavy" or aug_strength=="max":
            global_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, global_scale, ratio=(0.9,1.1)), # ratio=(1.0,1.0)
                flips, 
                jitter,
                blur
            ])
            local_transform = transforms.Compose([
                transforms.RandomResizedCrop(98, local_scale, ratio=(0.9,1.1)), # ratio=(1.0,1.0) # Used 224 previously, dinoV2 uses 96. For Virchow2, it must be divisible by 14, so use 98. Try a scale of 0.75 of global->168
                flips,
                jitter,
                blur
            ])
        
        #views = [image_bag]
        views = []
        local_views = []
        for i in range(global_crops_nr):
            views.append(global_transform(image_bag))
        for i in range(local_crops_nr):    # Different aug for each local aug
            local_views.append(local_transform(image_bag))
        # Returns 2+local_crops_nr different views w augmentations of the input image_bag
        if type(views[-1])==Image.Image:
            views = [totensor(v) for v in views]
            local_views = [totensor(v) for v in local_views]
        #print("Max before norm", views[0].max(), "min", views[0].min())
        views = [norm(view.type(torch.FloatTensor)) for view in views]
        local_views = [norm(view.type(torch.FloatTensor)) for view in local_views]
        #print("Max after norm", views[0].max(), "min", views[0].min())
        if org:
            views.append(norm(image_bag.type(torch.FloatTensor)/255))#.to(torch.float32)))

        if dinotype == 1:
            for lv in local_views:
                views.append(lv)
            return views
        #print("In augmentation func. Views as len", len(views))
        return [views, local_views]
