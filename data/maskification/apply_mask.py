'''
This file contains the code to apply the masks and store it in another folder

'''
import os

from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm

'''
NOTE: the Cam dataset needs mask only for the tumour part and for WBC you do have proper masks for everything.

Deleted the masks in the cam16/data/mask/normal as it was all black anyway

'''
class Maskify:
    def __init__(self, data_path, mask_path, masked_data_path):
        self.data_path = data_path
        self.mask_path = mask_path
        self.masked_data_path = masked_data_path

        # Create the output directory if it doesn't exist
        os.makedirs(masked_data_path, exist_ok=True)

        # transforms
        self.img_to_tensor_transform = transforms.ToTensor()
        self.tensor_to_img_transform = transforms.ToPILImage()

        # masked dataloader
        self.mask_dataset = ImageFolder(root=self.mask_path, transform=self.img_to_tensor_transform)

        # apply masks
        self.apply_masks()

    def apply_masks(self):
        # Iterate through the mask images and apply them to the corresponding data images
        for mask_file_path, _ in tqdm(self.mask_dataset.samples):
            mask_file_name = os.path.basename(mask_file_path)

            # Get the parent folder name
            mask_file_class = os.path.basename(os.path.dirname(mask_file_path))

            # Find the corresponding data image using the same filename
            data_file_path = os.path.join(self.data_path, mask_file_class, mask_file_name)

            #Find the corresponding masked data image path to which we will be saving the image to
            masked_data_file_path = os.path.join(self.masked_data_path, mask_file_class, f"masked_{mask_file_name}")

            # Check if the corresponding data image exists
            if os.path.exists(data_file_path):
                # Load the data image
                mask_image = Image.open(mask_file_path)
                data_image = Image.open(data_file_path)

                # Convert tensors back to PIL images
                mask_image_tensor = self.img_to_tensor_transform(mask_image)
                data_image_tensor = self.img_to_tensor_transform(data_image)

                # Apply the mask to the data image
                masked_data_image_tensor = mask_image_tensor * data_image_tensor
                masked_data_image = self.tensor_to_img_transform(masked_data_image_tensor)

                # Save the masked image to the output directory
                os.makedirs(os.path.dirname(masked_data_file_path), exist_ok=True)
                masked_data_image.save(f"{masked_data_file_path}")
            else:
                print(f"{data_file_path} not found")
        print("Finished applying masks and saving!")


if __name__ == '__main__':
    # for cam 16 dataset
    cam_data_path = os.path.abspath("../../datasets/CAM16_100cls_10mask/train/data")
    cam_mask_path = os.path.abspath("../../datasets/CAM16_100cls_10mask/train/mask")
    cam_masked_data_path = os.path.abspath("../../datasets/CAM16_100cls_10mask/train/masked_data")
    cam_maskify = Maskify(cam_data_path, cam_mask_path, cam_masked_data_path)
    print("Camelyon done!")

    # for wbc1
    wbc1_data_path = os.path.abspath("../../datasets/WBC_1/train/data")
    wbc1_mask_path = os.path.abspath("../../datasets/WBC_1/train/mask")
    wbc1_masked_data_path = os.path.abspath("../../datasets/WBC_1/train/masked_data")
    wbc1_maskify = Maskify(wbc1_data_path, wbc1_mask_path, wbc1_masked_data_path)
    print("WBC1 done!")

    # for wbc10
    wbc10_data_path = os.path.abspath("../../datasets/WBC_10/train/data")
    wbc10_mask_path = os.path.abspath("../../datasets/WBC_10/train/mask")
    wbc10_masked_data_path = os.path.abspath("../../datasets/WBC_10/train/masked_data")
    wbc10_maskify = Maskify(wbc10_data_path, wbc10_mask_path, wbc10_masked_data_path)
    print("WBC10 done!")

    # for wbc50
    wbc50_data_path = os.path.abspath("../../datasets/WBC_50/train/data")
    wbc50_mask_path = os.path.abspath("../../datasets/WBC_50/train/mask")
    wbc50_masked_data_path = os.path.abspath("../../datasets/WBC_50/train/masked_data")
    wbc50_maskify = Maskify(wbc50_data_path, wbc50_mask_path, wbc50_masked_data_path)
    print("WBC50 done!")

    # for wbc100
    wbc100_data_path = os.path.abspath("../../datasets/WBC_100/train/data")
    wbc100_mask_path = os.path.abspath("../../datasets/WBC_100/train/mask")
    wbc100_masked_data_path = os.path.abspath("../../datasets/WBC_100/train/masked_data")
    wbc100_maskify = Maskify(wbc100_data_path, wbc100_mask_path, wbc100_masked_data_path)
    print("WBC100 done!")

