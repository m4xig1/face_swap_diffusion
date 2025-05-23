from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import json
import numpy as np
from PIL import Image
import torch
import torch.utils
from torch.utils.data import Dataset, BatchSampler, Sampler
import torch.utils.data
import torchvision
import torchvision.transforms as T
from torchvision.transforms import ToTensor
import albumentations as A
from typing import Optional, Union, Tuple
import random
import cv2
from einops import rearrange


def un_norm_clip(x1):
    x = x1.clone()
    reduce = False
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
        reduce = True
    x[:, 0, :, :] = x[:, 0, :, :] * 0.26862954 + 0.48145466
    x[:, 1, :, :] = x[:, 1, :, :] * 0.26130258 + 0.4578275
    x[:, 2, :, :] = x[:, 2, :, :] * 0.27577711 + 0.40821073

    if reduce:
        x = x.squeeze(0)
    return x


def un_norm(x):
    return (x + 1.0) / 2.0


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [
            torchvision.transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            )
        ]
    return torchvision.transforms.Compose(transform_list)


# Utility function to create a decoy mask (similar to the one in celebA.py)
def decow(mask, scale=0.5):
    """
    Create a decoy mask by scaling the original mask.
    Args:
        mask (torch.Tensor): Original mask
        scale (float): Scale factor
    Returns:
        torch.Tensor: Scaled mask
    """
    mask = mask.clone()
    mask_np = mask.cpu().numpy()
    mask_np = mask_np.transpose(0, 2, 3, 1)
    mask_np = mask_np.squeeze(-1)

    for i in range(mask_np.shape[0]):
        mask_i = mask_np[i]
        mask_i = (mask_i * 255).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Get bounding rect
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Scale the bounding rect
            center_x, center_y = x + w // 2, y + h // 2
            new_w, new_h = int(w * scale), int(h * scale)
            new_x, new_y = center_x - new_w // 2, center_y - new_h // 2

            # Create new mask
            new_mask = np.zeros_like(mask_i)
            new_mask[new_y : new_y + new_h, new_x : new_x + new_w] = 255
            mask_np[i] = new_mask / 255.0

    mask_np = np.expand_dims(mask_np, -1)
    mask_np = mask_np.transpose(0, 3, 1, 2)
    return torch.from_numpy(mask_np).to(mask.device)

def get_bigger_crop(img, crop, scale=0.2):
    # to square crop 
    if crop[3] - crop[1] < crop[2] - crop[0]:
        diff = crop[2] - crop[0] - (crop[3] - crop[1])
        if diff % 2 != 0:
            crop[0] -= 1
            diff += 1
        crop[3] += diff // 2
        crop[1] -= diff // 2
    elif crop[2] - crop[0] < crop[3] - crop[1]:
        diff = crop[3] - crop[1] - (crop[2] - crop[0])
        if diff % 2 != 0:
            crop[1] -= 1
            diff += 1
        crop[2] += diff // 2
        crop[0] -= diff // 2
    assert crop[3] - crop[1] == crop[2] - crop[0], crop

    # upscale crop
    to_add = int((crop[3] - crop[1]) * scale)
    h, w, _ = np.array(img).shape
    crop = [max(0, crop[0] - to_add), max(0, crop[1] - to_add), min(w, crop[2] + to_add), min(h, crop[3] + to_add)]
    cropped_arr = np.array(img)[crop[1]:crop[3], crop[0]:crop[2]]
    return Image.fromarray(cropped_arr)


class WildFacesDataset(Dataset):
    def __init__(self, state, arbitrary_mask_percent=0, load_vis_img=False, label_transform=None, fraction=1.0, **args):
        """
        Initialize the dataset.

        Args:
            state (str): Dataset state ('train', 'validation', 'test').
            arbitrary_mask_percent (float): Percentage for arbitrary mask (not used here).
            load_vis_img (bool): Whether to load visualization images (not used here).
            label_transform (callable, optional): Transformation for labels (not used here).
            fraction (float): Fraction of data to use.
            **args: Additional arguments including 'dataset_dir', 'image_size', etc.
        """
        self.state = state
        self.args = args
        self.fraction = fraction
        self.arbitrary_mask_percent = arbitrary_mask_percent
        self.load_vis_img = load_vis_img
        self.label_transform = label_transform
        self.gray_outer_mask = args.get("gray_outer_mask", True)
        self.Fullmask = False
        self.n_src_imgs = args.get("n_src_imgs", 1)  # Number of reference images to sample
        self.img_size = args["image_size"]

        self.clip_img_size = (224, 224)
        if args.get("json_path"):
            json_path = args["json_path"]
        else:
            json_path = osp.join(args["dataset_dir"], "all_gathered.json")

        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.samples = []
        self.ids2samples = {}
        for id_key in self.data:
            for img_num in self.data[id_key]:
                image_path = osp.join(args["dataset_dir"], f"{id_key}/{img_num}.jpg")
                bbox = self.data[id_key][img_num]["new_face_crop"]
                self.samples.append((id_key, img_num, image_path, bbox))

        # Split samples by ids based on state
        ids = list(self.data.keys())
        total_samples = len(ids)
        if state == "train":
            ids = ids[: int(0.8 * total_samples)]
            # self.samples = self.samples[: int(0.8 * total_samples)]
        elif state == "validation":
            ids = ids[int(0.8 * total_samples) : int(0.9 * total_samples)]
            # self.samples = self.samples[int(0.8 * total_samples) : int(0.9 * total_samples)]
        else:  # 'test'
            ids = ids[int(0.9 * total_samples) :]
            # self.samples = self.samples[int(0.9 * total_samples) :]
        ids = ids[: int(len(ids) * fraction) + 1]
        ids = set(ids)
        self.samples = [sample for sample in self.samples if sample[0] in ids]

        # Apply fraction to limit dataset size
        # self.samples = self.samples[: int(len(self.samples) * fraction)]

        # Build ID to samples mapping for efficient reference image sampling
        self.ids2samples = {}
        for img_num, sample in enumerate(self.samples):
            id_key = sample[0]
            if id_key in self.ids2samples:
                self.ids2samples[id_key].append(img_num)
            else:
                self.ids2samples[id_key] = [img_num]

        # Define transformations for reference image (similar to CelebAdataset)

        # 224 is clip input size
        self.random_trans = T.Compose([T.ToTensor(), T.Resize((224, 224))])

        # Check if mask directory is provided for saving masks
        self.mask_dir = args.get("mask_dir", None)
        if self.mask_dir and not osp.exists(self.mask_dir):
            os.makedirs(self.mask_dir)

        # Image pairs indices
        self.indices = np.arange(len(self.samples))
        self.length = len(self.indices)

        # np.random.shuffle(self.samples)

    @staticmethod
    def interpolate_bbox(bbox, img_size, target_size: Union[int, tuple] = (512, 512)):
        """
        bbox: [x_min, y_min, x_max, y_max]
        img_size: [H, W]
        returns: new bbox interpolated to target_size (note: resize image to target_size after crop)
        """
        H, W = img_size
        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        h_min, w_min, h_max, w_max = bbox
        h_center = (h_max + h_min) // 2
        w_center = (w_max + w_min) // 2
        
        # Ensure center is within image bounds
        h_center = max(0, min(h_center, W))
        w_center = max(0, min(w_center, H))

        h_half = target_size[1] // 2
        w_half = target_size[0] // 2
        
        # Calculate new bounds with proper clamping
        h_min_new = max(0, h_center - h_half)
        h_max_new = min(H, h_center + h_half)
        w_min_new = max(0, w_center - w_half)
        w_max_new = min(W, w_center + w_half)
        
        # Adjust if the box is too small on one side (maintain target size when possible)
        if h_max_new - h_min_new < target_size[1] and h_max_new < H:
            h_max_new = min(H, h_min_new + target_size[1])
        if h_max_new - h_min_new < target_size[1] and h_min_new > 0:
            h_min_new = max(0, h_max_new - target_size[1])
            
        if w_max_new - w_min_new < target_size[0] and w_max_new < W:
            w_max_new = min(W, w_min_new + target_size[0])
        if w_max_new - w_min_new < target_size[0] and w_min_new > 0:
            w_min_new = max(0, w_max_new - target_size[0])
        return (h_min_new, w_min_new, h_max_new, w_max_new)
    

    def _preprocess_dataset_img(self, img: Image.Image, bbox: Tuple[int]) -> Tuple[Image.Image, Tuple[int]]:
        """
        gets: img and bbox from dataset
        returns: resized image and new face bbox points
        """
        new_bbox = bbox.copy() if hasattr(bbox, 'copy') else list(bbox)
        
        face_region_bbox = self.interpolate_bbox(new_bbox, img.size, self.img_size)
        
        face_region_bbox = [max(0, int(coord)) for coord in face_region_bbox]
        
        # Ensure the box has non-zero dimensions
        if face_region_bbox[2] <= face_region_bbox[0]:
            face_region_bbox[2] = face_region_bbox[0] + 1
        if face_region_bbox[3] <= face_region_bbox[1]:
            face_region_bbox[3] = face_region_bbox[1] + 1
            
        # Crop the face region
        face_region_img = img.crop(face_region_bbox)
        
        # Calculate relative coordinates
        new_bbox[0] = max(0, new_bbox[0] - face_region_bbox[0])
        new_bbox[1] = max(0, new_bbox[1] - face_region_bbox[1])
        new_bbox[2] = max(0, new_bbox[2] - face_region_bbox[0])
        new_bbox[3] = max(0, new_bbox[3] - face_region_bbox[1])
        
        # Get original dimensions before resize
        orig_width, orig_height = face_region_img.size
        
        # Prevent division by zero
        scale_w = self.img_size / max(1, orig_width)
        scale_h = self.img_size / max(1, orig_height)
        
        # Scale the bbox coordinates
        new_bbox[0] = int(new_bbox[0] * scale_w)
        new_bbox[1] = int(new_bbox[1] * scale_h)
        new_bbox[2] = int(new_bbox[2] * scale_w)
        new_bbox[3] = int(new_bbox[3] * scale_h)
        
        # Ensure bbox is within image bounds after scaling
        new_bbox[0] = max(0, min(new_bbox[0], self.img_size - 1))
        new_bbox[1] = max(0, min(new_bbox[1], self.img_size - 1))
        new_bbox[2] = max(new_bbox[0] + 1, min(new_bbox[2], self.img_size))
        new_bbox[3] = max(new_bbox[1] + 1, min(new_bbox[3], self.img_size))
        
        # Resize the image
        face_region_img = face_region_img.resize((self.img_size, self.img_size))
        
        return face_region_img, new_bbox

    def create_mask_from_bbox(self, widened_bbox, image_size):
        """
        Create a binary mask from a widened bounding box.

        Args:
            widened_bbox (list): [x_min, y_min, x_max, y_max]
            image_size (list): [height, width]

        Returns:
            np.ndarray: Binary mask (1 inside bbox, 0 outside)
        """
        image_height, image_width = image_size
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        x_min, y_min, x_max, y_max = map(int, widened_bbox)
        mask[y_min:y_max+1, x_min:x_max+1] = 1
        return mask

    def __getitem__(self, index):
        """
        Get a single item from the dataset with multiple reference images.

        Args:
            index (int): Index of the sample

        Returns:
            dict: Contains 'GT', 'inpaint_image', 'inpaint_mask', and 'ref_imgs'
        """
        id_key, img_num, image_path, bbox = self.samples[index]
        img_p = Image.open(image_path).convert("RGB")
        orig_image_size = self.data[id_key][img_num]["orig_image_size"]  # [height, width]

        # get face region
        img_p, bbox_p = self._preprocess_dataset_img(img_p, bbox)
        # Create or load mask
        mask_path = osp.join(self.mask_dir, f"{id_key}/{img_num}.png") if self.mask_dir else None

        if mask_path and osp.exists(mask_path):
            # use precomputed masks
            mask_img = Image.open(mask_path).convert("L")
        else:
            mask = self.create_mask_from_bbox(bbox_p, (self.img_size, self.img_size))
            mask_img = Image.fromarray(mask * 255).convert("L")

        # Save full mask for compatibility with CelebA dataset
        if self.Fullmask:
            mask_img_full = mask_img
            mask_img_full = get_tensor(normalize=False, toTensor=True)(mask_img_full)

        # Convert mask to tensor (1 for hole, 0 for known)
        mask_tensor = 1 - get_tensor(normalize=False, toTensor=True)(mask_img)
        # Sample reference images from the same ID
        ref_images_tensors = []
        if id_key in self.ids2samples and len(self.ids2samples[id_key]) > 1:
            # Get indices of all samples with the same ID, excluding current one
            same_id_indices = [i for i in self.ids2samples[id_key] if i != index]
            if self.n_src_imgs > len(same_id_indices):
                print(f"WARN: too low imgs with id {id_key}, while sampling {self.n_src_imgs} src's")
                ref_indices = random.choices(same_id_indices, k=self.n_src_imgs)
            else:
                ref_indices = random.sample(same_id_indices, k=self.n_src_imgs)
            
            # ref_img = cropped gt
            # ref_indices = [index] # TODO: убрать после теста

            # Process each reference image
            for ref_idx in ref_indices:
                _, _, ref_path, bbox = self.samples[ref_idx]

                ref_img = Image.open(ref_path).convert("RGB")
                # TODO: add transforms if needed
                # ref_img = ref_img.crop(bbox).resize((224, 224))
                ref_img = get_bigger_crop(ref_img, bbox, scale=0.05).resize((224, 224))
                ref_img_tensor = get_tensor_clip()(ref_img)
                ref_images_tensors.append(ref_img_tensor)
        else:
            raise ValueError("No reference images available for the given ID.")

        ref_images_tensor = torch.stack(ref_images_tensors)

        image_tensor = get_tensor()(img_p)

        # Apply random scaling to mask (similar to decow in CelebA)
        # TODO: don't use decow yet
        # scale = random.uniform(0.5, 1.0)
        # mask_tensor_resize = decow(mask_tensor_resize.unsqueeze(0), scale=scale).squeeze(0)

        # Create inpainted image (masked)
        inpaint_tensor = image_tensor * mask_tensor

        # Return in format compatible with CelebA dataset
        result = {
            "GT": image_tensor,
            "inpaint_image": inpaint_tensor,
            "inpaint_mask": mask_tensor,
            "ref_imgs": ref_images_tensor,
        }

        if self.Fullmask:
            result["inpaint_mask"] = mask_img_full

        return result

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.samples)


if __name__ == "__main__":
    random.seed(0)

    dataset = WildFacesDataset(
        "train",
        n_src_imgs=1,
        dataset_dir="/home/aalanov/Bobkov_Denis/datasets/my_original/final",
        json_path="/home/mdnikolaev/maignatov_2/face_swap_diffusion/data/filtered_ids3.json",
        image_size=512,
        fraction=0.0,
    )
    # print(len(dataset))
    # print(dataset.ids2samples)
    # print(dataset.samples)
    sample = dataset[1] # 13 - bad sample

    def save_clip_img(img, path, clip=False):
        if clip:
            img = un_norm_clip(img)
        else:
            img = torch.clamp(un_norm(img), min=0.0, max=1.0)
        img = img.cpu().numpy().transpose((1, 2, 0))
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(path)

    save_clip_img(sample["inpaint_image"], "./test_inpaint.jpeg")
    # save_clip_img(sample["inpaint_mask"], "./test_inpaint_mask.jpeg")
    save_clip_img(sample["GT"], "./test_gt.jpeg")
    # save_clip_img(sample["ref_imgs"][0], "./test_ref0.jpeg")
    # save_clip_img(sample["ref_imgs"][1], "./test_ref1.jpeg")
