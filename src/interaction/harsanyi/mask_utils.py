import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Union
from .set_utils import flatten, generate_all_masks
from tqdm import tqdm


def get_mask_input_function_image(grid_width: int) -> Callable:
    """
    Return the functional to mask an input image
    :param grid_width:
    :return:
    """

    def generate_masked_input(image: torch.Tensor, baseline: torch.Tensor, grid_indices_list: List): # todo: 感觉这里还是慢了，传入S_list就需要遍历，要一个for循环，不如直接传入mask
        device = image.device
        _, image_channel, image_height, image_width = image.shape
        grid_num_h = int(np.ceil((image_height * 1.0) / grid_width))
        grid_num_w = int(np.ceil((image_width * 1.0) / grid_width))
        grid_num = grid_num_h * grid_num_w

        batch_size = len(grid_indices_list)
        masks = torch.zeros(batch_size, image_channel, grid_num)
        for i in range(batch_size):
            grid_indices = flatten(grid_indices_list[i])
            masks[i, :, list(grid_indices)] = 1

        masks = masks.view(masks.shape[0], image_channel, grid_num_h, grid_num_w)
        masks = F.interpolate(
            masks.clone(),
            size=[grid_width * grid_num_h, grid_width * grid_num_w],
            mode="nearest"
        ).float()
        masks = masks[:, :, :image_height, :image_width].to(device)

        expanded_image = image.expand(batch_size, image_channel, image_height, image_width).clone()
        expanded_baseline = baseline.expand(batch_size, image_channel, image_height, image_width).clone()
        masked_image = expanded_image * masks + expanded_baseline * (1 - masks)

        return masked_image

    return generate_masked_input


def get_mask_input_function_binary() -> Callable:
    """
    Return the functional to mask an input image
    :param grid_width:
    :return:
    """

    def generate_masked_input(binary: torch.Tensor, baseline: torch.Tensor, grid_indices_list: List): # todo: 感觉这里还是慢了，传入S_list就需要遍历，要一个for循环，不如直接传入mask
        device = binary.device
        bs, length = binary.shape

        batch_size = len(grid_indices_list)

        masks = torch.zeros(batch_size, length)
        for i in range(batch_size):
            grid_indices = flatten(grid_indices_list[i])
            masks[i, list(grid_indices)] = 1

        masks = masks.to(device)
        expanded_binary = binary.expand(batch_size, length).clone()
        expanded_baseline = baseline.expand(batch_size, length).clone()
        masked_binary = expanded_binary * masks + expanded_baseline * (1 - masks)

        return masked_binary

    return generate_masked_input


def get_mask_input_function_pointcloud(**kwargs) -> Callable:
    """
    Return the functional to mask an input point cloud
    :return:
    """

    def generate_masked_input(point_cloud: torch.Tensor, baseline: torch.Tensor, point_indices_list: List):  # todo: 感觉这里还是慢了，传入S_list就需要遍历，要一个for循环，不如直接传入mask
        device = point_cloud.device
        input_bs, c, n_points = point_cloud.shape
        assert input_bs == 1

        batch_size = len(point_indices_list)
        masks = torch.zeros(batch_size, c, n_points).to(device)
        for i in range(batch_size):
            point_indices = flatten(point_indices_list[i])
            masks[i, :, point_indices] = 1

        expanded_pc = point_cloud.expand(batch_size, c, n_points).clone()
        expanded_baseline = baseline.expand(batch_size, c, n_points).clone()
        masked_image = expanded_pc * masks + expanded_baseline * (1 - masks)

        return masked_image

    return generate_masked_input


def get_mask_input_function_nlp() -> Callable[[torch.LongTensor, int, torch.BoolTensor], torch.LongTensor]:

    def mask_input_fn(input_ids: torch.LongTensor, baseline_flag: int, masks: torch.BoolTensor):
        """
        [Note] in `masks`: True means the token is NOT masked, False means the token is masked. Not the other way around!
        """

        assert input_ids.shape[0] == 1
        assert len(input_ids.shape) == 2
        # shape of input_ids must be (1, n_tokens)
        assert input_ids.shape[1] == masks.shape[1]

        num_masks = len(masks)
        n_tokens = input_ids.shape[1]

        masked_input_ids = input_ids.expand(num_masks, n_tokens).clone()

        # Use ~ to reverse the boolean masks, because we want to mask the tokens with False
        # Use the baseline flag (set to -42) to indicate position of the masked tokens
        # We will eventually replace the embedding of the masked tokens with the baseline embedding in the forward_function() (see calculate.py)
        masked_input_ids[~masks] = baseline_flag
        return masked_input_ids

    return mask_input_fn



