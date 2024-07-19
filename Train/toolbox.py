import torch
from torch import Tensor
from torchvision import transforms

to_tensor = transforms.Compose([transforms.ToTensor()])


def _rgb_to_y(r: Tensor, g: Tensor, b: Tensor) -> Tensor:
    y: Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    return y


def rgb_to_ycbcr(image):
    """
    IN: RGB (BCHW)
    OUT: Y Cb Cr ()

    """

    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: Tensor = image[..., 0, :, :]
    g: Tensor = image[..., 1, :, :]
    b: Tensor = image[..., 2, :, :]

    delta: float = 0.5
    y: Tensor = _rgb_to_y(r, g, b)
    cb: Tensor = (b - y) * 0.564 + delta
    cr: Tensor = (r - y) * 0.713 + delta
    return y.unsqueeze(1), cb.unsqueeze(1), cr.unsqueeze(1)


def ycbcr_to_rgb(y, cb, cr):
    """
    IN: Y Cb Cr (B1HW)
    OUT: RGB (BCHW)

    """

    if not isinstance(y, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(y)}")
    if not isinstance(cb, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(y)}")
    if not isinstance(cr, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(y)}")

    if len(y.shape) < 3 or y.shape[-3] != 1:
        raise ValueError(f"Input size must have a shape of (*, 1, H, W). Got {y.shape}")
    if len(cb.shape) < 3 or y.shape[-3] != 1:
        raise ValueError(f"Input size must have a shape of (*, 1, H, W). Got {y.shape}")
    if len(cr.shape) < 3 or y.shape[-3] != 1:
        raise ValueError(f"Input size must have a shape of (*, 1, H, W). Got {y.shape}")

    y: Tensor = y.squeeze(1)
    cb: Tensor = cb.squeeze(1)
    cr: Tensor = cr.squeeze(1)

    delta: float = 0.5
    cb_shifted: Tensor = cb - delta
    cr_shifted: Tensor = cr - delta

    r: Tensor = y + 1.403 * cr_shifted
    g: Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: Tensor = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3)
