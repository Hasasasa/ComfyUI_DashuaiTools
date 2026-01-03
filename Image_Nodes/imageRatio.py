import torch


def _get_first_image(image):
    while isinstance(image, list):
        if not image:
            return None
        image = image[0]
    return image


class ImageRetio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("ratio", "width_ratio", "height_ratio")
    FUNCTION = "get_ratio"
    CATEGORY = "Image"

    def get_ratio(self, image):
        img = _get_first_image(image)
        if not isinstance(img, torch.Tensor):
            return ("invalid image", 0, 0)
        arr = img.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            return ("invalid image shape", 0, 0)
        if arr.shape[0] in (1, 3):
            h, w = arr.shape[1], arr.shape[2]
        else:
            h, w = arr.shape[0], arr.shape[1]
        if h == 0:
            return ("invalid image height", 0, 0)
        g = int(torch.gcd(torch.tensor(w), torch.tensor(h)))
        if g <= 0:
            return ("invalid image size", 0, 0)
        w_ratio = int(w // g)
        h_ratio = int(h // g)
        return (f"{w_ratio}:{h_ratio}", w_ratio, h_ratio)


NODE_CLASS_MAPPINGS = {
    "ImageRetio": ImageRetio,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageRetio": "ImageRetio☀",
}
