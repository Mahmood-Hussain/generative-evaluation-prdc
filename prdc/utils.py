from torchvision import transforms
from PIL import Image

def resize(img, w, h):
    return img.resize((w, h), Image.Resampling.BILINEAR)
customtransforms = transforms.Compose([
        transforms.Lambda(lambda img: resize(img, 244, 244)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])