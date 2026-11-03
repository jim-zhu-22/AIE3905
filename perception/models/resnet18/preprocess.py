from torchvision import transforms
from torchvision.models import ResNet18_Weights

def transform(training: bool):
    weights = ResNet18_Weights.IMAGENET1K_V1
    if not training:
        return weights.transforms()
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
