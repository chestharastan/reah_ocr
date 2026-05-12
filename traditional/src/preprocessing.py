from torchvision import transforms

def get_transform(image_height: int, image_width: int):
    transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    return transform