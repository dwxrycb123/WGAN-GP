from torchvision import datasets, transforms

def get_dataset(name, train=True, download=False):
    dataset_class = AVAILABLE_DATASETS[name]
    dataset_transform = transforms.Compose([
        *AVAILABLE_TRANSFORMS[name]
    ])

    return dataset_class(
        './datasets/{name}'.format(name=name), train=train,
        download=download, transform=dataset_transform,
    )


AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST
}

AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.Scale(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        # transforms.Normalize((0.1307,), (0.3081,)),
    ]
}

DATASET_CONFIGS = {
    'mnist': {'size': 28, 'channels': 1, 'classes': 10}
}
