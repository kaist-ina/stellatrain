
import os
from torchvision.datasets import ImageFolder

class ImageNet100(ImageFolder):
    def __init__(self, root, split, transform):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'imagenet100.txt')) as f:
            classes = [line.strip() for line in f]
            class_to_idx = { cls: idx for idx, cls in enumerate(classes) }

        super().__init__(os.path.join(root, split), transform=transform)
        samples = []
        for path, label in self.samples:
            cls = self.classes[label]
            if cls not in class_to_idx:
                continue
            label = class_to_idx[cls]
            samples.append((path, label))

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in samples]
