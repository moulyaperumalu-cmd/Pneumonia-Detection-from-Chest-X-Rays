import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models

from torch.utils.data import DataLoader


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_PATH = "dataset"

MODEL_FOLDER = "model"

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "pneumonia_resnet50.pth"
)

IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 10

LEARNING_RATE = 0.0001


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("=" * 60)
print("AUTOMATED PNEUMONIA DETECTION")
print("RESNET-50 TRANSFER LEARNING")
print("=" * 60)
print()

print("Device:", device)
print()


# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        10
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )

])


test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )

])


# =========================================================
# DATASET PATHS
# =========================================================

train_path = os.path.join(
    DATASET_PATH,
    "train"
)

test_path = os.path.join(
    DATASET_PATH,
    "test"
)


if not os.path.exists(train_path):

    print("ERROR: dataset/train not found.")

    raise SystemExit


if not os.path.exists(test_path):

    print("ERROR: dataset/test not found.")

    raise SystemExit


# =========================================================
# LOAD DATA
# =========================================================

train_dataset = datasets.ImageFolder(

    train_path,

    transform=train_transform

)


test_dataset = datasets.ImageFolder(

    test_path,

    transform=test_transform

)


print(
    "Classes:",
    train_dataset.classes
)

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Testing images:",
    len(test_dataset)
)

print()


# =========================================================
# DATA LOADERS
# =========================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0

)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


dataloaders = {

    "train": train_loader,

    "test": test_loader

}


dataset_sizes = {

    "train": len(train_dataset),

    "test": len(test_dataset)

}


# =========================================================
# RESNET-50
# =========================================================

print("Loading ResNet-50...")


try:

    weights = models.ResNet50_Weights.DEFAULT

    model = models.resnet50(
        weights=weights
    )

except Exception:

    print(
        "Could not download pretrained weights."
    )

    model = models.resnet50(
        weights=None
    )


# =========================================================
# FREEZE RESNET
# =========================================================

for param in model.parameters():

    param.requires_grad = False


# =========================================================
# NEW CLASSIFIER
# =========================================================

number_of_features = (
    model.fc.in_features
)


model.fc = nn.Sequential(

    nn.Linear(
        number_of_features,
        256
    ),

    nn.ReLU(),

    nn.Dropout(
        0.4
    ),

    nn.Linear(
        256,
        2
    )

)


model = model.to(device)


# =========================================================
# LOSS
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = optim.Adam(

    model.fc.parameters(),

    lr=LEARNING_RATE

)


# =========================================================
# TRAINING
# =========================================================

best_accuracy = 0.0

best_model_weights = copy.deepcopy(
    model.state_dict()
)


print()
print("=" * 60)
print("TRAINING STARTED")
print("=" * 60)
print()


for epoch in range(EPOCHS):

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print("-" * 40)


    for phase in ["train", "test"]:

        if phase == "train":

            model.train()

        else:

            model.eval()


        running_loss = 0.0

        running_corrects = 0


        for images, labels in dataloaders[phase]:

            images = images.to(device)

            labels = labels.to(device)


            optimizer.zero_grad()


            with torch.set_grad_enabled(
                phase == "train"
            ):

                outputs = model(
                    images
                )


                loss = criterion(
                    outputs,
                    labels
                )


                predictions = torch.argmax(
                    outputs,
                    dim=1
                )


                if phase == "train":

                    loss.backward()

                    optimizer.step()


            running_loss += (

                loss.item()
                * images.size(0)

            )


            running_corrects += torch.sum(

                predictions == labels

            ).item()


        epoch_loss = (

            running_loss
            / dataset_sizes[phase]

        )


        epoch_accuracy = (

            running_corrects
            / dataset_sizes[phase]

        )


        print(

            f"{phase.upper()} "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: "
            f"{epoch_accuracy * 100:.2f}%"

        )


        if phase == "test":

            if epoch_accuracy > best_accuracy:

                best_accuracy = (
                    epoch_accuracy
                )

                best_model_weights = (
                    copy.deepcopy(
                        model.state_dict()
                    )
                )


    print()


# =========================================================
# SAVE MODEL
# =========================================================

model.load_state_dict(
    best_model_weights
)


os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


torch.save(

    {

        "model_state_dict":
            model.state_dict(),

        "classes":
            train_dataset.classes,

        "image_size":
            IMAGE_SIZE

    },

    MODEL_PATH

)


print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print()

print(
    f"Best Test Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print()

print(
    "Model saved:"
)

print(
    MODEL_PATH
)

print()