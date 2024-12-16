from typing import Any
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes

from lightly.transforms.swav_transform import SwaVTransform
from torchvision.models import ResNet18_Weights

from EndoFM import vision_transformer
from lightly.models.modules.heads import ProjectionHead


def ignore_target_transform(target):
    return 0


class SwAVProjectionHeadVIT(ProjectionHead):
    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 128
    ):
        super(SwAVProjectionHeadVIT, self).__init__(
            [
                (input_dim, hidden_dim, nn.LayerNorm(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class SwAV(pl.LightningModule):
    def __init__(self, weights_path="./EndoFM/endo_fm.pth", backbone: str = "resnet"):
        super(SwAV, self).__init__()
        if backbone == "vit":
            # Load pre-trained weights if provided
            self.backbone = vision_transformer.VisionTransformer()
            if weights_path is not None:
                state_dict = torch.load(weights_path, map_location=torch.device('cpu'))["student"]
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.backbone.'):
                        new_key = k[len('module.backbone.'):]
                        new_state_dict[new_key] = v
                    else:
                        new_state_dict[k] = v

                self.backbone.load_state_dict(new_state_dict, strict=False)
            self.projection_head = SwAVProjectionHeadVIT(input_dim=self.backbone.embed_dim, output_dim=128)
        elif backbone == "resnet":
            resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.projection_head = SwaVProjectionHead(512, 512, 128)

        self.prototypes = SwaVPrototypes(input_dim=128, n_prototypes=256)
        self.criterion = SwaVLoss()

    def forward(self, x) -> Any:
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p

    def training_step(self, batch, batch_idx):
        self.prototypes.normalize()
        views = batch[0]
        multi_crop_features = [self.forward(view.to(self.device)) for view in views]
        high_res = multi_crop_features[:2]
        low_res = multi_crop_features[2:]
        loss = self.criterion(high_res, low_res)
        return loss

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=0.0001)

def extract_features(feature_extractor,dataloader, model_device):
    features, labels = [], []
    with torch.no_grad():
        for (x, y) in dataloader:
            x = x.to(model_device)  # Move input data to the same device as the model
            y = y.to(model_device)
            feature = feature_extractor(x).flatten(start_dim=1)
            features.append(feature.cpu())  # Move features back to CPU for further processing
            labels.append(y.cpu())  # Move labels back to CPU for further processing
    return torch.cat(features), torch.cat(labels)


if __name__ == "__main__":
    model = SwAV()

    transform = SwaVTransform()

    dataset = torchvision.datasets.VOCDetection(
        "datasets/pascal_voc",
        download=True,
        transform=transform,
        target_transform=ignore_target_transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )

    # Set the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Move the model to the device
    model = model.to(device)

    trainer = pl.Trainer(max_epochs=2, devices=1, accelerator="mps", log_every_n_steps=1)
    trainer.fit(model=model, train_dataloaders=dataloader)

    # Save the model weights after training
    torch.save(model.state_dict(), "swav_model.pth")

    # # Load CIFAR-10 dataset
    # transform_cifar = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(224),  # Resize to match input size of the pretrained model
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # ])
    #
    # cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
    # cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
    #
    # train_loader = DataLoader(cifar_train, batch_size=128, shuffle=False, num_workers=8)
    # test_loader = DataLoader(cifar_test, batch_size=128, shuffle=False, num_workers=8)
    #
    # #------- With Pretraining-------#
    # # Feature extraction
    # print("-----WITH PRETRAINING-----")
    # feature_extractor = model.backbone
    # feature_extractor.to(device)
    #
    # # Extract features from CIFAR-10 dataset
    # train_features, train_labels = extract_features(feature_extractor,train_loader, device)
    # test_features, test_labels = extract_features(feature_extractor,test_loader, device)
    #
    # # Train a logistic regression model on the extracted features
    # clf = LogisticRegression(max_iter=1000)
    # clf.fit(train_features.numpy(), train_labels.numpy())
    #
    # # Evaluate on test set
    # test_predictions = clf.predict(test_features.numpy())
    # accuracy = accuracy_score(test_labels.numpy(), test_predictions)
    # print(f"Test Accuracy: {accuracy * 100:.2f}%")
    #
    # # ------- Without Pretraining-------#
    # # Feature extraction
    # print("-----WITHOUT PRETRAINING-----")
    # resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # feature_extractor =  nn.Sequential(*list(resnet.children())[:-1])
    # feature_extractor.to(device)
    #
    #
    # # Extract features from CIFAR-10 dataset
    # train_features, train_labels = extract_features(feature_extractor,train_loader, device)
    # test_features, test_labels = extract_features(feature_extractor,test_loader, device)
    #
    # # Train a logistic regression model on the extracted features
    # clf = LogisticRegression(max_iter=1000)
    # clf.fit(train_features.numpy(), train_labels.numpy())
    #
    # # Evaluate on test set
    # test_predictions = clf.predict(test_features.numpy())
    # accuracy = accuracy_score(test_labels.numpy(), test_predictions)
    # print(f"Test Accuracy: {accuracy * 100:.2f}%")