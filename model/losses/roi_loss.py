import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .lpips import LPIPS


class ROILoss(nn.Module):
    def __init__(
        self,
        model,
        target_layers=None,
        roi_perceptual_weight=1.0,
        device="cuda",
    ):
        super().__init__()

        if model == "resnet50":
            task_model = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1
            )
        else:
            raise ValueError(f"Unsupported ROI model: {model}")

        self.task_model = task_model.to(device).eval()
        for parameter in self.task_model.parameters():
            parameter.requires_grad_(False)

        if model == "resnet50":
            target_layers = [self.task_model.layer4[-1]]

        self.target_layers = target_layers
        self.perceptual_loss = LPIPS().eval().to(device)
        self.roi_perceptual_weight = roi_perceptual_weight

        with open("./dataset/imagenet_class_index.json", "r") as file:
            idx_to_synset = json.load(file)

        self.synset_to_idx = {
            value[0]: int(index)
            for index, value in idx_to_synset.items()
        }

        self.device = device
        self.mse = nn.MSELoss(reduction="none")
        self.ce_loss = nn.CrossEntropyLoss()

    def _preprocess(self, x):
        """Resize and normalize input for the ImageNet-pretrained classifier."""
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=x.device,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=x.device,
        ).view(1, 3, 1, 1)

        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        return (x - mean) / std

    def _activation_map_batch(self, x):
        """
        Compute forward activation maps from the selected classifier layer.

        The feature activations are averaged across channels and normalized
        independently for each image.
        """
        x_proc = self._preprocess(x)

        activations = {}

        def forward_hook(module, inputs, output):
            activations["value"] = output

        hook = self.target_layers[0].register_forward_hook(forward_hook)

        with torch.no_grad():
            _ = self.task_model(x_proc.to(self.device))

        activation_map = activations["value"].mean(
            dim=1,
            keepdim=True,
        )
        activation_map = F.relu(activation_map)

        maps = activation_map.detach().cpu().numpy()

        normalized_maps = []
        for activation in maps:
            activation = activation[0]
            activation = (
                (activation - activation.min())
                / (activation.max() - activation.min() + 1e-8)
            )
            normalized_maps.append(activation)

        hook.remove()

        return normalized_maps

    def roi_cropping(self, inputs, recons):
        """
        Generate an activation-based ROI mask from the original image and
        apply the same mask to the original and reconstructed images.
        """
        activation_maps = self._activation_map_batch(inputs)

        inputs_roi = []
        recons_roi = []
        masked_inputs = []
        masked_recons = []

        for index, activation in enumerate(activation_maps):
            height, width = inputs.shape[2:]

            activation = torch.tensor(
                activation,
                dtype=torch.float32,
                device=inputs.device,
            )

            activation = F.interpolate(
                activation.unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

            mask = (activation > 0.5).float()

            input_roi = inputs[index:index + 1] * mask
            recon_roi = recons[index:index + 1] * mask

            inputs_roi.append(input_roi)
            recons_roi.append(recon_roi)
            masked_inputs.append(mask * inputs[index:index + 1])
            masked_recons.append(mask * recons[index:index + 1])

        return (
            torch.cat(inputs_roi),
            torch.cat(recons_roi),
            torch.cat(masked_inputs),
            torch.cat(masked_recons),
        )

    def forward(self, reconstructions, inputs, labels=None):
        inputs_roi, recons_roi, mask_in, mask_rec = self.roi_cropping(
            inputs,
            reconstructions,
        )

        # LPIPS on the activation-selected ROI.
        roi_p_loss = self.perceptual_loss(
            inputs_roi.contiguous(),
            recons_roi.contiguous(),
        )

        # Reconstruction MSE on the masked images.
        roi_rec_loss = self.mse(
            mask_rec.contiguous(),
            mask_in.contiguous(),
        ).mean()

        roi_loss = (
            roi_rec_loss
            + self.roi_perceptual_weight * roi_p_loss.mean()
        )

        # Classification loss used by the Stage-2 objective.
        logits = self.task_model(
            reconstructions.to(self.device)
        )

        roi_task_loss = None

        if labels is not None:
            labels = [
                self.synset_to_idx.get(label, None)
                for label in labels
            ]
            labels = torch.tensor(
                labels,
                dtype=torch.long,
                device=self.device,
            )
            roi_task_loss = self.ce_loss(logits, labels)

        return (
            roi_rec_loss,
            roi_p_loss,
            roi_loss,
            roi_task_loss,
        )
