import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as tt
from loguru import logger
from torch import nn

from ...utils import download_from_yadisk


class Stage2Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        
        if not os.path.exists("./checkpoints/classifier.pt"):
            logger.info("Classification model's checkpoint was not found locally")
            logger.info("Classifier checkpoint downloading has started")
            download_from_yadisk(
                short_url="https://disk.yandex.ru/d/VEaGQKAfqAv8oQ",
                filename="classifier.pt",
                target_dir="./checkpoints",
            )
        else:
            logger.info("Classification model's checkpoint was found locally")

        self._backbone: nn.Module = torch.load("./checkpoints/classifier.pt")
        self._backbone.eval()

        logger.success("Classifier checkpoint downloaded")

        logger.info("Classes' mapping downloading has started")
        
        download_from_yadisk(
            short_url="https://disk.yandex.ru/d/ecFq83yHVePVZA",
            filename="classes.csv",
            target_dir="./checkpoints",
        )
        self._mapping: pd.DataFrame = pd.read_csv("./checkpoints/classes.csv")
        
        logger.success("Classes' mapping downloaded")

        self._tranforms = tt.Compose(
            [
                tt.Resize((224, 224)),
                tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def single_image_classification(self, input_image: np.ndarray) -> dict[str, float | int]:
        current_device: torch.device = next(self.parameters()).device

        input_as_tensor: torch.Tensor = torch.Tensor(input_image).unsqueeze(0)
        input_as_tensor /= 255
        input_as_tensor = torch.permute(input_as_tensor, (0, 3, 1, 2))

        input_as_tensor = input_as_tensor.to(current_device)

        input_as_tensor = self._tranforms(input_as_tensor)

        prediction: torch.Tensor = self._backbone(input_as_tensor)
        probas: torch.Tensor = torch.softmax(prediction, dim=-1).squeeze(0)
        index: torch.Tensor = torch.argmax(probas)
        label: str = self._mapping.iloc[index.item()].species

        return {
            "prediction": label,
            "confidence": probas[index].item(),
        }
