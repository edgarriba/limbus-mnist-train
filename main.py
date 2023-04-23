import kornia as K
from kornia.core import Tensor
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import asyncio
import numpy as np
import cv2
import visdom

import limbus
from limbus.core.component import Component, ComponentState


class DataLoader(Component):
    def __init__(self, name: str, mode: str) -> None:
        super().__init__(name)
        self.is_train = mode == "train"
        self.dataset = datasets.MNIST(root="data", train=self.is_train, download=True)
        self.dataset_iter = iter(self.dataset)

        self.outputs.declare("image", np.ndarray)
        self.outputs.declare("label", Tensor)
    
    async def forward(self) -> ComponentState:
        try:
            sample = next(self.dataset_iter)
        except StopIteration:
            return ComponentState.STOPPED

        image, label = sample[0], sample[1]

        image = np.asarray(image).copy()
        label_t = torch.as_tensor(label)

        await asyncio.gather(
            self.outputs.image.send(image), self.outputs.label.send(label_t))

        return ComponentState.OK


class Batcher(Component):
    def __init__(self, name: str, batch_size: int):
        super().__init__(name)
        self.batch_size = batch_size

        self.inputs.declare("image", Tensor)
        self.inputs.declare("label", Tensor)

        self.outputs.declare("image", Tensor)
        self.outputs.declare("label", Tensor)

    async def forward(self) -> ComponentState:
        images = []
        labels = []

        for _ in range(self.batch_size):
            image = await self.inputs.image.receive()
            label = await self.inputs.label.receive()

            images.append(image)
            labels.append(label)

        images = torch.stack(images)
        labels = torch.stack(labels)

        await asyncio.gather(
            self.outputs.image.send(images), self.outputs.label.send(labels))

        return ComponentState.OK


class Preprocessor(Component):
    def __init__(self, name: str):
        super().__init__(name)
        self.inputs.declare("image", np.ndarray)
        self.outputs.declare("image", Tensor)
    
    async def forward(self) -> ComponentState:
        image = await self.inputs.image.receive()
        image = K.utils.image_to_tensor(image)
        image = image.float() / 255.0
        image = image.repeat(3, 1, 1)
        await self.outputs.image.send(image)
        return ComponentState.OK


class Augmentations(Component):
    def __init__(self, name: str):
        super().__init__(name)
        self.aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomPerspective(distortion_scale=0.5, p=0.5)
        )

        self.inputs.declare("image", Tensor)
        self.outputs.declare("image", Tensor)
        self.outputs.declare("transformation_matrix", Tensor)

    async def forward(self) -> ComponentState:
        image = await self.inputs.image.receive()
        image_aug = self.aug(image)

        await asyncio.gather(
            self.outputs.image.send(image_aug),
            self.outputs.transformation_matrix.send(self.aug.transform_matrix))

        return ComponentState.OK
    

class OpencvWindow(Component):
    def __init__(self, name: str):
        super().__init__(name)
        self.inputs.declare("image", np.ndarray)
    
    async def forward(self) -> ComponentState:
        image = await self.inputs.image.receive()
        cv2.imshow(self.name, image)
        cv2.waitKey(10)
        return ComponentState.OK


class VisdomManager(Component):
    def __init__(self, name: str, port: int = 8097) -> None:
        super().__init__(name)
        self.inputs.declare("image", Tensor)
        self.inputs.declare("image_augmented", Tensor)
        self.inputs.declare("loss", Tensor)
        self.num_iters = 0

        self._manager = visdom.Visdom(port=port)
        if not self._manager.check_connection():
            raise ConnectionError(
                f"Error connecting with the visdom server. Run in your termnal: visdom -port {port}."
            )

    async def forward(self) -> ComponentState:
        coros = [
            self.inputs.image.receive(),
            self.inputs.image_augmented.receive(),
            self.inputs.loss.receive(),
        ]
        tasks = [asyncio.create_task(coro) for coro in coros]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            if task is tasks[0]:
                image = task.result()

                if image.shape[-3] == 1:
                    image = image.repeat(1, 3, 1, 1)
                self._manager.images(image, win=self.name)

            elif task is tasks[1]:
                image_augmented = task.result()

                if image_augmented.shape[-3] == 1:
                    image_augmented = image_augmented.repeat(1, 3, 1, 1)
                self._manager.images(image_augmented, win=f"{self.name}_augmented")

            elif task is tasks[2]:
                loss = task.result()
                print(f"Loss: {loss}")

                self._manager.line(
                    X=torch.tensor([self.num_iters]),
                    Y=torch.tensor([loss]),
                    win=f"{self.name}_loss", update="append")
                
                self.num_iters += 1

        return ComponentState.OK
    

class CrossEntropyLoss(Component):
    def __init__(self, name: str):
        super().__init__(name)
        self.inputs.declare("logits", Tensor)
        self.inputs.declare("label", Tensor)
        self.outputs.declare("loss", Tensor)

        self.criterion = torch.nn.CrossEntropyLoss()
    
    async def forward(self) -> ComponentState:
        prediction = await self.inputs.logits.receive()
        label = await self.inputs.label.receive()

        loss = self.criterion(prediction, label)
        await self.outputs.loss.send(loss)
        return ComponentState.OK


class Model(Component):
    def __init__(self, name: str):
        super().__init__(name)
        self.model = models.resnet18(pretrained=True)
        self.inputs.declare("image", Tensor)
        self.outputs.declare("logits", Tensor)
    
    async def forward(self) -> ComponentState:
        image = await self.inputs.image.receive()
        prediction = self.model(image)
        await self.outputs.logits.send(prediction)
        return ComponentState.OK


class Optimizer(Component):
    def __init__(self, name: str, model: torch.nn.Module, lr: float = 0.001):
        super().__init__(name)
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.inputs.declare("loss", Tensor)
    
    async def forward(self) -> ComponentState:
        loss = await self.inputs.loss.receive()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return ComponentState.OK


class ValidationProcess(Component):
    def __init__(self, name: str, model: torch.nn.Module, dataloader: DataLoader):
        super().__init__(name)

        self.model = model
        self.dataloader = dataloader
        self.inputs.declare("start", bool)
    
    async def forward(self) -> ComponentState:
        loss = await self.inputs.loss.receive()
        await self.outputs.loss.send(loss)
        return ComponentState.OK


class ValidationComponent(Component):
    def __init__(self, name: str, model: torch.nn.Module, dataloader: DataLoader):
        super().__init__(name)

        self.model = model
        self.dataloader = DataLoader("valid_dataloader", mode="valid")
        self.inputs.declare("start", bool)
    
    async def forward(self) -> ComponentState:
        await self.inputs.start.receive()

        loss = 0.
        await self.outputs.loss.send(loss)
        return ComponentState.OK


async def main():
    print(K.__version__)

    pipeline = limbus.Pipeline()

    # training process

    train_dataloader = DataLoader("train_dataloader", mode="train")

    batcher = Batcher("batcher", batch_size=32)
    preprocessor = Preprocessor("preprocessor")
    augmentations = Augmentations("augmentations")

    criterion = CrossEntropyLoss("criterion")
    model = Model("model")
    optimizer = Optimizer("optimizer", model.model)

    viz = OpencvWindow("viz")
    vizdom = VisdomManager("vizdom", port=8098)

    train_dataloader.outputs.image >> viz.inputs.image
    train_dataloader.outputs.image >> preprocessor.inputs.image
    train_dataloader.outputs.label >> batcher.inputs.label
    preprocessor.outputs.image >> batcher.inputs.image

    batcher.outputs.image >> vizdom.inputs.image

    # only for images
    batcher.outputs.image >> augmentations.inputs.image
    augmentations.outputs.image >> vizdom.inputs.image_augmented
    augmentations.outputs.image >> model.inputs.image

    model.outputs.logits >> criterion.inputs.logits
    batcher.outputs.label >> criterion.inputs.label

    criterion.outputs.loss >> optimizer.inputs.loss
    criterion.outputs.loss >> vizdom.inputs.loss

    # validation process

    valid_dataloader = DataLoader("valid_dataloader", mode="valid")

    valid_viz = OpencvWindow("valid_viz")

    train_dataloader.outputs.start >> valid_dataloader.inputs.start
    valid_dataloader.outputs.image >> valid_viz.inputs.image


    pipeline.add_nodes([train_dataloader,])

    await pipeline.async_run()


if __name__ == "__main__":
    asyncio.run(main())

