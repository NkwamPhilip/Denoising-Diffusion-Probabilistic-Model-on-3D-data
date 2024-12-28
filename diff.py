import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch import nn, optim
from monai.transforms import (
    Compose,
    LoadImaged,
    Lambdad,
    EnsureChannelFirstd,
    ScaleIntensityd,
    CenterSpatialCropd,
    Resized
)
import time
from monai.data import Dataset as MonaiDataset
from monai.utils import set_determinism
from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

class MRIMultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, views=['t1c.nii.gz', 't1n.nii.gz', 't2f.nii.gz', 't2w.nii.gz'], transform=None):
        self.root_dir = root_dir
        self.views = views
        self.data_list = self._prepare_data_list()

        if transform is None:
            transform = Compose([
                LoadImaged(keys=["image"]),
                Lambdad(keys="image", func=lambda x: x[:, :, :, 1] if len(x.shape) == 4 else x),
                EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                ScaleIntensityd(keys=["image"]),
                CenterSpatialCropd(keys=["image"], roi_size=[160, 200, 155]),
                Resized(keys=["image"], spatial_size=(32, 40, 32)),
            ])

        self.monai_dataset = MonaiDataset(data=self.data_list, transform=transform)

    def _prepare_data_list(self):
        data_list = []
        for d in os.listdir(self.root_dir):
            subject_path = os.path.join(self.root_dir, d)
            if os.path.isdir(subject_path):
                files = os.listdir(subject_path)
                view_paths = {}
                for view in self.views:
                    view_files = [f for f in files if view in f and 'seg.nii.gz' not in f]
                    if not view_files:
                        break
                    view_paths[view] = os.path.join(subject_path, view_files[0])

                if len(view_paths) == len(self.views):
                    data_list.append({"image": list(view_paths.values())[0]})

        if not data_list:
            raise ValueError(f"No subjects found with all required views in {self.root_dir}")

        return data_list

    def __len__(self):
        return len(self.monai_dataset)

    def __getitem__(self, idx):
        return self.monai_dataset[idx]


def custom_collate_fn(batch):
    batched_views = {"image": []}
    for sample in batch:
        batched_views["image"].append(sample["image"])
    batched_views["image"] = torch.stack(batched_views["image"])
    return batched_views


def split_dataset(dataset, train_ratio=0.8, random_seed=None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    return random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed) if random_seed is not None else None
    )

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def make():
    root_dir = "/Users/mac/Downloads/BRATS"
    batch_size = 4
    random_seed = 42
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    full_dataset = MRIMultiViewDataset(root_dir=root_dir)
    train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=0.8, random_seed=random_seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    # Visualize a few images from the dataset
    plt.subplots(1, 4, figsize=(10, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        img = train_dataset[i * 10]["image"]  # Use an index to get images from the dataset
        plt.imshow(img[0, :, :, 15].detach().cpu(), vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    # Initialize device
    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_res_blocks=2,
    )
    model.to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0005,
                              beta_end=0.0195)
    plt.plot(scheduler.alphas_cumprod.cpu(), color=(2 / 255, 163 / 255, 163 / 255), linewidth=2)
    plt.xlabel("Timestep [t]")
    plt.ylabel("alpha cumprod")

    inferer = DiffusionInferer(scheduler)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-5)

    n_epochs = 300
    val_interval = 25
    epoch_loss_list = []
    val_epoch_loss_list = []

    scaler = GradScaler()
    total_start = time.time()
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                noise = torch.randn_like(images).to(device)
                timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (images.shape[0],),
                                          device=images.device).long()

                noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        epoch_loss_list.append(epoch_loss / (step + 1))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            for step, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                noise = torch.randn_like(images).to(device)
                with torch.no_grad():
                    with autocast(enabled=True):
                        timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (images.shape[0],),
                                                  device=images.device).long()
                        noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())
                val_epoch_loss += val_loss.item()
                progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
            val_epoch_loss_list.append(val_epoch_loss / (step + 1))

            # Sampling image during training
            image = torch.randn((1, 1, 32, 40, 32))
            image = image.to(device)
            scheduler.set_timesteps(num_inference_steps=1000)
            with autocast(enabled=True):
                image = inferer.sample(input_noise=image, diffusion_model=model, scheduler=scheduler)

            plt.figure(figsize=(2, 2))
            plt.imshow(image[0, 0, :, :, 15].cpu(), vmin=0, vmax=1, cmap="gray")
            plt.tight_layout()
            plt.axis("off")
            plt.show()

    total_time = time.time() - total_start
    print(f"train completed, total time: {total_time}.")
    save_model(model, "diffusion_model_final.pt")

if __name__ == "__main__":
    make()

