# -*- coding: utf-8 -*-
import kagglehub

# Download latest version
path = kagglehub.dataset_download("jessicali9530/stanford-cars-dataset")

print("Path to dataset files:", path)

!pip install matplotlib

import os
import glob
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt # Importing matplotlib.pyplot for plotting
import math
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
import torchvision.models.segmentation as segmentation
import scheduler
torch.manual_seed(13)
torch.cuda.manual_seed(13)

class Config(dict):
    def __init__(self):
        super().__init__(
            dict(
                input_path = "/root/.cache/kagglehub/datasets/jessicali9530/stanford-cars-dataset/versions/2",
                output_path = "working",
                checkpoint_path = "mbr.pt",
                epochs = 100,
                input_size = (3,64,64),
                hidden_dim=96,
                patch_size=4,
                num_heads=6,
                window_size=7,
                upsample = 4,
                cpu_count = mp.cpu_count(),
                gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0,
                device = "cuda" if torch.cuda.is_available() else "cpu",
                num_workers=2,
                batch_size=1,
                dist_enabled = True,
                dist_worldsize = os.environ["DIST_WORLDSIZE"] if "DIST_WORLDSIZE" in os.environ else (torch.cuda.device_count() if torch.cuda.is_available() else 1),
                dist_port = os.environ["DIST_PORT"] if "DIST_PORT" in os.environ else 23456,
                dist_hostname = os.environ["DIST_HOSTNAME"] if "DIST_HOSTNAME" in os.environ else "localhost",
                rank = dist.get_rank() if dist.is_initialized() else None,
                localrank = dist.get_rank() % torch.cuda.device_count() if (torch.cuda.is_available() and dist.is_initialized()) else 0
            ))

import math

class Scheduler:
    def __init__(self, optimizer, iterations, mode="linear", start=0, value=1, final=0, warmup=0, name=""):
        self.iterations = iterations
        self.optimizer = optimizer
        self.mode = mode
        self.start = start
        self.value = value
        self.final = final
        self.warmup = warmup
        self.name = name

        self.i = 0

    def step(self, iteration=None):
        if iteration is None:
            iteration = self.i
        self.i += 1

        value = self.get_value(iteration)
        for param_group in self.optimizer.param_groups:
            param_group[self.name] = value

    def get_value(self, iteration):
        if iteration < self.warmup:
            return self.start + (self.value - self.start) * iteration / self.warmup

        if self.mode == "linear":
            return self.value + (self.final - self.value) * (iteration - self.warmup) / (self.iterations - self.warmup)

        if self.mode == "cosine":
            return self.final + (self.value - self.final) * 0.5 * (1 + math.cos(math.pi * (iteration - self.warmup) / (self.iterations - self.warmup)))

        return self.value

class Constant(Scheduler):
    def __init__(self, optimizer, value, name=""):
        super().__init__(optimizer, iterations=1, mode="constant", value=value, name=name)

    def step(self, iteration=None):
        super().step(0)

class Linear(Scheduler):
    def __init__(self, optimizer, iterations, start=0, value=1, final=0, warmup=0, name=""):
        super().__init__(optimizer, iterations, mode="linear", start=start, value=value, final=final, warmup=warmup, name=name)

class Cosine(Scheduler):
    def __init__(self, optimizer, iterations, value=1, final=0, warmup=0, name=""):
        super().__init__(optimizer, iterations, mode="cosine", start=final, value=value, final=final, warmup=warmup, name=name)

class LR(Scheduler):
    def __init__(self, optimizer, mode, iterations, start=0, value=1, final=0, warmup=0):
        name = "lr"
        for param_group in optimizer.param_groups:
            if name not in param_group:
                param_group[name] = 0.0
        super().__init__(optimizer, iterations, mode, start, value, final, warmup, name)

class CarDataset(torch.utils.data.Dataset):
    def __init__(self, config, train=True, repeat=1):
        self.files = glob.glob(os.path.join(config["input_path"],"cars_train/**/*.jpg" if train else "cars_test/**/*.jpg"), recursive=True)
        self.repeat = repeat
        self.mean, self.std = self.calculate_mean_std()
        self.training = train
        self.config=config

    def calculate_mean_std(self):
        # Calculate the mean and std based on the entire dataset
        # This is done only once during the initialization of the dataset
        all_images = []
        for path in self.files:
            image = torchvision.io.read_image(path) / 255
            if image.shape[0] == 1: continue
            all_images.append(image.view(3, -1))

        if not all_images:  # Handle the case where all images are grayscale
            return torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5])

        all_images = torch.cat(all_images, -1)
        return all_images.mean(-1), all_images.std(-1)

    def __len__(self):
        # Ensure len(self.files) is greater than 0 and adjust the divisor if needed
        # This example assumes at least 5 training images and 200 eval images
        return max(1, len(self.files) * self.repeat // (5 if self.training else 200))  # Ensure length is at least 1

    def __getitem__(self,idx):
        idx = idx % len(self.files)
        path = self.files[idx]
        image = torchvision.io.read_image(path) / 255
        if image.shape[0] == 1:
            image = image.repeat(3,1,1)
        image = T.RandomHorizontalFlip()(image)
        image = T.RandomResizedCrop(image.shape[-2:], scale=(0.25, 1.0), antialias=True)(image)
        image = T.ColorJitter(brightness=0.2,contrast=0.2, saturation=0.1, hue=0.05)(image)
        h,w = self.config["input_size"][-2:]
        image = T.Resize((h*config["upsample"], w*config["upsample"]), antialias=True)(image)
        image_lq = T.Resize((h,w), antialias=True)(image)
        return image_lq, image

import torchvision.models.swin_transformer as swin

class PatchUnembed(nn.Module):
    def forward(self, x):
        return x.permute(0,3,1,2)

class PatchEmbed(nn.Module):
    def forward(self, x):
        return x.permute(0,2,3,1)


class RSTB(nn.Sequential):
    def __init__(self, config, sdp):
        super().__init__(
                *[swin.SwinTransformerBlockV2(
                    dim=config["hidden_dim"], num_heads=4, window_size=[config["window_size"]]*2,
                    shift_size=[0,0] if (i%2==0) else [config["window_size"]//2]*2,
                    stochastic_depth_prob=sdp[i]) for i in range(6)],
                PatchUnembed(),
                nn.Conv2d(config["hidden_dim"], config["hidden_dim"]//4, 3, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(config["hidden_dim"]//4, config["hidden_dim"]//4, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(config["hidden_dim"]//4, config["hidden_dim"],3,padding=1),
                PatchEmbed(),
        )


class SwinIR(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embed = nn.Sequential(
            nn.Conv2d(config["input_size"][0],config["hidden_dim"], 3, padding=1),
            PatchEmbed()
        )
        sdp = [v.item() for v in torch.linspace(0,0.2,36)]
        self.rstb_layers = nn.ModuleList(
            [RSTB(config, sdp[i*6:(i+1)*6]) for i in range(6)]
        )
        self.unembed = nn.Sequential(
            PatchUnembed(),
            nn.Conv2d(config["hidden_dim"],config["hidden_dim"]//4,3,padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(config["hidden_dim"]//4,config["hidden_dim"]//4,1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(config["hidden_dim"]//4, config["hidden_dim"], 3, padding=1),

            nn.Conv2d(config["hidden_dim"], config["hidden_dim"], 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(config['hidden_dim'], config["hidden_dim"]*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(config["hidden_dim"], config["hidden_dim"], 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(config["hidden_dim"], config["hidden_dim"]*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(config["hidden_dim"], config["hidden_dim"], 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(config["hidden_dim"], config["input_size"][0], 3, padding=1)
        )


    def forward(self, x):
        y = self.embed(x)
        y0 = y
        for rstb in self.rstb_layers:
            y = rstb(y) + y
        y = y + y0
        return self.unembed(y) + F.interpolate(x, (self.config["input_size"][-2]*self.config["upsample"], self.config["input_size"][-1]*self.config["upsample"]), mode="bilinear")

def main(config, model):
    start_time = time.time()

    if dist.is_initialized():
        if config["localrank"] == 0:
            dst = CarDataset(config, train=True)
            dse = CarDataset(config, train=False)
            dist.barrier()
        else:
            dist.barrier()
            dst = CarDataset(config, train=True)
            dse = CarDataset(config, train=False)

        st = torch.utils.data.distributed.DistributedSampler(dst, drop_last=True)
        dataloader_train = DataLoader(dst, batch_size=config["batch_size"], num_workers=config["num_workers"], sampler=st, pin_memory=True, persistent_workers=config["num_workers"] > 0, drop_last=True)
        se = torch.utils.data.distributed.DistributedSampler(dse, drop_last=True)
        dataloader_eval = DataLoader(dse, batch_size=config["batch_size"], num_workers=config["num_workers"], sampler=se, pin_memory=True, persistent_workers=config["num_workers"] > 0, drop_last=True)

    else:
        dst = CarDataset(config, train=True)
        dse = CarDataset(config, train=False)
        dataloader_train = DataLoader(dst, batch_size=config["batch_size"], num_workers=config["num_workers"], pin_memory=True, persistent_workers=config["num_workers"] > 0, shuffle=True, drop_last=True)
        dataloader_eval = DataLoader(dse, batch_size=config["batch_size"], num_workers=config["num_workers"], pin_memory=True, persistent_workers=config["num_workers"] > 0, shuffle=True, drop_last=True)

    nit = len(dataloader_train) * config["epochs"]
    opt = torch.optim.AdamW(model.parameters())
    sched_lr = scheduler.LR(opt, scheduler.Linear, start=1e-7, value=2e-4, final=1e-7, iterations=nit, warmup=100)

    for start_epoch in range(config["epochs"],0,-1):
        path = os.path.join(config["output_path"], config["checkpoint_path"].format(epoch=start_epoch, **config))
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location="cpu")
            sd_model, sd_opt = checkpoint["model"], checkpoint["opt"]
            if sd_model:
                model.load_state_dict(sd_model)
            if sd_opt:
                opt.load_state_dict(sd_opt)
            start_epoch = checkpoint["epoch"]
            break
    else:
        start_epoch = 0

    sched_lr.step(start_epoch)

    for e in range(start_epoch, config["epochs"]):
        for training, dl in [(True, dataloader_train), (False, dataloader_eval)]:
            torch.set_grad_enabled(training)
            model.train(training)
            if isinstance(dl.sampler,torch.utils.data.distributed.DistributedSampler):
                dl.sampler.set_epoch(e)

            if config["localrank"] == 0:
                print()
                print("Training" if training else "Eval")
                print("--------" if training else "----")

            window = []
            for i, (batch, target) in enumerate(dl):
                current_time = time.time()
                batch = batch.to(config["device"], dtype=torch.float)
                target = target.to(config["device"], dtype=torch.float)
                batch = (batch - dl.dataset.mean[...,None,None].to(batch.device)) / dl.dataset.std[...,None,None].to(batch.device)
                target = (target - dl.dataset.mean[...,None,None].to(target.device)) / dl.dataset.std[...,None,None].to(target.device)
                pred = model(batch)
                loss = (pred - target).abs().mean()

                if training:
                    opt.zero_grad()
                    loss.mean().backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
                    opt.step()
                    sched_lr.step()

                if dist.is_initialized():
                    loss = loss.detach()
                    dist.all_reduce(loss, dist.ReduceOp.AVG)

                window.append(loss.cpu().item())
                if len(window) > 1000: window.pop(0)

                if config["localrank"] == 0:
                    dt = int(current_time - start_time)
                    print(f"E{e+1:03d} {i:06d}/{len(dl)} loss: {sum(window)/len(window):0.06f} time:{dt//3600:03d}:{(dt//60)%60:02d}:{dt%60:02d}", end="        \r")

                if config["localrank"] == 0 and (i == len(dl)-1):
                    with torch.no_grad():
                        b = F.interpolate(batch, pred.shape[-2:], mode="bilinear")* dl.dataset.std[...,None,None].to(batch.device) + dl.dataset.mean[...,None,None].to(batch.device)
                        p = pred * dl.dataset.std[...,None,None].to(pred.device) + dl.dataset.mean[...,None,None].to(pred.device)
                        t = target * dl.dataset.std[...,None,None].to(target.device) + dl.dataset.mean[...,None,None].to(target.device)
                        pt = torch.cat((b,p,t))

                        # Using matplotlib.pyplot for plotting instead of 'inline.plot'
                        for image in pt:
                            plt.imshow(image.permute(1, 2, 0).cpu())  # Assuming images are in CHW format
                            plt.show()

        if config["localrank"] == 0:
            path = os.path.join(config["output_path"], config["checkpoint_path"].format(epoch=e+1, **config))
            checkpoint = dict(
                            model=model.state_dict(),
                            opt=opt.state_dict(),
                            loss=sum(window)/len(window),
                            epoch=e+1,
                            time=time.time() - start_time)
            torch.save(checkpoint, path)

def run_distributed(rank):
    config = Config()
    localrank = rank % config["gpu_count"]
    torch.cuda.set_device("cuda:" + str(localrank))
    config["rank"] = rank
    config["localrank"] = localrank
    print(f"[{rank}] Waiting for peers...")
    dist.init_process_group(
            backend="nccl",
            init_method="tcp://" + config["dist_hostname"] + ":" + str(config["dist_port"]),
            rank=rank,
            world_size= config["dist_worldsize"])
    if localrank == 0:
        print("starting...")
    assert dist.is_initialized()
    model = DistributedDataParallel(SwinIR(config).cuda(), find_unused_parameters=False)
    main(config, model)

if __name__ == "__main__":
    config = Config()
    if config["device"] == "cpu":
        # CPU path
        print("Using CPU")
        model = SwinIR(config)
        main(config, model)

    elif config["gpu_count"] > 1:
        if config["dist_enabled"]:
            # DistributedDataParallel path
            print("Using distributed-data-parallel GPU")
            mp.spawn(run_distributed, nprocs=min(config["dist_worldsize"], config["gpu_count"]))
        else:
            # DataParallel path
            print("Using data-parallel GPU")
            model = DataParallel(SwinIR(config).cuda())
            main(config, model)
    else:
        # single GPU CUDA path
        print("Using single GPU")
        model = SwinIR(config).cuda()
        main(config, model)
