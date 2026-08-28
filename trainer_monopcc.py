from __future__ import absolute_import, division, print_function

import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import datasets
import networks
from layers import (
    BackprojectDepth,
    Project3D,
    SSIM,
    compute_errors,
    disp_to_depth,
    get_smooth_loss,
    transformation_from_parameters,
)
from utils import readlines, sec_to_hm_str


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(
            self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, \
            "height must be a multiple of 32"
        assert self.opt.width % 32 == 0, \
            "width must be a multiple of 32"
        assert self.opt.frame_ids[0] == 0, \
            "frame_ids must start with 0"
        assert self.opt.dataset == "endovis", \
            "this trainer supports the endovis dataset"
        assert not self.opt.use_stereo, \
            "this trainer supports monocular training only"
        assert self.opt.pose_model_type == "separate_resnet", \
            "pose_model_type must be separate_resnet"
        assert self.opt.pose_model_input == "pairs", \
            "pose_model_input must be pairs"
        assert not self.opt.v1_multiscale, \
            "this trainer uses full-resolution reprojection"

        self.device = torch.device(
            "cpu" if self.opt.no_cuda else "cuda")
        self.num_scales = len(self.opt.scales)
        self.ema = 0.75
        self.stm_amplitude_mix = 0.5
        self.validation_frequency = 100

        self.opt.learning_rate = 2e-5

        self.models = {}
        self.parameters_to_train = []

        self.models["encoder"] = networks.mpvit_small()
        self.models["encoder"].to(self.device)

        self.models["depth"] = networks.DepthDecoder()
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(
            self.models["depth"].parameters())

        self.models["frozen_encoder"] = networks.mpvit_small()
        self.models["frozen_encoder"].to(self.device)
        self.models["frozen_depth"] = networks.DepthDecoder()
        self.models["frozen_depth"].to(self.device)

        self.models["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=2,
        )
        self.models["pose_encoder"].to(self.device)
        self.parameters_to_train += list(
            self.models["pose_encoder"].parameters())

        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2,
        )
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(
            self.models["pose"].parameters())

        self.models["frozen_pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=2,
        )
        self.models["frozen_pose_encoder"].to(self.device)
        self.models["frozen_pose"] = networks.PoseDecoder(
            self.models["frozen_pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2,
        )
        self.models["frozen_pose"].to(self.device)

        parameter_groups = [
            {
                "params": self.parameters_to_train,
                "lr": self.opt.learning_rate,
            },
            {
                "params": list(self.models["encoder"].parameters()),
                "lr": self.opt.learning_rate,
            },
        ]
        self.model_optimizer = optim.AdamW(parameter_groups)
        self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.model_optimizer, gamma=0.9)

        if self.opt.load_weights_folder is not None:
            self.load_model()
        else:
            self.copy_student_to_teacher()

        for model_name, model in self.models.items():
            if model_name.startswith("frozen_"):
                for parameter in model.parameters():
                    parameter.requires_grad_(False)

        split_path = os.path.join(
            os.path.dirname(__file__),
            "splits",
            self.opt.split,
            "{}_files.txt",
        )
        train_filenames = readlines(split_path.format("train"))
        val_filenames = readlines(split_path.format("test"))
        image_extension = ".png" if self.opt.png else ".jpg"

        train_dataset = datasets.SCAREDRAWDataset(
            self.opt.data_path,
            train_filenames,
            self.opt.height,
            self.opt.width,
            self.opt.frame_ids,
            4,
            is_train=True,
            img_ext=image_extension,
        )
        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_dataset = datasets.SCAREDRAWDataset(
            self.opt.data_path,
            val_filenames,
            self.opt.height,
            self.opt.width,
            self.opt.frame_ids,
            4,
            is_train=False,
            img_ext=image_extension,
        )
        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            False,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        self.num_total_steps = (
            len(train_dataset) // self.opt.batch_size
            * self.opt.num_epochs
        )
        gt_path = os.path.join(
            os.path.dirname(__file__),
            "splits",
            self.opt.split,
            "gt_depths.npz",
        )
        self.gt_depths = np.load(
            gt_path,
            fix_imports=True,
            allow_pickle=True,
            encoding="latin1",
        )["data"]

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(
                os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            height = self.opt.height // (2 ** scale)
            width = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(
                self.opt.batch_size, height, width)
            self.backproject_depth[scale].to(self.device)
            self.project_3d[scale] = Project3D(
                self.opt.batch_size, height, width)
            self.project_3d[scale].to(self.device)

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ",
              self.opt.log_dir)
        print("Training is using:\n  ", self.device)
        print("Using split:\n  ", self.opt.split)
        print(
            "There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        for model_name, model in self.models.items():
            if model_name.startswith("frozen_"):
                model.eval()
            else:
                model.train()

    def set_eval(self):
        for model in self.models.values():
            model.eval()

    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.best_metrics = {
            "abs": float("inf"),
            "sq": float("inf"),
            "rm": float("inf"),
            "rmlog": float("inf"),
        }

        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            _, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time
            if batch_idx % self.opt.log_frequency == 0:
                self.log_time(
                    batch_idx, duration, losses["loss"].detach().cpu())
                self.log("train", losses)

            if self.step % self.validation_frequency == 0:
                self.val()

            self.step += 1

        self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        for key, value in inputs.items():
            inputs[key] = value.to(self.device)

        target_features = self.models["encoder"](
            inputs[("color_aug", 0, 0)])
        outputs = self.models["depth"](target_features)
        outputs.update(self.predict_poses(inputs, frozen=False))

        frozen_source_outputs = {}
        with torch.no_grad():
            frozen_pose_outputs = self.predict_poses(inputs, frozen=True)
            for frame_id in self.opt.frame_ids[1:]:
                source_features = self.models["frozen_encoder"](
                    inputs[("color_aug", frame_id, 0)])
                frozen_source_outputs[frame_id] = self.models[
                    "frozen_depth"](source_features)

        self.generate_images_pred(
            inputs,
            outputs,
            frozen_source_outputs,
            frozen_pose_outputs,
        )
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def predict_poses(self, inputs, frozen=False):
        prefix = "frozen_" if frozen else ""
        pose_encoder = self.models[prefix + "pose_encoder"]
        pose_decoder = self.models[prefix + "pose"]
        pose_features = {
            frame_id: inputs[("color_aug", frame_id, 0)]
            for frame_id in self.opt.frame_ids
        }
        outputs = {}

        for frame_id in self.opt.frame_ids[1:]:
            if frame_id < 0:
                pose_inputs = [pose_features[frame_id], pose_features[0]]
            else:
                pose_inputs = [pose_features[0], pose_features[frame_id]]

            pose_inputs = pose_encoder(torch.cat(pose_inputs, 1))
            axisangle, translation = pose_decoder([pose_inputs])

            outputs[("axisangle", 0, frame_id)] = axisangle
            outputs[("translation", 0, frame_id)] = translation
            outputs[("cam_T_cam", 0, frame_id)] = (
                transformation_from_parameters(
                    axisangle[:, 0],
                    translation[:, 0],
                    invert=frame_id < 0,
                )
            )
            outputs[("cam_T_cam", frame_id, 0)] = (
                transformation_from_parameters(
                    axisangle[:, 0],
                    translation[:, 0],
                    invert=frame_id > 0,
                )
            )

        return outputs

    def val(self):
        self.set_eval()
        predicted_disparities = []

        with torch.no_grad():
            for data in self.val_loader:
                input_color = data[("color", 0, 0)].to(self.device)
                output = self.models["depth"](
                    self.models["encoder"](input_color))
                disparity, _ = disp_to_depth(
                    output[("disp", 0)],
                    self.opt.min_depth,
                    self.opt.max_depth,
                )
                predicted_disparities.append(
                    disparity.cpu()[:, 0].numpy())

        predicted_disparities = np.concatenate(predicted_disparities)
        print(predicted_disparities.shape)
        mean_errors = self.evaluate_disparities(predicted_disparities)

        metric_indices = {
            "abs": 0,
            "sq": 1,
            "rm": 2,
            "rmlog": 3,
        }
        improved = False
        for metric_name, metric_index in metric_indices.items():
            metric_value = mean_errors[metric_index]
            if metric_value < self.best_metrics[metric_name]:
                self.save_model_best(metric_name, mean_errors)
                print(mean_errors)
                self.best_metrics[metric_name] = metric_value
                improved = True

        if improved:
            self.update_ema_model()

        self.set_train()

    def evaluate_disparities(self, predicted_disparities):
        errors = []

        for index, predicted_disparity in enumerate(
                predicted_disparities):
            gt_depth = self.gt_depths[index]
            gt_height, gt_width = gt_depth.shape[:2]
            predicted_disparity = cv2.resize(
                predicted_disparity, (gt_width, gt_height))
            predicted_depth = 1 / predicted_disparity

            mask = np.logical_and(
                gt_depth > self.opt.min_depth,
                gt_depth < self.opt.max_depth,
            )
            gt_depth = gt_depth[mask]
            predicted_depth = predicted_depth[mask]
            predicted_depth *= self.opt.pred_depth_scale_factor

            if not self.opt.disable_median_scaling:
                predicted_depth *= (
                    np.median(gt_depth) / np.median(predicted_depth))

            predicted_depth = np.clip(
                predicted_depth,
                self.opt.min_depth,
                self.opt.max_depth,
            )
            errors.append(compute_errors(gt_depth, predicted_depth))

        return np.array(errors).mean(0)

    def generate_images_pred(
        self,
        inputs,
        outputs,
        frozen_source_outputs,
        frozen_pose_outputs,
    ):
        target_image = inputs[("color", 0, 0)]

        for scale in self.opt.scales:
            target_disparity = F.interpolate(
                outputs[("disp", scale)],
                [self.opt.height, self.opt.width],
                mode="bilinear",
                align_corners=True,
            )
            _, target_depth = disp_to_depth(
                target_disparity,
                self.opt.min_depth,
                self.opt.max_depth,
            )
            outputs[("depth", 0, scale)] = target_depth

            for frame_id in self.opt.frame_ids[1:]:
                with torch.no_grad():
                    source_disparity = F.interpolate(
                        frozen_source_outputs[frame_id][("disp", scale)],
                        [self.opt.height, self.opt.width],
                        mode="bilinear",
                        align_corners=True,
                    )
                    _, source_depth = disp_to_depth(
                        source_disparity,
                        self.opt.min_depth,
                        self.opt.max_depth,
                    )
                    forward_grid = self.project_to_image(
                        inputs,
                        source_depth,
                        frozen_pose_outputs[
                            ("cam_T_cam", frame_id, 0)],
                    )
                    forward_image = F.grid_sample(
                        target_image,
                        forward_grid,
                        padding_mode="border",
                        align_corners=True,
                    )
                    forward_image_stm = self.structure_transplant(
                        inputs[("color", frame_id, 0)],
                        forward_image.detach(),
                    )

                backward_grid = self.project_to_image(
                    inputs,
                    target_depth,
                    outputs[("cam_T_cam", 0, frame_id)],
                )
                cycle_image = F.grid_sample(
                    forward_image_stm,
                    backward_grid,
                    padding_mode="border",
                    align_corners=True,
                )

                outputs[("color_forward", frame_id, scale)] = forward_image
                outputs[("color_forward_stm", frame_id, scale)] = (
                    forward_image_stm)
                outputs[("cycle", frame_id, scale)] = cycle_image
                outputs[("sample", frame_id, scale)] = backward_grid

    def project_to_image(self, inputs, depth, transform):
        camera_points = self.backproject_depth[0](
            depth, inputs[("inv_K", 0)])
        return self.project_3d[0](
            camera_points, inputs[("K", 0)], transform)

    def structure_transplant(self, source_image, warped_image):
        source_fft = torch.fft.fftn(source_image.to(torch.complex64))
        warped_fft = torch.fft.fftn(warped_image.to(torch.complex64))

        source_amplitude = torch.abs(source_fft)
        source_phase = torch.angle(source_fft)
        warped_amplitude = torch.abs(warped_fft)
        mixed_amplitude = (
            (1 - self.stm_amplitude_mix) * source_amplitude
            + self.stm_amplitude_mix * warped_amplitude
        )

        transplanted_fft = torch.polar(mixed_amplitude, source_phase)
        transplanted = torch.fft.ifftn(transplanted_fft).real.float()
        valid = (transplanted > 0) & (transplanted < 1)
        return torch.where(
            valid, transplanted, torch.zeros_like(transplanted))

    def compute_reprojection_loss(self, prediction, target):
        l1_loss = torch.abs(target - prediction).mean(1, True)
        if self.opt.no_ssim:
            return l1_loss

        ssim_loss = self.ssim(prediction, target).mean(1, True)
        return 0.85 * ssim_loss + 0.15 * l1_loss

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0
        target_image = inputs[("color", 0, 0)]

        for scale in self.opt.scales:
            reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                reprojection_losses.append(
                    self.compute_reprojection_loss(
                        outputs[("cycle", frame_id, scale)],
                        target_image,
                    )
                )

            reprojection_losses = torch.cat(reprojection_losses, 1)
            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(
                    1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            disparity = outputs[("disp", scale)]
            mean_disparity = disparity.mean(2, True).mean(3, True)
            normalized_disparity = disparity / (mean_disparity + 1e-7)
            smoothness_loss = get_smooth_loss(
                normalized_disparity,
                inputs[("color", 0, scale)],
            )

            scale_loss = reprojection_loss.mean()
            scale_loss += (
                self.opt.disparity_smoothness
                * smoothness_loss
                / (2 ** scale)
            )
            losses["loss/{}".format(scale)] = scale_loss
            total_loss += scale_loss

        losses["loss"] = total_loss / self.num_scales
        return losses

    @torch.no_grad()
    def update_ema_model(self):
        model_pairs = [
            ("frozen_encoder", "encoder"),
            ("frozen_depth", "depth"),
            ("frozen_pose_encoder", "pose_encoder"),
            ("frozen_pose", "pose"),
        ]
        for frozen_name, model_name in model_pairs:
            frozen_model = self.models[frozen_name]
            model = self.models[model_name]
            for frozen_parameter, parameter in zip(
                    frozen_model.parameters(), model.parameters()):
                frozen_parameter.mul_(self.ema).add_(
                    parameter, alpha=1 - self.ema)

    def copy_student_to_teacher(self):
        for model_name in ["encoder", "depth", "pose_encoder", "pose"]:
            self.models["frozen_" + model_name].load_state_dict(
                self.models[model_name].state_dict())

    def log_time(self, batch_idx, duration, loss):
        samples_per_second = self.opt.batch_size / duration
        time_so_far = time.time() - self.start_time
        if self.step > 0:
            time_left = (
                self.num_total_steps / self.step - 1.0) * time_so_far
        else:
            time_left = 0

        message = (
            "epoch {:>3} | batch {:>6} | examples/s: {:5.1f} | "
            "loss: {:.5f} | time elapsed: {} | time left: {}"
        )
        print(message.format(
            self.epoch,
            batch_idx,
            samples_per_second,
            loss,
            sec_to_hm_str(time_so_far),
            sec_to_hm_str(time_left),
        ))

    def log(self, mode, losses):
        writer = self.writers[mode]
        for name, value in losses.items():
            writer.add_scalar(name, value, self.step)

    def save_opts(self):
        models_dir = os.path.join(self.log_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, "opt.json"), "w") as file_handle:
            json.dump(self.opt.__dict__, file_handle, indent=2)

    def save_model(self):
        save_folder = os.path.join(
            self.log_path, "models", "weights_{}".format(self.epoch))
        self.save_checkpoint(save_folder)

    def save_model_best(self, metric, mean_errors):
        save_folder = os.path.join(
            self.log_path, "best_{}_model".format(metric))
        self.save_checkpoint(save_folder)

        result_path = os.path.join(save_folder, "result.txt")
        with open(result_path, "w") as file_handle:
            file_handle.write(
                "epoch: {} batch idx: {}\n".format(
                    self.epoch, self.step))
            file_handle.write(
                "  " + ("{:>8} | " * 7).format(
                    "abs_rel", "sq_rel", "rmse", "rmse_log",
                    "a1", "a2", "a3"))
            file_handle.write("\n")
            file_handle.write(
                ("&{: 8.3f}  " * 7).format(*mean_errors.tolist())
                + "\\\\")

    def save_checkpoint(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        for model_name, model in self.models.items():
            state_dict = model.state_dict()
            if model_name == "encoder":
                state_dict["height"] = self.opt.height
                state_dict["width"] = self.opt.width
                state_dict["use_stereo"] = self.opt.use_stereo
            torch.save(
                state_dict,
                os.path.join(save_folder, model_name + ".pth"),
            )

        torch.save(
            self.model_optimizer.state_dict(),
            os.path.join(save_folder, "adam.pth"),
        )

    def load_model(self):
        weights_folder = os.path.expanduser(
            self.opt.load_weights_folder)
        if not os.path.isdir(weights_folder):
            raise FileNotFoundError(
                "Cannot find folder {}".format(weights_folder))

        print("Loading model from folder {}".format(weights_folder))
        for model_name in self.opt.models_to_load:
            model_path = os.path.join(
                weights_folder, model_name + ".pth")
            print("Loading {} weights".format(model_name))

            model = self.models[model_name]
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_path)
            pretrained_dict = {
                key: value
                for key, value in pretrained_dict.items()
                if key in model_dict
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        self.copy_student_to_teacher()
        print("Adam is randomly initialized")
