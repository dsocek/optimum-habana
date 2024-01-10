# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import tempfile
from io import BytesIO
from pathlib import Path
from unittest import TestCase

import numpy as np
import requests
import torch

from transformers import (
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    AutoencoderKL,
    AutoencoderKLTemporalDecoder,
    UNet2DConditionModel,
    UNetSpatioTemporalConditionModel,
    EulerDiscreteScheduler
)
from diffusers.utils.testing_utils import (
    floats_tensor,
)

from parameterized import parameterized
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from transformers.testing_utils import slow

from optimum.habana import GaudiConfig
from optimum.habana.diffusers import (
    GaudiDDIMScheduler,
    GaudiEulerDiscreteScheduler,
    GaudiDiffusionPipeline,
    GaudiStableDiffusionLDM3DPipeline,
    GaudiStableDiffusionPipeline,
    GaudiStableDiffusionUpscalePipeline,
    GaudiStableVideoDiffusionPipeline
)
from optimum.habana.utils import set_seed


if os.environ.get("GAUDI2_CI", "0") == "1":
    THROUGHPUT_BASELINE_BF16 = 1.019
    THROUGHPUT_BASELINE_AUTOCAST = 0.389
else:
    THROUGHPUT_BASELINE_BF16 = 0.301
    THROUGHPUT_BASELINE_AUTOCAST = 0.108


class GaudiPipelineUtilsTester(TestCase):
    """
    Tests the features added on top of diffusers/pipeline_utils.py.
    """

    def test_use_hpu_graphs_raise_error_without_habana(self):
        with self.assertRaises(ValueError):
            _ = GaudiDiffusionPipeline(
                use_habana=False,
                use_hpu_graphs=True,
            )

    def test_gaudi_config_raise_error_without_habana(self):
        with self.assertRaises(ValueError):
            _ = GaudiDiffusionPipeline(
                use_habana=False,
                gaudi_config=GaudiConfig(),
            )

    def test_device(self):
        pipeline_1 = GaudiDiffusionPipeline(
            use_habana=True,
            gaudi_config=GaudiConfig(),
        )
        self.assertEqual(pipeline_1._device.type, "hpu")

        pipeline_2 = GaudiDiffusionPipeline(
            use_habana=False,
        )
        self.assertEqual(pipeline_2._device.type, "cpu")

    def test_gaudi_config_types(self):
        # gaudi_config is a string
        _ = GaudiDiffusionPipeline(
            use_habana=True,
            gaudi_config="Habana/stable-diffusion",
        )

        # gaudi_config is instantiated beforehand
        gaudi_config = GaudiConfig.from_pretrained("Habana/stable-diffusion")
        _ = GaudiDiffusionPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
        )

    def test_default(self):
        pipeline = GaudiDiffusionPipeline(
            use_habana=True,
            gaudi_config=GaudiConfig(),
        )

        self.assertTrue(hasattr(pipeline, "htcore"))

    def test_use_hpu_graphs(self):
        pipeline = GaudiDiffusionPipeline(
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(),
        )

        self.assertTrue(hasattr(pipeline, "ht"))
        self.assertTrue(hasattr(pipeline, "hpu_stream"))
        self.assertTrue(hasattr(pipeline, "cache"))

    def test_save_pretrained(self):
        model_name = "hf-internal-testing/tiny-stable-diffusion-torch"
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        pipeline = GaudiStableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            gaudi_config=GaudiConfig(),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir)
            self.assertTrue(Path(tmp_dir, "gaudi_config.json").is_file())


class GaudiStableDiffusionPipelineTester(TestCase):
    """
    Tests the StableDiffusionPipeline for Gaudi.
    """

    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=1,
            sample_size=32,
            time_cond_proj_dim=time_cond_proj_dim,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            norm_num_groups=2,
        )
        scheduler = GaudiDDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[4, 8],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=2,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=64,
            layer_norm_eps=1e-05,
            num_attention_heads=8,
            num_hidden_layers=3,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_ddim(self):
        device = "cpu"

        components = self.get_dummy_components()
        gaudi_config = GaudiConfig(use_torch_autocast=False)

        sd_pipe = GaudiStableDiffusionPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images[0]

        image_slice = image[-3:, -3:, -1]

        self.assertEqual(image.shape, (64, 64, 3))
        expected_slice = np.array([0.3203, 0.4555, 0.4711, 0.3505, 0.3973, 0.4650, 0.5137, 0.3392, 0.4045])

        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-2)

    def test_stable_diffusion_no_safety_checker(self):
        gaudi_config = GaudiConfig()
        scheduler = GaudiDDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        pipe = GaudiStableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            scheduler=scheduler,
            safety_checker=None,
            use_habana=True,
            gaudi_config=gaudi_config,
        )
        self.assertIsInstance(pipe, GaudiStableDiffusionPipeline)
        self.assertIsInstance(pipe.scheduler, GaudiDDIMScheduler)
        self.assertIsNone(pipe.safety_checker)

        image = pipe("example prompt", num_inference_steps=2).images[0]
        self.assertIsNotNone(image)

        # Check that there's no error when saving a pipeline with one of the models being None
        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe = GaudiStableDiffusionPipeline.from_pretrained(
                tmpdirname,
                use_habana=True,
                gaudi_config=tmpdirname,
            )

        # Sanity check that the pipeline still works
        self.assertIsNone(pipe.safety_checker)
        image = pipe("example prompt", num_inference_steps=2).images[0]
        self.assertIsNotNone(image)

    @parameterized.expand(["pil", "np", "latent"])
    def test_stable_diffusion_output_types(self, output_type):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        num_prompts = 2
        num_images_per_prompt = 3

        outputs = sd_pipe(
            num_prompts * [prompt],
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=2,
            output_type=output_type,
        )

        self.assertEqual(len(outputs.images), 2 * 3)
        # TODO: enable safety checker
        # if output_type == "latent":
        #     self.assertIsNone(outputs.nsfw_content_detected)
        # else:
        #     self.assertEqual(len(outputs.nsfw_content_detected), 2 * 3)

    # TODO: enable this test when PNDMScheduler is adapted to Gaudi
    # def test_stable_diffusion_negative_prompt(self):
    #     device = "cpu"  # ensure determinism for the device-dependent torch.Generator
    #     unet = self.dummy_cond_unet
    #     scheduler = PNDMScheduler(skip_prk_steps=True)
    #     vae = self.dummy_vae
    #     bert = self.dummy_text_encoder
    #     tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

    #     # make sure here that pndm scheduler skips prk
    #     sd_pipe = StableDiffusionPipeline(
    #         unet=unet,
    #         scheduler=scheduler,
    #         vae=vae,
    #         text_encoder=bert,
    #         tokenizer=tokenizer,
    #         safety_checker=None,
    #         feature_extractor=self.dummy_extractor,
    #     )
    #     sd_pipe = sd_pipe.to(device)
    #     sd_pipe.set_progress_bar_config(disable=None)

    #     prompt = "A painting of a squirrel eating a burger"
    #     negative_prompt = "french fries"
    #     generator = torch.Generator(device=device).manual_seed(0)
    #     output = sd_pipe(
    #         prompt,
    #         negative_prompt=negative_prompt,
    #         generator=generator,
    #         guidance_scale=6.0,
    #         num_inference_steps=2,
    #         output_type="np",
    #     )

    #     image = output.images
    #     image_slice = image[0, -3:, -3:, -1]

    #     assert image.shape == (1, 128, 128, 3)
    #     expected_slice = np.array([0.4851, 0.4617, 0.4765, 0.5127, 0.4845, 0.5153, 0.5141, 0.4886, 0.4719])
    #     assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_num_images_per_prompt(self):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # Test num_images_per_prompt=1 (default)
        images = sd_pipe(prompt, num_inference_steps=2, output_type="np").images

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, (64, 64, 3))

        # Test num_images_per_prompt=1 (default) for several prompts
        num_prompts = 3
        images = sd_pipe([prompt] * num_prompts, num_inference_steps=2, output_type="np").images

        self.assertEqual(len(images), num_prompts)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        images = sd_pipe(
            prompt, num_inference_steps=2, output_type="np", num_images_per_prompt=num_images_per_prompt
        ).images

        self.assertEqual(len(images), num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Test num_images_per_prompt for several prompts
        num_prompts = 2
        images = sd_pipe(
            [prompt] * num_prompts,
            num_inference_steps=2,
            output_type="np",
            num_images_per_prompt=num_images_per_prompt,
        ).images

        self.assertEqual(len(images), num_prompts * num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

    def test_stable_diffusion_batch_sizes(self):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # Test num_images > 1 where num_images is a divider of the total number of generated images
        batch_size = 3
        num_images_per_prompt = batch_size**2
        images = sd_pipe(
            prompt,
            num_inference_steps=2,
            output_type="np",
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        self.assertEqual(len(images), num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Same test for several prompts
        num_prompts = 3
        images = sd_pipe(
            [prompt] * num_prompts,
            num_inference_steps=2,
            output_type="np",
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        self.assertEqual(len(images), num_prompts * num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Test num_images when it is not a divider of the toal number of generated images for a single prompt
        num_images_per_prompt = 7
        images = sd_pipe(
            prompt,
            num_inference_steps=2,
            output_type="np",
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        self.assertEqual(len(images), num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Same test for several prompts
        num_prompts = 2
        images = sd_pipe(
            [prompt] * num_prompts,
            num_inference_steps=2,
            output_type="np",
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        self.assertEqual(len(images), num_prompts * num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

    def test_stable_diffusion_bf16(self):
        """Test that stable diffusion works with bf16"""
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device="cpu").manual_seed(0)
        image = sd_pipe([prompt], generator=generator, num_inference_steps=2, output_type="np").images[0]

        self.assertEqual(image.shape, (64, 64, 3))

    def test_stable_diffusion_default(self):
        components = self.get_dummy_components()

        sd_pipe = GaudiStableDiffusionPipeline(
            use_habana=True,
            gaudi_config="Habana/stable-diffusion",
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device="cpu").manual_seed(0)
        images = sd_pipe(
            [prompt] * 2,
            generator=generator,
            num_inference_steps=2,
            output_type="np",
            batch_size=3,
            num_images_per_prompt=5,
        ).images

        self.assertEqual(len(images), 10)
        self.assertEqual(images[-1].shape, (64, 64, 3))

    def test_stable_diffusion_hpu_graphs(self):
        components = self.get_dummy_components()

        sd_pipe = GaudiStableDiffusionPipeline(
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion",
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device="cpu").manual_seed(0)
        images = sd_pipe(
            [prompt] * 2,
            generator=generator,
            num_inference_steps=2,
            output_type="np",
            batch_size=3,
            num_images_per_prompt=5,
        ).images

        self.assertEqual(len(images), 10)
        self.assertEqual(images[-1].shape, (64, 64, 3))

    @slow
    def test_no_throughput_regression_bf16(self):
        prompts = [
            "An image of a squirrel in Picasso style",
            "High quality photo of an astronaut riding a horse in space",
        ]
        num_images_per_prompt = 11
        batch_size = 4
        model_name = "runwayml/stable-diffusion-v1-5"
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

        pipeline = GaudiStableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig.from_pretrained("Habana/stable-diffusion"),
            torch_dtype=torch.bfloat16,
        )
        set_seed(27)
        outputs = pipeline(
            prompt=prompts,
            num_images_per_prompt=num_images_per_prompt,
            batch_size=batch_size,
        )
        self.assertEqual(len(outputs.images), num_images_per_prompt * len(prompts))
        self.assertGreaterEqual(outputs.throughput, 0.95 * THROUGHPUT_BASELINE_BF16)

    @slow
    def test_no_throughput_regression_autocast(self):
        prompts = [
            "An image of a squirrel in Picasso style",
            "High quality photo of an astronaut riding a horse in space",
        ]
        num_images_per_prompt = 11
        batch_size = 4
        model_name = "stabilityai/stable-diffusion-2-1"
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

        pipeline = GaudiStableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig.from_pretrained("Habana/stable-diffusion-2"),
        )
        set_seed(27)
        outputs = pipeline(
            prompt=prompts,
            num_images_per_prompt=num_images_per_prompt,
            batch_size=batch_size,
            height=768,
            width=768,
        )
        self.assertEqual(len(outputs.images), num_images_per_prompt * len(prompts))
        self.assertGreaterEqual(outputs.throughput, 0.95 * THROUGHPUT_BASELINE_AUTOCAST)

    @slow
    def test_no_generation_regression(self):
        model_name = "CompVis/stable-diffusion-v1-4"
        # fp32
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        pipeline = GaudiStableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            safety_checker=None,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(use_torch_autocast=False),
        )
        set_seed(27)
        outputs = pipeline(
            prompt="An image of a squirrel in Picasso style",
            output_type="np",
        )

        if os.environ.get("GAUDI2_CI", "0") == "1":
            expected_slice = np.array(
                [
                    0.68306947,
                    0.6812112,
                    0.67309505,
                    0.70057267,
                    0.6582885,
                    0.6325019,
                    0.6708976,
                    0.6226433,
                    0.58038336,
                ]
            )
        else:
            expected_slice = np.array(
                [0.70760196, 0.7136303, 0.7000798, 0.714934, 0.6776865, 0.6800843, 0.6923707, 0.6653969, 0.6408076]
            )
        image = outputs.images[0]

        self.assertEqual(image.shape, (512, 512, 3))
        self.assertLess(np.abs(expected_slice - image[-3:, -3:, -1].flatten()).max(), 5e-3)

    @slow
    def test_no_generation_regression_ldm3d(self):
        model_name = "Intel/ldm3d-4c"
        # fp32
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        pipeline = GaudiStableDiffusionLDM3DPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            safety_checker=None,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(),
        )
        set_seed(27)
        outputs = pipeline(
            prompt="An image of a squirrel in Picasso style",
            output_type="np",
        )

        if os.environ.get("GAUDI2_CI", "0") == "1":
            expected_slice_rgb = np.array(
                [
                    0.2099357,
                    0.16664368,
                    0.08352646,
                    0.20643419,
                    0.16748399,
                    0.08781305,
                    0.21379063,
                    0.19943115,
                    0.04389626,
                ]
            )
            expected_slice_depth = np.array(
                [
                    0.68369114,
                    0.6827824,
                    0.6852779,
                    0.6836072,
                    0.6888298,
                    0.6895473,
                    0.6853674,
                    0.67561126,
                    0.660434,
                ]
            )
        else:
            expected_slice_rgb = np.array([0.7083766, 1.0, 1.0, 0.70610344, 0.9867363, 1.0, 0.7214538, 1.0, 1.0])
            expected_slice_depth = np.array(
                [
                    0.919621,
                    0.92072034,
                    0.9184986,
                    0.91994286,
                    0.9242079,
                    0.93387043,
                    0.92345214,
                    0.93558526,
                    0.9223714,
                ]
            )
        rgb = outputs.rgb[0]
        depth = outputs.depth[0]

        self.assertEqual(rgb.shape, (512, 512, 3))
        self.assertEqual(depth.shape, (512, 512, 1))
        self.assertLess(np.abs(expected_slice_rgb - rgb[-3:, -3:, -1].flatten()).max(), 5e-3)
        self.assertLess(np.abs(expected_slice_depth - depth[-3:, -3:, -1].flatten()).max(), 5e-3)

    @slow
    def test_no_generation_regression_upscale(self):
        model_name = "stabilityai/stable-diffusion-x4-upscaler"
        # fp32
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        pipeline = GaudiStableDiffusionUpscalePipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(use_torch_autocast=False),
        )
        set_seed(27)

        url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
        response = requests.get(url)
        low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
        low_res_img = low_res_img.resize((128, 128))
        prompt = "a white cat"
        upscaled_image = pipeline(prompt=prompt, image=low_res_img, output_type="np").images[0]
        if os.environ.get("GAUDI2_CI", "0") == "1":
            expected_slice = np.array(
                [
                    0.16527882,
                    0.161616,
                    0.15665859,
                    0.1660901,
                    0.1594379,
                    0.14936888,
                    0.1578255,
                    0.15342498,
                    0.14590919,
                ]
            )
        else:
            expected_slice = np.array(
                [
                    0.1652787,
                    0.16161594,
                    0.15665877,
                    0.16608998,
                    0.1594378,
                    0.14936894,
                    0.15782538,
                    0.15342498,
                    0.14590913,
                ]
            )
        self.assertEqual(upscaled_image.shape, (512, 512, 3))
        self.assertLess(np.abs(expected_slice - upscaled_image[-3:, -3:, -1].flatten()).max(), 5e-3)


class GaudiStableVideoDiffusionPipelineTester(TestCase):
    """
    Tests the StableVideoDiffusionPipeline for Gaudi.
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.24.0-release/tests/pipelines/stable_video_diffusion/test_stable_video_diffusion.py
    """

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNetSpatioTemporalConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=8,
            out_channels=4,
            down_block_types=(
                "CrossAttnDownBlockSpatioTemporal",
                "DownBlockSpatioTemporal",
            ),
            up_block_types=("UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal"),
            cross_attention_dim=32,
            num_attention_heads=8,
            projection_class_embeddings_input_dim=96,
            addition_time_embed_dim=32,
        )
        scheduler = GaudiEulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            interpolation_type="linear",
            num_train_timesteps=1000,
            prediction_type="v_prediction",
            sigma_max=700.0,
            sigma_min=0.002,
            steps_offset=1,
            timestep_spacing="leading",
            timestep_type="continuous",
            trained_betas=None,
            use_karras_sigmas=True,
        )

        torch.manual_seed(0)
        vae = AutoencoderKLTemporalDecoder(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            latent_channels=4,
        )

        torch.manual_seed(0)
        config = CLIPVisionConfig(
            hidden_size=32,
            projection_dim=32,
            num_hidden_layers=5,
            num_attention_heads=4,
            image_size=32,
            intermediate_size=37,
            patch_size=1,
        )
        image_encoder = CLIPVisionModelWithProjection(config)

        torch.manual_seed(0)
        feature_extractor = CLIPImageProcessor(crop_size=32, size=32)
        components = {
            "unet": unet,
            "image_encoder": image_encoder,
            "scheduler": scheduler,
            "vae": vae,
            "feature_extractor": feature_extractor,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        generator = torch.Generator(device="cpu").manual_seed(seed)

        image = floats_tensor((1, 3, 32, 32), rng=random.Random(0)).to(device)
        inputs = {
            "generator": generator,
            "image": image,
            "num_inference_steps": 2,
            "output_type": "pt",
            "min_guidance_scale": 1.0,
            "max_guidance_scale": 2.5,
            "num_frames": 2,
            "height": 32,
            "width": 32,
        }
        return inputs

    def test_stable_video_diffusion_single_video(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        #gaudi_config = GaudiConfig(use_torch_autocast=False)
        sd_pipe = GaudiStableVideoDiffusionPipeline(
            # use_habana=True,
            # gaudi_config=gaudi_config,
            **components
        )
        for component in sd_pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        outputs = sd_pipe(**self.get_dummy_inputs(device),).frames
        image = outputs[0]
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(len(outputs), 1)
        self.assertEqual(image.shape, (2, 3, 32, 32))

        expected_slice = np.array([0.5910, 0.5797, 0.5521, 0.6628, 0.6212, 0.6422, 0.5681, 0.5232, 0.5343])

        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-2)

    def test_stable_video_diffusion_batch_sizes(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig(use_torch_autocast=False)
        sd_pipe = GaudiStableVideoDiffusionPipeline(
            # use_habana=True,
            # gaudi_config=gaudi_config,
            **components
        )
        for component in sd_pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        # Test num_videos_per_prompt > 1 where num_videos is divisible by batch size
        batch_size = 3
        num_videos_per_prompt = batch_size**2

        outputs = sd_pipe(
            **self.get_dummy_inputs(device),
            num_videos_per_prompt=num_videos_per_prompt,
            batch_size=batch_size,
        ).frames
        image = outputs[0]
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(len(outputs), num_videos_per_prompt)
        self.assertEqual(image.shape, (2, 3, 32, 32))
        # output of diffusers cpu model for 3 video generations (i.e., batch size)
        # this output is identical for batch size=3
        expected_slice = np.array([0.5899, 0.5799, 0.5515, 0.6655, 0.6219, 0.6437, 0.5732, 0.5227, 0.5328])
        # output of diffusers cpu model for 9 video generations (i.e., total videos)
        # the output of model changes based on number of generations
        # this output is within 1e-02
        # expected_slice = np.array([0.5877, 0.5823, 0.5585, 0.6662, 0.6258, 0.6459, 0.5753, 0.5325, 0.5387])

        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-2)

        # Test num_videos_per_prompt > 1 where num_videos is not divisible by batch size
        batch_size = 3
        num_videos_per_prompt = 7

        outputs = sd_pipe(
            **self.get_dummy_inputs(device),
            num_videos_per_prompt=num_videos_per_prompt,
            batch_size=batch_size
        ).frames
        image = outputs[0]
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(len(outputs), num_videos_per_prompt)
        self.assertEqual(image.shape, (2, 3, 32, 32))

        # output of diffusers cpu model for 3 video generations (i.e., batch size)
        # this output is identical for batch size=3
        expected_slice = np.array([0.5899, 0.5799, 0.5515, 0.6655, 0.6219, 0.6437, 0.5732, 0.5227, 0.5328])
        # output of diffusers cpu model for 7 video generations (i.e., total videos)
        # the output of model changes based on number of generations
        # this output is within 1e-02
        # expected_slice = np.array([0.5946, 0.5807, 0.5518, 0.6601, 0.6246, 0.6469, 0.5651, 0.5270, 0.5334])

        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-2)

        # Same test without classifier-free guidance
        inputs = self.get_dummy_inputs(device)
        inputs["max_guidance_scale"] = 1.0
        outputs = sd_pipe(
            **inputs,
            num_videos_per_prompt=num_videos_per_prompt,
            batch_size=batch_size
        ).frames
        image = outputs[0]
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(len(outputs), num_videos_per_prompt)
        self.assertEqual(image.shape, (2, 3, 32, 32))

        # output of diffusers cpu model for 7 video generations (i.e., total videos)
        # this output is identical when we don't use classifier-free guidance
        expected_slice = np.array([0.5879, 0.5690, 0.5291, 0.6601, 0.6250, 0.6384, 0.5533, 0.5244, 0.5091])

        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-2)

        # Test with several image prompts
        batch_size = 3
        num_images = 3
        num_videos_per_prompt = 2

        inputs = self.get_dummy_inputs(device)
        inputs["image"] = inputs["image"].repeat(num_images, 1, 1, 1)
        outputs = sd_pipe(
            **inputs,
            num_videos_per_prompt=num_videos_per_prompt,
            batch_size=batch_size
        ).frames
        image = outputs[0]
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(len(outputs), num_images * num_videos_per_prompt)
        self.assertEqual(image.shape, (2, 3, 32, 32))

        # output of diffusers cpu model for 3 imput images with 1 video per prompt
        # this output is identical for batch size=3
        expected_slice = np.array([0.5026, 0.5873, 0.5591, 0.6236, 0.6200, 0.6158, 0.5214, 0.5603, 0.5245])
        # the output of model changes based on number of generations
        # this output is within 2e-02
        # expected_slice = np.array([0.5135, 0.5769, 0.5805, 0.6354, 0.6215, 0.6266, 0.5182, 0.5513, 0.5291])

        # TODO: Determine expected input
        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-2)

