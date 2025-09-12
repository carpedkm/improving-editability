# Copyright 2023 PixArt-Alpha Authors and The HuggingFace Team. All rights reserved.
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

import html
import inspect
import re
import urllib.parse as ul
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer

import sys
sys.path.append('../')

from diffusers.image_processor import PipelineImageInput, PixArtImageProcessor, VaeImageProcessor
from diffusers.models import AutoencoderKL, Transformer2DModel

from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import (
    BACKENDS_MAPPING,
    deprecate,
    is_bs4_available,
    is_ftfy_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
import torch.nn as nn

# from custom_diffusers.transformer_2d_custom import Transformer2DModelCustom


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import PixArtAlphaInpaintPipeline

        >>> # You can replace the checkpoint id with "PixArt-alpha/PixArt-XL-2-512x512" too.
        >>> pipe = PixArtAlphaInpaintPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
        >>> # Enable memory optimizations.
        >>> pipe.enable_model_cpu_offload()

        >>> prompt = ""
        >>> image = Image.open('')
        >>> image = pipe(prompt,
                        image=image,
                        mask_image=mask_image,
                        strength=1.0).images[0]
        ```
"""

ASPECT_RATIO_1024_BIN = {
    "0.25": [512.0, 2048.0],
    "0.28": [512.0, 1856.0],
    "0.32": [576.0, 1792.0],
    "0.33": [576.0, 1728.0],
    "0.35": [576.0, 1664.0],
    "0.4": [640.0, 1600.0],
    "0.42": [640.0, 1536.0],
    "0.48": [704.0, 1472.0],
    "0.5": [704.0, 1408.0],
    "0.52": [704.0, 1344.0],
    "0.57": [768.0, 1344.0],
    "0.6": [768.0, 1280.0],
    "0.68": [832.0, 1216.0],
    "0.72": [832.0, 1152.0],
    "0.78": [896.0, 1152.0],
    "0.82": [896.0, 1088.0],
    "0.88": [960.0, 1088.0],
    "0.94": [960.0, 1024.0],
    "1.0": [1024.0, 1024.0],
    "1.07": [1024.0, 960.0],
    "1.13": [1088.0, 960.0],
    "1.21": [1088.0, 896.0],
    "1.29": [1152.0, 896.0],
    "1.38": [1152.0, 832.0],
    "1.46": [1216.0, 832.0],
    "1.67": [1280.0, 768.0],
    "1.75": [1344.0, 768.0],
    "2.0": [1408.0, 704.0],
    "2.09": [1472.0, 704.0],
    "2.4": [1536.0, 640.0],
    "2.5": [1600.0, 640.0],
    "3.0": [1728.0, 576.0],
    "4.0": [2048.0, 512.0],
}

ASPECT_RATIO_512_BIN = {
    "0.25": [256.0, 1024.0],
    "0.28": [256.0, 928.0],
    "0.32": [288.0, 896.0],
    "0.33": [288.0, 864.0],
    "0.35": [288.0, 832.0],
    "0.4": [320.0, 800.0],
    "0.42": [320.0, 768.0],
    "0.48": [352.0, 736.0],
    "0.5": [352.0, 704.0],
    "0.52": [352.0, 672.0],
    "0.57": [384.0, 672.0],
    "0.6": [384.0, 640.0],
    "0.68": [416.0, 608.0],
    "0.72": [416.0, 576.0],
    "0.78": [448.0, 576.0],
    "0.82": [448.0, 544.0],
    "0.88": [480.0, 544.0],
    "0.94": [480.0, 512.0],
    "1.0": [512.0, 512.0],
    "1.07": [512.0, 480.0],
    "1.13": [544.0, 480.0],
    "1.21": [544.0, 448.0],
    "1.29": [576.0, 448.0],
    "1.38": [576.0, 416.0],
    "1.46": [608.0, 416.0],
    "1.67": [640.0, 384.0],
    "1.75": [672.0, 384.0],
    "2.0": [704.0, 352.0],
    "2.09": [736.0, 352.0],
    "2.4": [768.0, 320.0],
    "2.5": [800.0, 320.0],
    "3.0": [864.0, 288.0],
    "4.0": [1024.0, 256.0],
}


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

class LayerStore:
    # store prompt_embedding, prompt_mask, negative_embedding, negative_mask, attention_mask (mask_image)
    def __init__(self):
        self.prompt_embedding = {}
        self.prompt_mask = {}
        self.negative_embedding = {}
        self.negative_mask = {}
        self.attention_mask = {}
        self.mask = {}
    def _store_layer(self, prompt_embedding, prompt_mask, negative_embedding, negative_mask, attention_mask, mask):
        stored_mem_len = self._len_of_stored_layers()
        self.prompt_embedding[stored_mem_len] = prompt_embedding
        self.prompt_mask[stored_mem_len] = prompt_mask
        self.negative_embedding[stored_mem_len] = negative_embedding
        self.negative_mask[stored_mem_len] = negative_mask
        self.attention_mask[stored_mem_len] = attention_mask
        self.mask[stored_mem_len] = mask
        
    def _len_of_stored_layers(self):
        return len(self.prompt_embedding)

    def _get_layer(self, index): # return prompt_embedding, prompt_mask, negative_embedding, negative_mask, attention_mask
        return self.prompt_embedding[index], self.prompt_mask[index], self.negative_embedding[index], self.negative_mask[index], self.attention_mask[index], self.mask[index]

    def _get_all_layers(self):
        # convert from dict to list before return
        tmp_prompt_embedding = list(self.prompt_embedding.values())
        tmp_prompt_mask = list(self.prompt_mask.values())
        tmp_negative_embedding = list(self.negative_embedding.values())
        tmp_negative_mask = list(self.negative_mask.values())
        tmp_attention_mask = list(self.attention_mask.values())
        tmp_mask = list(self.mask.values())
        
        return tmp_prompt_embedding, tmp_prompt_mask, tmp_negative_embedding, tmp_negative_mask, tmp_attention_mask, tmp_mask
    def _empty_all_layers(self):
        self.prompt_embedding = {}
        self.prompt_mask = {}
        self.negative_embedding = {}
        self.negative_mask = {}
        self.attention_mask = {}
        self.mask = {}
    def _empty_last_layer(self):
        stored_mem_len = self._len_of_stored_layers()
        self.prompt_embedding.pop(stored_mem_len-1)
        self.prompt_mask.pop(stored_mem_len-1)
        self.negative_embedding.pop(stored_mem_len-1)
        self.negative_mask.pop(stored_mem_len-1)
        self.attention_mask.pop(stored_mem_len-1)
        self.mask.pop(stored_mem_len-1)
    def _empty_prev_layer(self):
        stored_mem_len = self._len_of_stored_layers()
        self.prompt_embedding.pop(stored_mem_len-2)
        self.prompt_mask.pop(stored_mem_len-2)
        self.negative_embedding.pop(stored_mem_len-2)
        self.negative_mask.pop(stored_mem_len-2)
        self.attention_mask.pop(stored_mem_len-2)
        self.mask.pop(stored_mem_len-2)
    
    

class PixArtAlphaInpaintLMPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using PixArt-Alpha.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. PixArt-Alpha uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`Transformer2DModelCustom`]):
            A text conditioned `Transformer2DModelCustom` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + "\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa

    _optional_components = ["tokenizer", "text_encoder"]
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKL,
        transformer: Transformer2DModel,
        scheduler: DPMSolverMultistepScheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )
        # self.tokenizer = tokenizer
        # self.text_encoder = text_encoder
        # self.vae = vae
        # self.transformer = transformer
        # self.scheduler = scheduler

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        self.layerstore = LayerStore()
        self.prev_prompt = None
        
        self.prev_prompt_2nd = None
        self.prev_prompt_embeds_2nd = None
        self.prev_prompt_attention_mask_2nd = None

    # Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/utils.py
    def mask_text_embeddings(self, emb, mask):
        if emb.shape[0] == 1:
            keep_index = mask.sum().item()
            return emb[:, :, :keep_index, :], keep_index
        else:
            masked_feature = emb * mask[:, None, :, None]
            return masked_feature, emb.shape[2]

    # Adapted from diffusers.pipelines.deepfloyd_if.pipeline_if.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        clean_caption: bool = False,
        **kwargs,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                PixArt-Alpha, this should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Alpha, it's should be the embeddings of the ""
                string.
            clean_caption (bool, defaults to `False`):
                If `True`, the function will preprocess and clean the provided caption before encoding.
        """

        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

        if device is None:
            device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # See Section 3.1. of the paper.
        max_length = 120

        if prompt_embeds is None:
            prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )

            prompt_attention_mask = text_inputs.attention_mask
            prompt_attention_mask = prompt_attention_mask.to(device)

            prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens = [negative_prompt] * batch_size
            uncond_tokens = self._text_preprocessing(uncond_tokens, clean_caption=clean_caption)
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            negative_prompt_attention_mask = uncond_input.attention_mask
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device), attention_mask=negative_prompt_attention_mask
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_steps,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing
    def _text_preprocessing(self, text, clean_caption=False):
        if clean_caption and not is_bs4_available():
            logger.warn(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
            logger.warn("Setting `clean_caption` to False...")
            clean_caption = False

        if clean_caption and not is_ftfy_available():
            logger.warn(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
            logger.warn("Setting `clean_caption` to False...")
            clean_caption = False

        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            if clean_caption:
                text = self._clean_caption(text)
                text = self._clean_caption(text)
            else:
                text = text.lower().strip()
            return text

        return [process(t) for t in text]

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption
    def _clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",
            # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = ftfy.fix_text(caption)
        caption = html.unescape(html.unescape(caption))

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_image_latents=True,
        inpaint=False,  # Include inpaint flag here
    ):
        # Define the shape of the latents tensor
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)

        # Initialize `image_latents` only if inpainting or when image is provided
        image_latents = None
        
        # Handle cases when inpainting or image is provided, and we need to encode the image
        if inpaint and image is not None and return_image_latents:
            # Ensure the image is on the right device and in the correct data type
            image = image.to(device=device, dtype=dtype)

            # Check if the image is already in latent space (has 4 channels), otherwise encode it
            if image.shape[1] == 4:
                image_latents = image  # Use the image as-is (it's already in latent space)
            else:
                image_latents = self._encode_vae_image(image=image, generator=generator)  # Encode the image to latents

            # Repeat image_latents for the batch size if needed
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        # If no latents are provided, we need to generate latents (using noise or encoded image)
        if latents is None:
            # Generate random noise in the shape of the latents
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

            # If strength is 1 (max strength), we use pure noise; otherwise, blend noise with the encoded image latents
            if is_strength_max:
                latents = noise
            else:
                # Mix the image latents with noise, based on the timestep (for inpainting or img2img)
                latents = self.scheduler.add_noise(image_latents, noise, timestep)
            
            # Scale the latents with the scheduler's initial noise sigma if using pure noise (max strength)
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            # Use the provided latents (convert to the correct device)
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        # If inpainting, we need to return both latents and image_latents for the mixing step
        if inpaint:
            return latents, noise, image_latents
        else:
            # For standard diffusion, we don't need image_latents
            return latents, noise, None
    def erode_tensor(self, tensor_mask, kernel_size=3):
        # Apply erosion using a convolutional filter (structuring element)
        # Tensor mask shape: [1, 1, 128, 128]
        if kernel_size % 2 == 0:
            padding = 0  # For even kernel sizes (like 2x2), set padding to 0
        else:
            padding = kernel_size // 2  # For odd kernel sizes, use padding to keep the size
        # Create a square kernel (structuring element) for erosion
        kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float16).to(tensor_mask.device)
        
        # Perform 2D convolution to simulate erosion
        eroded_mask = F.conv2d(tensor_mask, kernel, padding=padding)
        
        # Threshold the result to make the mask binary again
        eroded_mask = (eroded_mask == (kernel.numel() / 3)).half()
        
        return eroded_mask

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    def prepare_mask_latents(
        self, mask, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # Resize the mask to match latent resolution
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        if mask.shape[0] < batch_size:
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        return mask

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start
    def _prev_prompt_store(self, prompt, prompt_embeds, prompt_attention_mask):
        if self.prev_prompt is not None:
            self._new_generation_storage(self.prev_prompt, self.prev_prompt_embeds, self.prev_prompt_attention_mask)
        self.prev_prompt = prompt
        self.prev_prompt_embeds = prompt_embeds
        self.prev_prompt_attention_mask = prompt_attention_mask
        
        
    def _new_generation_storage(self, prompt, prompt_embeds, prompt_attention_mask):
        self.prev_prompt_2nd = prompt
        self.prev_prompt_embeds_2nd = prompt_embeds
        self.prev_prompt_attention_mask_2nd = prompt_attention_mask
    
    def _select_no_new_generation(self):
        self.prev_prompt = self.prev_prompt_2nd
        self.prev_prompt_embeds = self.prev_prompt_embeds_2nd
        self.prev_prompt_attention_mask = self.prev_prompt_attention_mask_2nd
            
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        strength: float = 1.0,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        subject_token_idx: list = None,
        sigma : float = 0.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        use_resolution_binning: bool = True,
        inpaint: bool = True,
        latent_memory: dict = None,
        vanilla_ratio: float = 0.0,
        cattn_masking: bool = False,
        multi_query_disentanglement: bool = False,
        object_gen: bool = False,
        utilize_cache: bool = False,
        new_generation: bool = True,
        remove_prev: bool = False,
        real_image: bool = False,
        # direct_inpaint: bool = False,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Main inference function to handle generation or inpainting.
        """
        import time
        start_time = time.time()
        # 1. Set default height and width based on transformer sample size
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor

        # 2. Check inputs for validity and perform resolution binning if necessary
        if use_resolution_binning:
            aspect_ratio_bin = ASPECT_RATIO_1024_BIN if self.transformer.config.sample_size == 128 else ASPECT_RATIO_512_BIN
            orig_height, orig_width = height, width
            height, width = self.image_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

        if remove_prev is True:
            self.latent0 = latent_memory[1]
            self.latent2 = latent_memory[0]
            # get all layer info
            prev_prompt_embeds_list, prev_prompt_attention_mask_list, _, _, prev_attention_mask_list, mask_list = self.layerstore._get_all_layers()
            # prompt_embeds = prev_prompt_embeds_list[-1]
            prompt_mask = prev_prompt_attention_mask_list[-1]
            mask2 = mask_list[-1]
            
            
        # if remove_prev is not True:
        self.check_inputs(
                prompt,
                height,
                width,
                negative_prompt,
                callback_steps,
                prompt_embeds,
                negative_prompt_embeds,
                prompt_attention_mask,
                negative_prompt_attention_mask,
            )

        # 3. Set batch size based on prompt or prompt_embeds
        batch_size = 1 if isinstance(prompt, str) else len(prompt) if isinstance(prompt, list) else prompt_embeds.shape[0]
        device = self._execution_device
        
        do_classifier_free_guidance = guidance_scale > 1.0
        

        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
                prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                clean_caption=clean_caption,
            )

        # Concatenate negative and positive embeddings if guidance scale > 1.0
        if guidance_scale > 1.0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )

        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        is_strength_max = strength == 1.0

        # 6. Handle image input for inpainting vs. standard generation
        if inpaint:
            if image is None:
                pass
        else:
                init_image = None

        # 7. Prepare latents
        # if remove_prev is not True:
        latents, noise, image_latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                self.transformer.config.in_channels,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents=latents,
                image=None, # FIXME : it was init_image
                timestep=latent_timestep,
                is_strength_max=is_strength_max,
                inpaint=False, # FIXME it was inpaint
            )
        # if inpaint:
        #     mask_condition = self.mask_processor.preprocess(mask_image, height=height, width=width)


        # 8. Prepare mask if inpainting is used
        if inpaint:
            mask_condition = self.mask_processor.preprocess(mask_image, height=height, width=width).to(dtype=prompt_embeds.dtype, device=device)
            mask = self.prepare_mask_latents(
                    mask_condition,
                    batch_size * num_images_per_prompt,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    do_classifier_free_guidance,
                )
        else:
            mask = None

        # 9. Prepare micro-conditions (resolution, aspect ratio)
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if self.transformer.config.sample_size == 128:
            resolution = torch.tensor([height, width]).repeat(batch_size * num_images_per_prompt, 1).to(device=device)
            aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size * num_images_per_prompt, 1).to(device=device)
            if guidance_scale > 1.0:
                resolution = torch.cat([resolution, resolution], dim=0)
                aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)
            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        # 10. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        
        if latent_memory is None:
            latent_memory = {}
        latent_memory_new = {}
        # if direct_inpaint is True:
        #     # Get latent_memory of the given image by forward/backward pass
        #     with torch.no_grad():
        #         # Forward and Backward diffusion process to get latent memory
        #         # FILL IN
        
        # PREPROCESS MASK FOR NOISECOLLAGE STYLE MASKED CROSS ATTENTION
        if mask is not None:
            mask_orig = mask.clone()
            mask = mask[0,...]
            pool_64 = nn.AdaptiveAvgPool2d((64, 64)) # FIXME : 64 is hardcoded -> 32 (for 512x512)
            mask = pool_64(mask)
            # pool_32 = nn.AdaptiveAvgPool2d((32, 32)) # FIXME : 64 is hardcoded -> 32 (for 512x512)
            # mask = pool_32(mask)
            mask = mask.squeeze()
        
        if new_generation is True:
            pass
        else:
            self._select_no_new_generation()
            self.layerstore._empty_last_layer()
            

        decay_ratio = 0.
        if remove_prev is True:
            vanilla_ratio = 0.5
            self.layerstore._empty_prev_layer() # the layer before the last layer -- delete
            
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Ensure latents are duplicated correctly if guidance_scale > 1.0
                if remove_prev is not True :
                    latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    prev_attention_masks = None
                    # Expand current timestep for the entire batch
                    current_timestep = t.expand(latent_model_input.shape[0])
                elif remove_prev is True and i >= len(timesteps) - int(num_inference_steps * vanilla_ratio):
                    latent_model_input = latents #torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    prev_attention_masks = None
                    # Expand current timestep for the entire batch
                    current_timestep = t.expand(latent_model_input.shape[0])
                # elif remove_prev is not True:
                #     latent_model_input = torch.cat([self.latent2[int(t.detach().cpu().numpy())]] * 2) if guidance_scale > 1.0 else self.latent2[int(t.detach().cpu().numpy())]
                #     latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                #     prev_attention_masks = None
                #     current_timestep = t.expand(latent_model_input.shape[0])
                # SAVING CACHE (SOLE OBJECT GENERATION FOR QUERY CACHING)
                if remove_prev is not True or (remove_prev is True and i >= len(timesteps) - int(num_inference_steps * vanilla_ratio)):
                        # Predict the noise
                        if cattn_masking is True:
                            if multi_query_disentanglement is True: 
                                if self.layerstore._len_of_stored_layers() == 0:
                                    prev_prompt_embeds = prompt_embeds.to(device)
                                    prev_prompt_attention_mask = prompt_attention_mask.to(device)
                                else:  
                                    # Get all the prev_prompt_embeds_list
                                    prev_prompt_embeds_list, prev_prompt_attention_mask_list, _, _, prev_attention_mask_list, _ = self.layerstore._get_all_layers()
                                    # concatenate all the prev_prompt_embeds_list
                                    if len(prev_prompt_embeds_list) == 1:
                                        prev_prompt_embeds = prev_prompt_embeds_list[0].to(device)
                                        prev_prompt_attention_mask = prev_prompt_attention_mask_list[0].to(device)
                                        prev_attention_masks = None
                                        # background_attention_mask = prev_attention_mask_list[0].to(device)
                                    else:
                                        prev_prompt_embeds = torch.cat(prev_prompt_embeds_list, dim=2)
                                        prev_prompt_attention_mask = torch.cat(prev_prompt_attention_mask_list, dim=1)
                                        if len(prev_attention_mask_list) == 2:
                                            prev_attention_masks = prev_attention_mask_list[1]
                                        else:
                                            prev_attention_mask_list = prev_attention_mask_list[1:]
                                            prev_attention_masks = torch.cat(prev_attention_mask_list, dim=1)
                                noise_pred = self.transformer(
                                    latent_model_input.to(device),
                                    encoder_hidden_states=prompt_embeds.to(device),
                                    prev_encoder_hidden_states=prev_prompt_embeds.to(device),
                                    encoder_attention_mask=prompt_attention_mask.to(device),
                                    prev_encoder_attention_masks=prev_prompt_attention_mask.to(device),
                                    attention_mask=mask.to(device) if mask is not None else None,
                                    prev_attention_masks=prev_attention_masks.to(device) if prev_attention_masks is not None else None,
                                    # background_attention_mask=background_attention_mask.to(device),
                                    # attention_mask(?)=mask.to(device) if mask is not None else None,
                                    timestep=current_timestep.to(device),
                                    actual_timestep=i,
                                    multi_query_disentanglement=multi_query_disentanglement,
                                    object_gen=object_gen,
                                    utilize_cache=utilize_cache,
                                    added_cond_kwargs=added_cond_kwargs,
                                    return_dict=False,
                                )[0]
                                    
                            else: # ONLY CATTN MASKING IS TRUE
                                if self.prev_prompt_embeds is not None:
                                    prev_prompt_embeds = self.prev_prompt_embeds.to(device)
                                    prev_prompt_attention_mask = self.prev_prompt_attention_mask.to(device)
                                else:
                                    prev_prompt_embeds = prompt_embeds.to(device)
                                    prev_prompt_attention_mask = prompt_attention_mask.to(device)
                                noise_pred = self.transformer(
                                    latent_model_input.to(device),
                                    encoder_hidden_states=prompt_embeds.to(device),
                                    prev_encoder_hidden_states=prev_prompt_embeds.to(device),
                                    prev_encoder_attention_masks=prev_prompt_attention_mask.to(device),
                                    attention_mask=mask.to(device) if mask is not None else None,
                                    encoder_attention_mask=prompt_attention_mask.to(device),
                                    subject_token_idx=subject_token_idx,
                                    sigma=sigma,
                                    # attention_mask(?)=mask.to(device) if mask is not None else None,
                                    timestep=current_timestep.to(device),
                                    actual_timestep=i,
                                    added_cond_kwargs=added_cond_kwargs,
                                    object_gen=object_gen,
                                    utilize_cache=utilize_cache,
                                    return_dict=False,
                                )[0]

                        else:     
                            # dealing with prev_prompt_embed stuffs
                            prev_prompt_embeds = prompt_embeds.to(device)
                            prev_prompt_attention_mask = prompt_attention_mask.to(device)               
                            noise_pred = self.transformer(
                                latent_model_input.to(device),
                                encoder_hidden_states=prompt_embeds.to(device),
                                prev_encoder_hidden_states=prev_prompt_embeds.to(device),
                                encoder_attention_mask=prompt_attention_mask.to(device),
                                prev_encoder_attention_masks=prev_prompt_attention_mask.to(device),
                                subject_token_idx=subject_token_idx,
                                sigma=sigma,
                                attention_mask=None,
                                # attention_mask(?)=mask.to(device) if mask is not None else None,
                                timestep=current_timestep.to(device),
                                actual_timestep=i,
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )[0]
                if remove_prev is not True or (remove_prev is True and i >= len(timesteps) - int(num_inference_steps * vanilla_ratio)):
                    # Split the noise prediction into two parts if the model predicts both mean and variance
                    if noise_pred.shape[1] == latents.shape[1] * 2:
                        noise_pred_mean, noise_pred_variance = noise_pred.chunk(2, dim=1)
                        noise_pred = noise_pred_mean  # Only use the mean for denoising

                    # Perform guidance if guidance_scale > 1.0
                    if do_classifier_free_guidance:
                        # Separate noise predictions for the unconditional and text-conditioned latents
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        # Apply guidance
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        
                    if inpaint:
                        if do_classifier_free_guidance:
                            init_mask, _ = mask.chunk(2)
                        else:
                            init_mask = mask

                    # Ensure latents and noise_pred are of compatible sizes before the step
                    latents = self.scheduler.step(noise_pred, t, latents, **self.prepare_extra_step_kwargs(generator, eta))[0]

                    # Apply inpainting if enabled and not the last timestep
                    if remove_prev is not True:
                        if inpaint and i < len(timesteps) - int(num_inference_steps * vanilla_ratio): # minimum (1)
                            # Prepare the latents for the next timestep with inpainting
                            noise_timestep = timesteps[i + int(num_inference_steps * vanilla_ratio)] # match the minimum starting timestep with this (as we skip some of these steps)
                            # init_latents_proper = self.scheduler.add_noise(image_latents, noise, torch.tensor([noise_timestep]))
                            # Blend latents with the inpaint mask
                            mask_orig_init, _ = mask_orig.chunk(2)
                            # mask_orig_init = mask_orig_init * (1 - 2 * decay_ratio) + decay_ratio
                            # mask_orig_init = self.erode_tensor(mask_orig_init, kernel_size=3)
                            latents = (1 - mask_orig_init) * latent_memory[int(t.detach().cpu().numpy())] + mask_orig_init * latents # utilizing latent_memory
                            # mask_orig_init, _ = mask_orig.chunk(2)
                            # latents = (1 - mask_orig_init) * init_latents_proper + mask_orig_init * latents
                    else:
                        if inpaint and i < len(timesteps) - int(num_inference_steps * vanilla_ratio / 2): # minimum (1)
                            # Prepare the latents for the next timestep with inpainting
                            # noise_timestep = timesteps[i + int(num_inference_steps * vanilla_ratio)] # match the minimum starting timestep with this (as we skip some of these steps)
                            # init_latents_proper = self.scheduler.add_noise(image_latents, noise, torch.tensor([noise_timestep]))
                            # Blend latents with the inpaint mask
                            mask_orig_init, _ = mask_orig.chunk(2)
                            # mask_orig_init = mask_orig_init * (1 - 2 * decay_ratio) + decay_ratio
                            # mask_orig_init = self.erode_tensor(mask_orig_init, kernel_size=3)
                            latents = (1 - mask_orig_init) * self.latent0[int(t.detach().cpu().numpy())] + mask_orig_init * latents # utilizing latent_memory
                    if object_gen is True and utilize_cache is False: # SOLE OBJECT GENERATION
                        # NO UPDATE ON LATENT -> PASSING THE GIVEN LATENT BACK TO LATENT MEMORY
                        pass
                    else:
                        # UPDATING THE LATENT MEMORY (UTILIZING CACHING OR BACKGROUND GEN)
                        latent_memory_new[int(t.detach().cpu().numpy())] = latents
                else : # remove previous object with latent memory                  
                    latents = self.latent2[int(t.detach().cpu().numpy())] * mask2 + self.latent0[int(t.detach().cpu().numpy())] * (1 - mask2)
                    latent_memory_new[int(t.detach().cpu().numpy())] = latents
                    self.latent2[int(t.detach().cpu().numpy())] = latents
                # Update the progress bar
                if i == len(timesteps) - 1 or (i + 1 > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                    # Call the callback if provided
                    if callback and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
                decay_ratio += i * 0.01
        # Store latent info into layerstore
        if self.layerstore._len_of_stored_layers() == 0: # attention_mask for background is WHOLE - mask, so skipping saving for the first time
            self.layerstore._store_layer(prompt_embeds, prompt_attention_mask, None, None, None, None)
        elif remove_prev is not True:
            self.layerstore._store_layer(prompt_embeds, prompt_attention_mask, None, None, mask, mask_orig)

        # 11. Decode latents into images (only if output_type is not 'latent')
        if output_type != "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            if use_resolution_binning:
                image = self.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)
            image = self.image_processor.postprocess(image, output_type=output_type)
        # STORE THE PREVIOUS PROMPT EMBEDS (FOR CROSS_ATTENTION MASKING)]
        if remove_prev is not True:
            self._prev_prompt_store(prompt, prompt_embeds, prompt_attention_mask)


        # Offload models
        self.maybe_free_model_hooks()

        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time} in seconds")
        # 12. Return the output
        if not return_dict:
            return (image, latent_memory_new)
        
        return ImagePipelineOutput(images=image), latent_memory_new
