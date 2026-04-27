from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3Model
except ImportError:  # pragma: no cover - guarded by requirements
    Qwen3Config = None
    Qwen3ForCausalLM = None
    Qwen3Model = None

from llava.model.llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwen2Config(Qwen2Config):
    model_type = "llava_qwen2"


if Qwen3Config is not None:
    class LlavaQwen3Config(Qwen3Config):
        model_type = "llava_qwen3"
else:  # pragma: no cover - only used when transformers is too old
    LlavaQwen3Config = None


class _LlavaQwenMixin(LlavaMetaForCausalLM):
    def get_model(self):
        return self.model

    @staticmethod
    def _is_rank0():
        return not (torch.distributed.is_available() and torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0

    @staticmethod
    def _count_active_token(labels, active_mask, token_id):
        if token_id is None or token_id < 0:
            return 0
        return int(((labels == int(token_id)) & active_mask).sum().item())

    @staticmethod
    def _count_active_range(labels, active_mask, start_id, end_id):
        if start_id is None or end_id is None or start_id < 0 or end_id < 0:
            return 0
        start_id, end_id = sorted((int(start_id), int(end_id)))
        return int((((labels >= start_id) & (labels <= end_id)) & active_mask).sum().item())

    def _maybe_print_label_debug(self, labels):
        debug_steps = int(getattr(self.config, "label_debug_steps", 0) or 0)
        if debug_steps <= 0 or labels is None or not self.training or not self._is_rank0():
            return

        debug_seen = getattr(self, "_label_debug_seen", 0)
        if debug_seen >= debug_steps:
            return
        self._label_debug_seen = debug_seen + 1

        with torch.no_grad():
            active_mask = labels != -100
            active_count = int(active_mask.sum().item())
            total_count = labels.numel()
            active_ratio = active_count / max(total_count, 1)

            active_labels = labels[active_mask]
            vocab_size = getattr(self.config, "vocab_size", None)
            invalid_count = 0
            if vocab_size is not None and active_count > 0:
                invalid_count = int(((active_labels < 0) | (active_labels >= int(vocab_size))).sum().item())

            pad_count = self._count_active_token(labels, active_mask, getattr(self.config, "label_debug_pad_token_id", None))
            bos_count = self._count_active_token(labels, active_mask, getattr(self.config, "label_debug_bos_token_id", None))
            eos_count = self._count_active_token(labels, active_mask, getattr(self.config, "label_debug_eos_token_id", None))
            qwen_start_count = self._count_active_token(labels, active_mask, getattr(self.config, "label_debug_qwen_im_start_id", None))
            qwen_end_count = self._count_active_token(labels, active_mask, getattr(self.config, "label_debug_qwen_im_end_id", None))
            newline_count = sum(
                self._count_active_token(labels, active_mask, token_id)
                for token_id in getattr(self.config, "label_debug_newline_ids", [])
            )

            image_tag_count = (
                self._count_active_token(labels, active_mask, getattr(self.config, "image_start_tag_id", None))
                + self._count_active_token(labels, active_mask, getattr(self.config, "image_end_tag_id", None))
            )
            image_token_count = self._count_active_range(
                labels,
                active_mask,
                getattr(self.config, "image_start_token_id", None),
                getattr(self.config, "image_end_token_id", None),
            )
            scale_token_count = self._count_active_range(
                labels,
                active_mask,
                getattr(self.config, "scale_start_token_id", None),
                getattr(self.config, "scale_end_token_id", None),
            )

        print(
            "[LabelDebug "
            f"{self._label_debug_seen}/{debug_steps}] "
            f"shape={tuple(labels.shape)} "
            f"active={active_count}/{total_count}({active_ratio:.3f}) "
            f"pad={pad_count} bos={bos_count} eos={eos_count} "
            f"qwen_im_start={qwen_start_count} qwen_im_end={qwen_end_count} newline={newline_count} "
            f"image_tags={image_tag_count} image_tokens={image_token_count} scale_tokens={scale_token_count} "
            f"invalid={invalid_count}",
            flush=True,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = None,
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if modalities is None:
            modalities = ["image"]
        if inputs_embeds is None:
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                modalities,
                image_sizes,
            )

        self._maybe_print_label_debug(labels)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if modalities is None:
            modalities = ["image"]
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            inputs, position_ids, attention_mask, _, inputs_embeds, _ = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                modalities,
                image_sizes=image_sizes,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super().__init__(config)


class LlavaQwen2Model(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwen2Config

    def __init__(self, config: Qwen2Config):
        super().__init__(config)


class LlavaQwenForCausalLM(_LlavaQwenMixin, Qwen2ForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        config.model_type = self.config_class.model_type
        Qwen2ForCausalLM.__init__(self, config)
        config.rope_scaling = getattr(config, "rope_scaling", None)

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


class LlavaQwen2ForCausalLM(_LlavaQwenMixin, Qwen2ForCausalLM):
    config_class = LlavaQwen2Config

    def __init__(self, config):
        config.model_type = self.config_class.model_type
        Qwen2ForCausalLM.__init__(self, config)
        config.rope_scaling = getattr(config, "rope_scaling", None)

        self.model = LlavaQwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


if Qwen3ForCausalLM is not None:
    class LlavaQwen3Model(LlavaMetaModel, Qwen3Model):
        config_class = LlavaQwen3Config

        def __init__(self, config: Qwen3Config):
            super().__init__(config)


    class LlavaQwen3ForCausalLM(_LlavaQwenMixin, Qwen3ForCausalLM):
        config_class = LlavaQwen3Config

        def __init__(self, config):
            config.model_type = self.config_class.model_type
            Qwen3ForCausalLM.__init__(self, config)
            config.rope_scaling = getattr(config, "rope_scaling", None)

            self.model = LlavaQwen3Model(config)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()
else:  # pragma: no cover - only used when transformers is too old
    LlavaQwen3Model = None
    LlavaQwen3ForCausalLM = None


def get_llava_qwen_config_class(model_name_or_path=None, config=None):
    if config is None:
        config = AutoConfig.from_pretrained(model_name_or_path)
    model_type = getattr(config, "model_type", "")

    if model_type in {"llava_qwen", "llava_qwen2", "qwen2"}:
        return LlavaQwen2Config if model_type == "llava_qwen2" else LlavaQwenConfig
    if model_type in {"llava_qwen3", "qwen3"}:
        if LlavaQwen3Config is None:
            raise ImportError("Qwen3 support requires transformers>=4.51.0.")
        return LlavaQwen3Config

    raise ValueError(f"Unsupported Qwen config type: {model_type}")


def get_llava_qwen_model_class(model_name_or_path=None, config=None):
    config_class = get_llava_qwen_config_class(model_name_or_path=model_name_or_path, config=config)
    if config_class is LlavaQwenConfig:
        return LlavaQwenForCausalLM
    if config_class is LlavaQwen2Config:
        return LlavaQwen2ForCausalLM
    if config_class is LlavaQwen3Config:
        return LlavaQwen3ForCausalLM
    raise ValueError(f"Unsupported Llava Qwen config class: {config_class}")


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoConfig.register("llava_qwen2", LlavaQwen2Config)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
AutoModelForCausalLM.register(LlavaQwen2Config, LlavaQwen2ForCausalLM)
if LlavaQwen3Config is not None:
    AutoConfig.register("llava_qwen3", LlavaQwen3Config)
    AutoModelForCausalLM.register(LlavaQwen3Config, LlavaQwen3ForCausalLM)
