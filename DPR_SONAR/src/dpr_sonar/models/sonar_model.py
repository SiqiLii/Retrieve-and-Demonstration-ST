
from dpr_sonar.models.biencoder_SONAR import BiEncoder

import logging
from dpr_sonar.utils.data_utils_SONAR import Tensorizer
import torch
import torch.nn as nn
from torch import Tensor as T
import sys
from omegaconf import OmegaConf
from typing import Tuple, List
import transformers
import numpy as np

from transformers import BertConfig, BertModel
from transformers import AdamW
from transformers import BertTokenizer
from transformers import RobertaTokenizer

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union
from sonar.inference_pipelines.utils import add_progress_bar, extract_sequence_batch
from fairseq2.data import SequenceData
import torch
from fairseq2.data.text import TextTokenizer, read_text
from fairseq2.data.data_pipeline import read_sequence
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from fairseq2.data import (
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    FileMapper,
    StringLike,
)
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.data.data_pipeline import read_sequence
from fairseq2.memory import MemoryBlock
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
import torchaudio
from torch.utils.checkpoint import checkpoint
from fairseq2.nn.padding import PaddingMask

logger = logging.getLogger(__name__)

class SonarEncoderModel(nn.Module):
        def __init__(self,input_type ="text"):
                super(SonarEncoderModel, self).__init__()
                self.input_type=input_type
                if input_type =="text":
                    logging.info("initializing text sonar basic encoder")
                    t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                           tokenizer="text_sonar_basic_encoder")
                    self.encoder=t2vec_model.model
                elif input_type =="audio":
                    logging.info("initializing speech sonar basic encoder")
                    s2vec_model = SpeechToEmbeddingModelPipeline(encoder="sonar_speech_encoder_eng",fbank_dtype=torch.float32)
                    self.encoder=s2vec_model.model
                else:
                    raise ValueError("input_type should be either text or speech")
                
        def init_encoder(self):
                if self.input_type=="audio":
                    for n,param in self.encoder.named_parameters():
                        name_parts = n.split('.')
                    
                        #if ('frontend' in n) or ('decoder' in n) or ('pooler' in n):
                        if ('encoder_frontend' in n):
                            print(f"Freezing {n}")
                            param.requires_grad = False
                        elif len(name_parts) > 3 and name_parts[2].isdigit() and int(name_parts[2]) < 22: #audio <22 #text freeze_10 <11 freeze_15 <16
                            print(f"Freezing {n}")
                            param.requires_grad = False
                        else:
                            param.requires_grad = True  # Ensure gradients are required
                elif self.input_type=="text":
                    for n,param in self.encoder.named_parameters():
                        name_parts = n.split('.')
                        if ('encoder_frontend' in n):
                            print(f"Freezing {n}")
                            param.requires_grad = False
                        elif len(name_parts) > 3 and name_parts[2].isdigit() and int(name_parts[2]) < 21:
                            print(f"Freezing {n}")
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                self.encoder.cuda()

        def forward(self, input):
           
            #embeddings=[]
            #for item in input:
            #    embeddings.append(self.encoder(item).sentence_embeddings)
            #embeddings=torch.cat(embeddings,dim=0) #.to('cuda')
            if input.padding_mask is not None:
                input.padding_mask.seq_lens=input.padding_mask.seq_lens.to(input.seqs.device)
            embeddings=self.encoder(input).sentence_embeddings
            return embeddings

        # def forward(self, input):
            
        #     x=checkpoint(self.checkpointed_forward, input)
        #     print("embeddings in forward requires_grad?",x.requires_grad)
        #     return x
                
               
        # def checkpointed_forward(self,input):
        #         for name, param in self.encoder.named_parameters():
        #             if param.grad is not None:
        #                 print(f"Gradient for {name}: {param.grad}")
        #             else:
        #                 print(f"No gradient for {name}")
        #         # Process each batch through the model
        #         embeddings=[]
        #         for item in input:
        #             print("item in checkpointed_forward requires_grad?",item.seqs.requires_grad)
        #             #item.seqs = item.seqs.float().clone().detach().requires_grad_(True)
        #             embeddings.append(self.encoder(item).sentence_embeddings)
        #             print("embeddings?",embeddings)
        #         embeddings=torch.cat(embeddings,dim=0).to('cuda')
        #         print("embeddings in checkpointed_forward requires_grad?",embeddings.requires_grad)
        #         return embeddings
        
        def state_dict(self):
                return self.encoder.state_dict()
        


def get_bert_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    question_input_type=cfg.question_input_type
    ctx_input_type=cfg.ctx_input_type
    logging.info("initializing question encoder")
    question_encoder = SonarEncoderModel(question_input_type)
    question_encoder.init_encoder()
    logging.info("initializing context encoder")
    ctx_encoder = SonarEncoderModel(ctx_input_type)
    ctx_encoder.init_encoder()

    fix_ctx_encoder = cfg.encoder.fix_ctx_encoder if hasattr(cfg.encoder, "fix_ctx_encoder") else False
    biencoder = BiEncoder(question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder)

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_tensorizer(cfg)
    return tensorizer, biencoder, optimizer



def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    optimizer_grouped_parameters = get_sonar_model_param_grouping(model, weight_decay)
    return get_optimizer_grouped(optimizer_grouped_parameters, learning_rate, adam_eps)


def get_sonar_model_param_grouping(
    model: nn.Module,
    weight_decay: float = 0.0,
):
    no_decay = ["bias", "layer_norm.weight"]
    #for n,p in model.named_parameters():
    #    print(n)
    #    if p.grad is not None:
    #        print(n)
    #        print("grad is ",p.grad.abs().max())
        
    return [
        {
            "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay) and p.requires_grad)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay) and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]


def get_optimizer_grouped(
    optimizer_grouped_parameters: List,
    learning_rate: float = 1e-6,
    adam_eps: float = 1e-8,
) -> torch.optim.Optimizer:

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer

def get_tensorizer(cfg):

    return SonarTensorizer(cfg)

def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return BertTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)


class SonarTensorizer(Tensorizer):
    def __init__(self, cfg, pad_to_max: bool = True):
        t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                           tokenizer="text_sonar_basic_encoder")
        self.tokenizer=t2vec_model.tokenizer
        self.tokenizer_encoder = self.tokenizer.create_encoder(lang="eng_Latn")
        
    def audio_to_tensor(self, input=List[str]):
         
        inps=[]
        for item in input:
             inps.append(torchaudio.load(item)[0].to('cpu'))
        def _decode_audio(inp: Union[str, torch.Tensor]) -> dict:
            if isinstance(inp, torch.Tensor):
                return {
                    "waveform": inp.transpose(1, 0),
                    "sample_rate": 16000.0,
                    "format": -1,
                }
            else:
                with Path(str(inp)).open("rb") as fb:
                    block = MemoryBlock(fb.read())
                return AudioDecoder(block)  # type: ignore
    
        
        decoded_audio = [_decode_audio(item) for item in inps]
        

        def convert_to_fbank_(item):
            convert_to_fbank = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=True,
            device=torch.device("cpu"),  # Use torch.device for the device argument
            dtype=torch.float32,  # dtype is correct, but ensure it's compatible with your fairseq2n version
            )
    
            out=convert_to_fbank(item)
            if out["fbank"].size(0) >4096:
                out["fbank"] = out["fbank"][:4096,:]
            return out

            # Convert decoded audio to fbank features, possibly in parallel
        fbank_features = [convert_to_fbank_(item) for item in decoded_audio]
        
        # Group the data into batches
        batch=fbank_features

        pad_idx =  torch.tensor(0, dtype=torch.int32)
        # Pad and collate batches
        collated_batch = Collater(pad_value=pad_idx, pad_to_multiple=2)(batch)
        
        # Extract sequence batches and move to device
        sequence_batch=collated_batch
        sequence_batch["fbank"] = extract_sequence_batch(collated_batch["fbank"], torch.device("cpu")) 

        return sequence_batch["fbank"].seqs, sequence_batch["fbank"].padding_mask

    def text_to_tensor(self, input=List[str],max_seq_len=514):
        

        # Tokenize the data
        tokenized_data = [self.tokenizer_encoder(item) for item in input]

        def truncate(x: torch.Tensor) -> torch.Tensor:
            return x[:max_seq_len]
        # Truncate the data
        truncated_data = [truncate(item) for item in tokenized_data]

        batch=truncated_data
        # Pad the batches and prepare them for the model
        collated_batch = Collater(self.tokenizer.vocab_info.pad_idx)(batch)

        # Extract sequence batches and move to the device
        sequence_batch = extract_sequence_batch(collated_batch, 'cpu') 
        
        return sequence_batch.seqs, sequence_batch.padding_mask

    
    
    def state_dict(self):
        return {'tokenizer_encoder':'text_sonar_basic_encoder'}
    #def text_to_tensor(self,input):
    #    encoder_input, encoder_padding_mask = self.text_encoder_prenet(input)
    #    return encoder_input, encoder_padding_mask
