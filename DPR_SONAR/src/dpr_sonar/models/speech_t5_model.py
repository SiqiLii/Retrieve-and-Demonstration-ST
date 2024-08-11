
from dpr_t5.models.biencoder import BiEncoder
from SpeechT5.SpeechT5.speecht5.models.modules.text_encoder_prenet import TextEncoderPrenet
from SpeechT5.SpeechT5.speecht5.models.modules.speech_encoder_prenet import SpeechEncoderPrenet
from SpeechT5.SpeechT5.speecht5.models.modules.text_encoder_prenet import TextEncoderPrenet
from SpeechT5.SpeechT5.speecht5.models.modules.encoder import TransformerEncoder
from fairseq.models.transformer import Embedding
from fairseq.data import Dictionary
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
    get_waveform
)

import logging
from dpr_t5.utils.data_utils import Tensorizer
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

logger = logging.getLogger(__name__)

class T5EncoderModel(nn.Module):
        def __init__(self, cfg):
                super(T5EncoderModel, self).__init__()
                self.encoder=TransformerEncoder(args=cfg.encoder)
                self.cfg = cfg
        def init_encoder(self):
                checkpoint = torch.load('/home/sli/DPR_t5/dpr_t5/SpeechT5/speecht5_base.pt')
                module_prefix='encoder.'
                filtered_checkpoint={(k.replace(module_prefix,"")):v for k,v in checkpoint['model'].items() if ((module_prefix in k) and (k.replace(module_prefix,"") in self.encoder.state_dict().keys()))}
                self.encoder.load_state_dict(filtered_checkpoint,strict=True)
                print("successfully load encoder checkpoint")

                self.encoder.to('cuda')
                self.encoder.eval()
                
               
        def forward(self, encoder_input,segment, attn_mask, representation_token_pos=0):
                #print("encoder input",encoder_input.size())
                #print("encoder_padding_mask",attn_mask)

                #print("encoder input on cuda?",encoder_input.is_cuda)
                #print("encoder_padding_mask on cuda?",attn_mask.is_cuda)
                #print("model on cuda?",next(self.encoder.parameters()).is_cuda)
                encoder_output = self.encoder(encoder_input, encoder_padding_mask=attn_mask)
                sequence_output=encoder_output['encoder_out'][0]
                #print("sequence output size",sequence_output.size()) #(bsz,256,768)
                attn_mask_transposed = attn_mask.t()
                valid_mask = 1 - attn_mask_transposed
                #print("valid mask unsqueezed size",valid_mask.unsqueeze(-1).size()) #(bsz,256,1)
                masked_tensor = sequence_output* valid_mask.unsqueeze(-1)

                summed_tensor = masked_tensor.sum(dim=1)
                valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
                pooled_output = summed_tensor / valid_counts.float()  # Ensure division is in float for correct averaging
                #print("pooled output size",pooled_output.size()) #(bsz,768)
                

                #if isinstance(representation_token_pos, int):
                #
                #    pooled_output = sequence_output[:, representation_token_pos, :]
                #else:  # treat as a tensor
                #    bsz = sequence_output.size(0)
                #    assert representation_token_pos.size(0) == bsz, "query bsz={} while representation_token_pos bsz={}".format(
                #    bsz, representation_token_pos.size(0)
                #    )
                #    pooled_output = torch.stack([sequence_output[i, representation_token_pos[i, 1], :] for i in range(bsz)])
                
                hidden_states=encoder_output['encoder_out'][0]

                #print("encoder output size",encoder_output['encoder_out'][0].size())
                return encoder_output['encoder_out'][0],pooled_output, hidden_states
        
        def state_dict(self):
                #speech_encoder_pre_dict=self.speech_encoder_prenet.state_dict()
                #speech_encoder_pre_dict_={'speech_encoder_prenet.'+k: v for k, v in speech_encoder_pre_dict.items()}
                #encoder_dict=self.encoder.state_dict()
                #encoder_dict_={'encoder.'+k: v for k, v in encoder_dict.items()}
                #return {**encoder_dict_}
                return self.encoder.state_dict()
        def load_state_dict(self, state_dict):
                #speech_encoder_pre_dict={k[22:]: v for k, v in state_dict.items() if k.startswith('speech_encoder_prenet.')}
                #self.speech_encoder_prenet.load_state_dict(speech_encoder_pre_dict)
                encoder_dict={k[8:]: v for k, v in state_dict.items() if k.startswith('encoder.')}
                self.encoder.load_state_dict(encoder_dict)


def get_bert_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    question_encoder = T5EncoderModel(cfg)
    question_encoder.init_encoder()
    ctx_encoder = T5EncoderModel(cfg)
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




sys.path.insert(1, '/home/sli/DPR_t5/dpr_t5/SpeechT5/SpeechT5/')
from speecht5.tasks.speecht5 import SpeechT5Task
from speecht5.models.speecht5 import T5TransformerModel


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    optimizer_grouped_parameters = get_t5_model_param_grouping(model, weight_decay)
    return get_optimizer_grouped(optimizer_grouped_parameters, learning_rate, adam_eps)


def get_t5_model_param_grouping(
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
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
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

    return T5Tensorizer(cfg)

def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return BertTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)

def build_embedding(dictionary, embed_dim, max_num_embeddings=None):
            num_embeddings = len(dictionary)
            if max_num_embeddings is not None and isinstance(max_num_embeddings, int):
                num_embeddings = min(num_embeddings, max_num_embeddings)  
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

class T5Tensorizer(Tensorizer):
    def __init__(self, cfg, pad_to_max: bool = True):
        #self.text_encoder_prenet=TextEncoderPrenet(embed_tokens=None,args=cfg.encoder)
        

        #d=Dictionary(extra_special_symbols=["<cls>","<sep>"])
        #d.add_from_file('/export/data2/sli/data/MuST-C_synthesized/de/en-de/spm_unigram5000_asr.txt')

        
        #self.dictionary=d  
        import sentencepiece as spm

        spm_file_path='/home/sli/DPR_t5/spm_char.model'
        dict_path='/home/sli/DPR_t5/dict.txt'
        #spm_model = spm.SentencePieceProcessor(model_file="bpe.model")
        spm_model=spm.SentencePieceProcessor(model_file=spm_file_path)
        dict=Dictionary.load(dict_path)
        mask_idx = dict.add_symbol("<mask>")
        # add blank token for ctc
        # if args.ctc_weight > 0:
        blank_symbol_idx = dict.add_symbol("<ctc_blank>")
        blank_symbol = "<ctc_blank>"

        self.tokenizer=spm_model
        embed_tokens = build_embedding(dict, embed_dim=cfg.encoder.encoder_embed_dim) 
        self.text_encoder_prenet=TextEncoderPrenet(embed_tokens,cfg.encoder)
        #checkpoint = torch.load('path_to_checkpoint.pth', map_location=torch.device('cuda'))

        self.speech_encoder_prenet=SpeechEncoderPrenet(args=cfg.encoder)
        self.speech_encoder_prenet.padding_idx=self.tokenizer.pad_id()
        # Load the state dictionary from the checkpoint into your model
        #self.speech_encoder_prenet.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load('/home/sli/DPR_t5/dpr_t5/SpeechT5/speecht5_base.pt')
        module_prefix='text_encoder_prenet.'
        filtered_checkpoint={(k.replace(module_prefix,"")):v for k,v in checkpoint['model'].items() if 
                             ((module_prefix in k) and (k.replace(module_prefix,"") in self.text_encoder_prenet.state_dict().keys()))}
        self.text_encoder_prenet.load_state_dict(filtered_checkpoint,strict=True)
        print("successfully load text_encoder_prenet checkpoint")
        self.text_encoder_prenet.to('cuda')

        module_prefix='speech_encoder_prenet.'
        filtered_checkpoint={(k.replace(module_prefix,"")):v for k,v in checkpoint['model'].items() if 
                             ((module_prefix in k) and (k.replace(module_prefix,"") in self.speech_encoder_prenet.state_dict().keys()))}
        self.speech_encoder_prenet.load_state_dict(filtered_checkpoint,strict=True)
        print("successfully load speech_encoder_prenet checkpoint")
        self.speech_encoder_prenet.to('cuda')
        
        

    def audio_to_tensor(self, input,padding_mask=None, mask=True,audio_padding_size=None):
        
        wav, curr_sample_rate = get_waveform(input)  #input is wav file path
        padding_idx=self.tokenizer.pad_id()
        audio_padding_size=500000
        if audio_padding_size is not None:
            target_length = audio_padding_size
            current_length = wav.shape[1]
            pad_length = target_length - current_length
            if pad_length<0:
                #crop the wav
                padded_wav=wav[:,:target_length]
            else:

            # Pad the array
            # The padding format is ((0, 0), (0, pad_length)) 
            # which means no padding for the first dimension and pad_length padding for the second dimension
                padded_wav = np.pad(wav, ((0, 0), (0, pad_length)), 'constant', constant_values=(padding_idx,))
        else:
            padded_wav=wav
        #torch.from_numpy(wav).size()
        src_tokens=torch.from_numpy(padded_wav) 
        src_tokens=src_tokens.cuda()
        #print(src_tokens.size())
        
        padding_mask = src_tokens.eq(padding_idx)
        encoder_input, encoder_padding_mask=self.speech_encoder_prenet(src_tokens,padding_mask=padding_mask,mask=True)
        
        encoder_input_requires_grad = encoder_input.clone().detach().requires_grad_(True)

        return encoder_input_requires_grad, (encoder_padding_mask).float()

    def text_to_tensor(self, input,padding_mask=None, mask=True):
        
        
        #torch.from_numpy(wav).size()
        tokenized_line=self.tokenizer.encode(input.strip(), out_type=int)
        #print("tokenized_line",tokenized_line)
        seq_len = 256 #hard coded
        pad_to_max=True
        apply_max_len=True
        if pad_to_max:
            if len(tokenized_line) < seq_len:
                tokenized_line_ = tokenized_line + [self.tokenizer.pad_id()] * (seq_len - len(tokenized_line))
            elif len(tokenized_line) >= seq_len:
                tokenized_line_ = tokenized_line[0:seq_len] if apply_max_len else tokenized_line
                tokenized_line_[-1] = self.tokenizer.eos_id()
        else:
            tokenized_line_ = tokenized_line
        src_tokens=torch.from_numpy(np.array(tokenized_line_)).unsqueeze(1)
        src_tokens=src_tokens.cuda()
       
        encoder_input, encoder_padding_mask=self.text_encoder_prenet(src_tokens)
        encoder_input_requires_grad = encoder_input.clone().detach().requires_grad_(True)
        return encoder_input_requires_grad, (encoder_padding_mask).float()
    
    #def text_to_tensor(self,input):
    #    encoder_input, encoder_padding_mask = self.text_encoder_prenet(input)
    #    return encoder_input, encoder_padding_mask

    
    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_id()

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


    def get_token_id(self, token: str) -> int:
        return self.tokenizer.vocab[token]
    def state_dict(self):
        return {'text_encoder_prenet':self.text_encoder_prenet.state_dict(),'speech_encoder_prenet':self.speech_encoder_prenet.state_dict()}