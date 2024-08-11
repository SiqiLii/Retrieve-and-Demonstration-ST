#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""

import collections
import logging
import random
from typing import Tuple, List
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn
import sys
sys.path.append('/home/sli/DPR_SONAR/')
sys.path.append('/home/sli/DPR_SONAR/src/dpr_sonar/')
from dpr_sonar.data.biencoder_data_SONAR import BiEncoderSample
from dpr_sonar.utils.data_utils_SONAR import Tensorizer
from dpr_sonar.utils.model_utils_SONAR import CheckpointState
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask
from typing import Iterable, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

BiEncoderBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "question_ids",
        "question_lens",
        "context_ids",
        "context_lens",
        "is_positive",
        "hard_negatives",
        "encoder_type",
    ],
)
# TODO: it is only used by _select_span_with_token. Move them to utils
rnd = random.Random(0)


def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    
    return r


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)


class BiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
    ):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        attn_mask: PaddingMask,
        fix_encoder: bool = False,
    ) -> T:
        # Assuming 'model' is your model instance
        outputs=[]
        out=None
        pooled_output = None
        if ids is not None:
            model_input = SequenceBatch(ids, attn_mask)
            if fix_encoder:
                # for item in ids:

                #     with torch.no_grad():
                #         out= sub_model(item)
                #         outputs.append(out)
                # pooled_output=torch.cat(outputs,dim=0) #.to('cuda')
                with torch.no_grad():
                    pooled_output = sub_model(model_input)
                if sub_model.training:
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                # for item in ids:
                #     #print(item)
                    
                #     # Assuming 'model' is your PyTorch model
                    
                #     out=sub_model(item)
                #     outputs.append(out)
                    
                #pooled_output=torch.cat(outputs,dim=0) #.to('cuda')
                
                pooled_output= sub_model(model_input)
                #print("model_input: ",model_input)
                #print("model input padding",model_input.padding_mask.seq_lens)
                #print("pooled_output: ",pooled_output) 
        return pooled_output
    
    def get_attn_mask(self, tokens_tensor: T,seq_lens:Union[T,None] = None):
        if seq_lens is None:
            return None
        seq_lens = seq_lens.squeeze(-1)
        device=tokens_tensor.device
        seq_lens = seq_lens.to(device)
        return PaddingMask(seq_lens, batch_seq_len=tokens_tensor.size(1))

    def forward(
        self,
        question_ids: T,
        question_lens: T,
        context_ids: T,
        context_lens: T,
        encoder_type: str = None,
        representation_token_pos=0,
    ) -> Tuple[T, T]:
       

        q_encoder = self.question_model if encoder_type is None or encoder_type == "question" else self.ctx_model
        q_attn_mask = self.get_attn_mask(question_ids,question_lens)
        q_pooled_out= self.get_representation(
            q_encoder,
            question_ids,
            q_attn_mask,
            self.fix_q_encoder,
        )
        #print("question_ids: ",question_ids)
        #print("q_pooled_out: ",q_pooled_out)
        ctx_encoder = self.ctx_model if encoder_type is None or encoder_type == "ctx" else self.question_model
        ctx_attn_mask = self.get_attn_mask(context_ids,context_lens)
        ctx_pooled_out= self.get_representation(
            ctx_encoder, context_ids, ctx_attn_mask,self.fix_ctx_encoder
        )
        
        #print("context_ids: ",context_ids)
        #print("ctx_pooled_out: ",ctx_pooled_out)
        return q_pooled_out, ctx_pooled_out

    def create_biencoder_input(
        self,
        samples: List[BiEncoderSample],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
        ctx_input_type: str = "text",
        question_input_type: str = "audio",
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        questions=[]
        ctxs=[]
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only
            if question_input_type=="audio":
                questions.append(sample.query_audio)
            else:
                questions.append(sample.query)

            current_ctxs_len = len(ctxs)

            if shuffle and shuffle_positives:
                #print("shuffle_positives")
                #print("sample.positive_passages: ", len(sample.positive_passages))
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
                
            else:
                positive_ctx = sample.positive_passages[0]

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            #question = sample.query
            #question_audio=sample.query_audio
            # question = normalize_question(sample.query)

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )

            if ctx_input_type=="audio":
                for item in all_ctxs:
                    ctxs.append(item.audio)
            elif ctx_input_type=="text":
                for item in all_ctxs:
                    ctxs.append(item.text)
        
        
            
        if ctx_input_type=="audio":
                
            c_tensor,c_att=tensorizer.audio_to_tensor(ctxs)
        else:
                
            c_tensor,c_att=tensorizer.text_to_tensor(ctxs)
                   
        ctxs_tensor=c_tensor
        if c_att == None:
            ctxs_lens=None
        else:
            ctxs_lens=c_att.seq_lens.unsqueeze(-1)
               
         #query_audio=False
        if query_token:
                #print("query_token: ", query_token)
                # TODO: tmp workaround for EL, remove or revise
            if query_token == "[START_ENT]":
                    query_span = _select_span_with_token(questions, tensorizer, token_str=query_token)
                    question_tensors.append(query_span)
            else:
                if question_input_type=="audio":
                        
                        q_tensor,q_att=tensorizer.audio_to_tensor(questions)
                else:
                        q_tensor,q_att=tensorizer.text_to_tensor(questions)
                question_tensors.append(q_tensor)
                    
        else:
            if question_input_type=="audio":
                    
                    q_tensor,q_att=tensorizer.audio_to_tensor(questions)
            else:
                    q_tensor,q_att=tensorizer.text_to_tensor(questions)
                   
        questions_tensor=q_tensor
        if q_att == None:
            questions_lens=None
        else:
            questions_lens=q_att.seq_lens.unsqueeze(-1)
             
        
        return BiEncoderBatch(
            questions_tensor,
            questions_lens,
            ctxs_tensor,
            ctxs_lens,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            "question",
        )

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        # TODO: make a long term HF compatibility fix
        # if "question_model.embeddings.position_ids" in saved_state.model_dict:
        #    del saved_state.model_dict["question_model.embeddings.position_ids"]
        #    del saved_state.model_dict["ctx_model.embeddings.position_ids"]
        print("saved_state.model_dict: ", saved_state.model_dict.keys())
        self.load_state_dict(saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        return {"question_model":self.question_model.state_dict(),"ctx_model":self.ctx_model.state_dict(),"fix_q_encoder":self.fix_q_encoder,"fix_ctx_encoder":self.fix_ctx_encoder}


class BiEncoderNllLoss(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        #print(q_vectors.requires_grad)  # Output: True
        #print(ctx_vectors.requires_grad)  # Output: True
        scores = self.get_scores(q_vectors, ctx_vectors)
        #print("scores size:",scores.size())
        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)
        
        def get_list_shape(lst):
            shape = []

            while isinstance(lst, list):
                shape.append(len(lst))
                # Check the first element to go deeper into the nested structure
                lst = lst[0] if lst else []

            return shape
        
        
        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )
        
        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores


def _select_span_with_token(text: str, tensorizer: Tensorizer, token_str: str = "[START_ENT]") -> T:
    id = tensorizer.get_token_id(token_str)
    query_tensor = tensorizer.text_to_tensor(text)

    if id not in query_tensor:
        query_tensor_full = tensorizer.text_to_tensor(text, apply_max_len=False)
        token_indexes = (query_tensor_full == id).nonzero()
        if token_indexes.size(0) > 0:
            start_pos = token_indexes[0, 0].item()
            # add some randomization to avoid overfitting to a specific token position

            left_shit = int(tensorizer.max_length / 2)
            rnd_shift = int((rnd.random() - 0.5) * left_shit / 2)
            left_shit += rnd_shift

            query_tensor = query_tensor_full[start_pos - left_shit :]
            cls_id = tensorizer.tokenizer.cls_token_id
            if query_tensor[0] != cls_id:
                query_tensor = torch.cat([torch.tensor([cls_id]), query_tensor], dim=0)

            from dpr_t5.models.reader import _pad_to_len

            query_tensor = _pad_to_len(query_tensor, tensorizer.get_pad_id(), tensorizer.max_length)
            query_tensor[-1] = tensorizer.tokenizer.sep_token_id
            # logger.info('aligned query_tensor %s', query_tensor)

            assert id in query_tensor, "query_tensor={}".format(query_tensor)
            return query_tensor
        else:
            raise RuntimeError("[START_ENT] toke not found for Entity Linking sample query={}".format(text))
    else:
        return query_tensor
