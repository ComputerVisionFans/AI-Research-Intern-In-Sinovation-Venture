# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import os
import argparse
import logging
import torch
import torch.nn.functional as F
from modeling_gpt2 import GPT2LMHeadModel, GPT2Config
from tokenization_bert import BertTokenizer
import math

logger = logging.getLogger(__name__)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def top_kp_search(args, model, tokenizer, curr_input_tensor):
    sep_token_id = tokenizer.vocab["[SEP]"]
    unk_token_id = tokenizer.vocab["[UNK]"]

    generated = []
    for _ in range(args.max_pred_len):
        with torch.no_grad():
            outputs = model(input_ids=curr_input_tensor)
            next_token_logits = outputs[0][-1, :]
            for id in set(generated):
                next_token_logits[id] /= args.repetition_penalty
            next_token_logits = next_token_logits / args.temperature
            next_token_logits[unk_token_id] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if next_token == sep_token_id:
                break
            generated.append(next_token.item())
            curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)
    return "".join([token.replace("##","") for token in tokenizer.convert_ids_to_tokens(generated)])


def k_best_outputs(curr_input_tensor, outputs, log_scores, index, beam_size):
    probs, ix = outputs[:, index-1].data.topk(beam_size)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(beam_size, -1) + log_scores.transpose(0, 1)
    k_probs, k_ix = log_probs.view(-1).topk(beam_size)

    row = k_ix // beam_size
    col = k_ix % beam_size

    curr_input_tensor[:, :index] = curr_input_tensor[row, :index]
    curr_input_tensor[:, index] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)

    return curr_input_tensor, log_scores

def beam_search(args, model, tokenizer, input_tensor):
    sep_token_id = tokenizer.vocab["[SEP]"]
    unk_token_id = tokenizer.vocab["[UNK]"]
    beam_size = args.beam_size
    log_scores = torch.FloatTensor([0.0]*beam_size).unsqueeze(0)
    input_tensor_len = input_tensor.size(-1)

    ind = None
    curr_input_tensor = torch.zeros(beam_size, input_tensor_len + args.max_pred_len).long().to(args.device)
    curr_input_tensor[:, :input_tensor_len] = input_tensor
    for index in range(args.max_pred_len):
        with torch.no_grad():
            outputs = model(input_ids=curr_input_tensor)
            outputs = F.softmax(outputs[0], dim=-1)
            curr_input_tensor, log_scores = k_best_outputs(curr_input_tensor, outputs, log_scores, input_tensor_len+index, beam_size)

            ones = (curr_input_tensor == sep_token_id).nonzero()  # Occurrences of end symbols for all input sentences.
            sentence_lengths = torch.zeros(len(curr_input_tensor), dtype=torch.long)
            for vec in ones:
                sentence_lengths[vec[0]] += 1

            num_finished_sentences = len([s for s in sentence_lengths if s == 2])

            if num_finished_sentences == beam_size:
                alpha = 0.7
                div = 1 / (sentence_lengths.type_as(log_scores) ** alpha)
                _, ind = torch.max(log_scores * div, 1)
                ind = ind.data[0]
                break

    if ind is None:
        ind = 0
    best_output = curr_input_tensor[ind]
    sep_indexs = [x[0].item() for x in (best_output == sep_token_id).nonzero()]
    if len(sep_indexs) == 1:
        sep_indexs.append(input_tensor_len+args.max_pred_len)
    generated = best_output[input_tensor_len: sep_indexs[1]].detach().cpu().numpy()
    return "".join([token.replace("##","") for token in tokenizer.convert_ids_to_tokens(generated)])

def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_path", default=None, type=str, help="model path")
    parser.add_argument("--max_pred_len", default=300, type=int, help="max length of predicted text")

    parser.add_argument('--temperature', default=1, type=float, required=False, help='temperature')
    parser.add_argument('--topk', default=5, type=int, required=False, help='')
    parser.add_argument('--topp', default=0, type=float, required=False, help='')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False, help="")
    parser.add_argument('--beam_size', default=5, type=int, required=False, help="")
    parser.add_argument('--search', default="beam", type=str, required=False, help="beam seatch, top_kp")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args

def main():
    args = get_args()

    config = GPT2Config.from_pretrained(args.model_path)
    tokenizer = BertTokenizer.from_pretrained(os.path.join(args.model_path, "vocab.txt"))
    model = GPT2LMHeadModel.from_pretrained(args.model_path, config=config)

    model.to(args.device)

    #demo: Given a text
    text = "深圳改革开放四十周年"
    token_ids = tokenizer.convert_tokens_to_ids(["[CLS]"])
    token_ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)))
    token_ids.extend(tokenizer.convert_tokens_to_ids(["[SEP]"]))
    input_features = torch.tensor(token_ids, dtype=torch.long).to(args.device)
    if args.search == "top_kp":
        gen_text = top_kp_search(args, model, tokenizer, input_features)
    elif args.search == "beam":
        gen_text = beam_search(args, model, tokenizer, input_features)
    else:
        raise Exception
    print("Generate text: {}".format(gen_text))

if __name__ == "__main__":
    main()
