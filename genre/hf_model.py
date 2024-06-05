# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pdb
import logging
from typing import List, Dict

import torch
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    XLMRobertaTokenizer,
    MBartForConditionalGeneration,
)

from genre.utils import chunk_it, post_process_wikidata

logger = logging.getLogger(__name__)


class _GENREHubInterface:
    def sample(
        self,
        sentences: List[str],
        num_beams: int = 5,
        num_return_sequences=5,
        text_to_id: Dict[str, str] = None,
        marginalize: bool = False,
        **kwargs
    ) -> List[str]:

        input_args = {
            k: v.to(self.device)
            for k, v in self.tokenizer.batch_encode_plus(
                sentences, padding=True, return_tensors="pt", add_special_tokens=False,
            ).items()
        }

        outputs = self.generate(
            **input_args,
            min_length=0,
            max_length=1024,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True,
            # bos_token_id=0,
            # eos_token_id=50118,
            # bad_words_ids=[[0]],
            # decoder_start_token_id=2,
            no_repeat_ngram_size=-1,
            forced_bos_token_id=None,
            # forced_eos_token_id=2,
            **kwargs
        )

        # pdb.set_trace()

        outputs = chunk_it(
            [
                {"text": text, "score": score,}
                for text, score in zip(
                    self.tokenizer.batch_decode(
                        outputs.sequences, skip_special_tokens=True
                    ),
                    outputs.sequences_scores,
                )
            ],
            len(sentences),
        )

        outputs = post_process_wikidata(
            outputs, text_to_id=text_to_id, marginalize=marginalize
        )

        return outputs

    def encode(self, sentence):
        return self.tokenizer.encode(sentence, return_tensors="pt")[0]

class GENREHubInterface(_GENREHubInterface, BartForConditionalGeneration):
    pass
    
class mGENREHubInterface(_GENREHubInterface, MBartForConditionalGeneration):
    pass

class GENRE(BartForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, model_name_or_path, adapter_name_or_path_list=None, checkpoint_file=None):
        model = GENREHubInterface.from_pretrained(model_name_or_path)
        model.tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
        if adapter_name_or_path_list:
            for adapter_path in adapter_name_or_path_list:
                adapter_name = model.load_adapter(adapter_path)
                model.set_active_adapters(adapter_name)
                print(f'load {adapter_name}')
        '''special setting for Adapter stack'''
        # import transformers.adapters.composition as ac
        # model.active_adapters = ac.Stack("fc", "qa", "el", "sf", "dlg")

        '''special setting for Adapter fusion'''
        # import transformers.adapters.composition as ac
        # adapter_setup = ac.Fuse("fc", "qa", "el", "sf", "dlg")
        # model.load_adapter_fusion('models/R0-fusion/', set_active=True, with_head=False)
        return model


class mGENRE(MBartForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        model = mGENREHubInterface.from_pretrained(model_name_or_path)
        model.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name_or_path)
        return model
