from typing import Optional, Union, List, Tuple, Dict
from time import time
from tqdm import tqdm
import copy
import json
import os
import torch
import numpy as np
import random
from ..models.melo.melo import LORA
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
from transformers import LlamaTokenizer,PreTrainedTokenizerFast, LlamaTokenizerFast
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from .utils import _chunks, _prepare_requests, summary_metrics
from .batch_editor import BatchEditor
from ..evaluate import compute_edit_quality, compute_icl_edit_quality, compute_sent_metric
from ..util import nethook, perplexity
from ..util.hparams import HyperParams
from ..util.alg_dict import *
from ..evaluate.evaluate_utils import test_generation_quality
from ..models.memit_recursive.memit_recursive_main import execute_memit

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)
def make_logs():

    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
seed_everything(42)
  
class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self, hparams: HyperParams):
        assert hparams is not None, 'Error: hparams is None.'
        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        make_logs()
        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            
            # QLoRA configuration
            if hparams.alg_name == 'QLoRA':
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=(hparams.quantization_bit == 4),
                    bnb_4bit_use_double_quant=hparams.double_quant,
                    bnb_4bit_quant_type=hparams.quant_type,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                model_kwargs = {
                    "quantization_config": bnb_config,
                    "torch_dtype": torch_dtype,
                    "device_map": {'': hparams.device}
                }
            else:
                model_kwargs = {
                    "torch_dtype": torch_dtype,
                    "device_map": device_map
                }

            if 't5' in self.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
                self.tok = T5Tokenizer.from_pretrained(self.model_name)
            elif 'chatglm-api' in self.model_name.lower():
                self.model, self.tok = None, None
                self.hparams = hparams
                return
            elif 'gpt-3.5' in self.model_name.lower():
                self.model, self.tok = None, None
            elif 'gpt' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'llama' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'baichuan' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs, trust_remote_code=True)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'chatglm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                if 'chatglm2'in self.model_name.lower(): 
                    self.tok.unk_token_id = 64787
                else: 
                    self.tok.pad_token_id = self.tok.eos_token_id
            elif 'internlm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'qwen2' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch_dtype if hparams.alg_name not in ['MEND'] else torch.bfloat16, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
            elif 'qwen' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,fp32=False,trust_remote_code=True, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
            elif 'mistral' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            else:
                raise NotImplementedError

            if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer) or isinstance(self.tok, LlamaTokenizerFast) or isinstance(self.tok, PreTrainedTokenizerFast)) and (hparams.alg_name not in ['ROME', 'MEMIT', 'EMMET', 'R-ROME', 'MEMIT_RECURSIVE', 'MEMIT_RECURSIVE_SPREAD', 'PMET_RECURSIVE', 'ROME_RECURSIVE', 'MEMIT_RECURSIVE_NEIGHBOR', 'PMET_RECURSIVE_NEIGHBOR', 'AlphaEdit_RECURSIVE', 'AlphaEdit_RECURSIVE_NEIGHBOR']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'
            if self.tok is not None and ('mistral' in self.model_name.lower() or 'llama' in self.model_name.lower() or 'qwen' in self.model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT', 'EMMET', 'R-ROME', 'MEMIT_RECURSIVE', 'MEMIT_RECURSIVE_SPREAD','PMET_RECURSIVE', 'ROME_RECURSIVE', 'MEMIT_RECURSIVE_NEIGHBOR', 'PMET_RECURSIVE_NEIGHBOR', 'AlphaEdit_RECURSIVE', 'AlphaEdit_RECURSIVE_NEIGHBOR']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
                self.tok.padding_side = 'right'
        else:
            self.model, self.tok = self.model_name

        if hparams.model_parallel: 
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device') and hparams.alg_name != 'QLoRA':
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams

    def edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             target_neg: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs:  Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             sequential_edit=False,
             verbose=True,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        test_generation = kwargs.pop('test_generation', False)

        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        if hasattr(self.hparams, 'batch_size') and not BatchEditor.is_batchable_method(self.alg_name):  # For Singleton Editing, bs=1
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'

        if ground_truth is not None:
            ground_truth = [ground_truth,] if isinstance(ground_truth, str) else ground_truth
        else:# Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>'] * (len(prompts))

        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = _prepare_requests(prompts, target_new, ground_truth, target_neg, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)

        return self.edit_requests(requests, sequential_edit, verbose, test_generation=test_generation, **kwargs)

    def batch_edit(self,
                   prompts: List[str],
                   target_new: List[str],
                   ground_truth: Optional[List[str]] = None,
                   target_neg: Optional[List[str]] = None,
                   rephrase_prompts: Optional[List[str]] = None,
                   locality_inputs: Optional[Dict] = None,
                   portability_inputs: Optional[Dict] = None,
                   sequential_edit=False,
                   verbose=True,
                   **kwargs
                   ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        assert BatchEditor.is_batchable_method(self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'

        requests = _prepare_requests(prompts, target_new, ground_truth, target_neg, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)

        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'
        all_metrics = []
        for record_chunks in _chunks(requests, self.hparams.batch_size):
            start = time()

            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True
            )
            exec_time = time() - start
            LOG.info(f"Execution editing took {exec_time}")

            start = time()
            chunk_metrics = []
            for i, request in enumerate(record_chunks):

                metrics = {
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation),
                }

                chunk_metrics.append(metrics)

            if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                with torch.no_grad():
                    weights_copy()
            elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                edited_model.unload()
                del self.model.peft_config
            elif self.alg_name == 'MELO':
                self.model = edited_model
            elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                self.model = edited_model
            else:
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            for i, request in enumerate(record_chunks):
                chunk_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation)

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {chunk_metrics[i]}"
                    )

            LOG.info(f"Evaluation took {time() - start}")
            all_metrics.extend(chunk_metrics)
        return all_metrics, edited_model, weights_copy
    

    def batch_edit_recursive(self,
                   prompts: List[str],
                   target_new: List[str],
                   iterations: int,
                   ds_name: str,
                   num_edits,
                   ds_subset,
                   neighborhood_samples,
                   ground_truth: Optional[List[str]] = None,
                   target_neg: Optional[List[str]] = None,
                   rephrase_prompts: Optional[List[str]] = None,
                   locality_inputs: Optional[Dict] = None,
                   portability_inputs: Optional[Dict] = None,
                   sequential_edit=False,
                   verbose=False,
                   **kwargs
                   ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        if ds_name == "zsre":
            requests_zsre = copy.deepcopy(prompts)
            prompts = []
            for case_id, request in enumerate(requests_zsre):
                prompts.append(request['src'])
                ground_truth.append("<|endoftext|>")
                target_new.append(request["answers"][0])
                # ds_subject.append(request["subject"])
                rephrase_prompts.append([request["rephrase"]])
                ans_toks = self.tok.encode(request["loc_ans"], return_tensors="pt", add_special_tokens=False)[0]
                if ans_toks[0] == self.tok.bos_token_id or ans_toks[0] == self.tok.unk_token_id:
                    ans_toks = ans_toks[1:]
                neighbor_promts = []
                neighbor_gts = []
                for i in range(len(ans_toks)):
                    neighbor_promts.append(request["loc"] + "?" + self.tok.decode(ans_toks[:i]))
                    neighbor_gts.append(self.tok.decode(ans_toks[i]))
                locality_inputs['neighborhood']['prompt'].append(neighbor_promts)
                locality_inputs['neighborhood']['ground_truth'].append(neighbor_gts)
        assert self.hparams.batch_size == num_edits
        assert len(prompts) == len(target_new)
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        assert BatchEditor.is_batchable_method(self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'

        requests = _prepare_requests(prompts, target_new, ground_truth, target_neg, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)
        print(requests[0])

        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'
        
        ppl_check_file = os.path.join('data', 'ME-PPL_50.json')
        with open(ppl_check_file, 'r') as file:
            ppl_check_datas = json.load(file)
        
        m_ppl_pre = 0
        for ppl_check_data in ppl_check_datas:
            text = ppl_check_data["Text"]
            m_ppl_pre += perplexity.perplexity(self.model, self.tok, text)
        m_ppl_pre /= len(ppl_check_datas)
        
        pre_metrics = []
        requests_drop_lst = []
        for record_chunks in _chunks(requests, self.hparams.batch_size):
            for i, request in enumerate(record_chunks):
                metrics = {                      
                    "pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, None, test_generation=test_generation),
                }
                """This portion filters the data which are eligible for neighbor-assisted model editing"""
                # if 'locality' in metrics['pre'].keys():
                #     for locality_key in request['locality'].keys():
                #         neighbor_drop = []                       
                #         for i_idxs, token_idxs in enumerate(metrics['pre']['locality'][f'{locality_key}_output']):
                #             request['locality'][locality_key]["ground_truth"][i_idxs] = self.tok.decode(token_idxs).strip()
                #             # uncomment this when filtering neighbors targets equal to ground truth 
                #             if 'NEIGHBOR' in self.alg_name:
                #                 if request['locality'][locality_key]["ground_truth"][i_idxs] != request["ground_truth"]:                                    
                #                     neighbor_drop.append(i_idxs)
                        
                #         # uncomment this when filtering neighbors targets equal to ground truth
                #         if 'NEIGHBOR' in self.alg_name:
                #             neighborhood_samples[i] = [n for i_n, n in enumerate(neighborhood_samples[i]) if i_n not in neighbor_drop]   
                #         if len(neighborhood_samples[i]) <= 1:
                #             requests_drop_lst.append(i)
                #         else:
                #             record_chunks[i]['locality'][locality_key]['prompt'] = [n for i_n, n in enumerate(record_chunks[i]['locality'][locality_key]['prompt']) if i_n not in neighbor_drop]
                #             record_chunks[i]['locality'][locality_key]['ground_truth'] = [n for i_n, n in enumerate(record_chunks[i]['locality'][locality_key]['ground_truth']) if i_n not in neighbor_drop]
                #             metrics['pre']['locality'][f'{locality_key}_output'] = [n for i_n, n in enumerate(metrics['pre']['locality'][f'{locality_key}_output']) if i_n not in neighbor_drop]
                #             metrics['pre']['locality'][f'{locality_key}_success'] = [n for i_n, n in enumerate(metrics['pre']['locality'][f'{locality_key}_success']) if i_n not in neighbor_drop]

                #             del record_chunks[i]['locality'][locality_key]['prompt'][0]
                #             del record_chunks[i]['locality'][locality_key]['ground_truth'][0]
                #             del metrics['pre']['locality'][f'{locality_key}_output'][0]
                #             del metrics['pre']['locality'][f'{locality_key}_success'][0]
                #             record_chunks[i]["neighborhood_samples"] = [neighborhood_samples[i][0]]
                """End of filtering data"""
                if  i not in requests_drop_lst:           
                    pre_metrics.append(metrics)
        
        
        requests = [n for i_n, n in enumerate(requests) if i_n not in requests_drop_lst]
        """This portion saves the data which are eligible for neighbor-assisted model editing"""
        # _id = 0
        # for request in requests:
        #     request["case_id"] = _id
        #     _id += 1

        # if 'NEIGHBOR' in self.alg_name:
        #     save_path = os.path.join('.', 'data', f'{ds_name}_random_subsets_{len(requests)}')
        #     os.makedirs(save_path, exist_ok=True)
        #     save_path = os.path.join(save_path, f'random_subset_{len(requests)}.json')
        #     print(save_path)
        #     with open(save_path, 'w') as file:
        #         json.dump(requests, file, indent=4)
        """End of saving data"""

        if 'NEIGHBOR' in self.alg_name:
            assert len(neighborhood_samples) == len(requests)
            for i, request in enumerate(requests):
                request["neighborhood_samples"] = neighborhood_samples[i]
        print(requests[0])
        
        all_summary = []
        all_summary_false_case = []
        false_case_ids = []
        for run_i in range(1, iterations+1):
                        
            for record_chunks in _chunks(requests, self.hparams.batch_size):
                start = time()

                edited_model, weights_copy, init_ppl_lst, final_ppl_lst, context_templates = self.apply_algo(
                    self.model,
                    self.tok,
                    record_chunks,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                )
                exec_time = time() - start
                LOG.info(f"Execution editing took {exec_time}")
                # # verify if init_ppl_lst is correctly computed : Remove later
                # original_v_num_grad_steps = self.hparams.v_num_grad_steps
                # self.hparams.v_num_grad_steps = 2
                # _deltas, _init_ppl_lst, _final_ppl_lst, _context_templates = execute_memit(
                #     edited_model, 
                #     self.tok, 
                #     record_chunks, 
                #     self.hparams,
                #     skip_spread=True, 
                # )
                # self.hparams.v_num_grad_steps = original_v_num_grad_steps

                start = time()
                chunk_metrics = []
                for i, request in enumerate(record_chunks):
                    metrics = {
                        'case_id': i,
                        'time': exec_time,
                        'requested_rewrite': request,                        
                        'pre': pre_metrics[i]["pre"],
                        'post': compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, context_templates, test_generation=test_generation),
                    }

                    # metrics['requested_rewrite']['context_templates'] = context_templates
                    
                    metrics['post']['init_ppl'] = init_ppl_lst[i]
                    metrics['post']['final_ppl'] = final_ppl_lst[i]
                    # metrics['post']['init_ppl_verify'] = _init_ppl_lst[i]
                    metrics['post'].update(compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, context_templates, 'ppl', test_generation=test_generation))
                    
                    if 'locality' in metrics['post'].keys():
                        for locality_key in request['locality'].keys():
                            locality_result = []
                            for ans, label in zip(metrics['post']['locality'][f'{locality_key}_output'], metrics['pre']['locality'][f'{locality_key}_output']):
                                locality_result.append(np.mean(np.array_equal(ans, label)))
                            metrics['post']['locality'][f'{locality_key}_acc'] = locality_result
                    
                    chunk_metrics.append(metrics)
                    
                    if run_i == 1:
                        if metrics['post']['rewrite_acc'][0] == 0.0:
                            false_case_ids.append(metrics['case_id'])

                LOG.info(f"Evaluation took {time() - start}")
                summary = summary_metrics(self.model_name.split('/')[-1], self.alg_name, ds_name, run_i, num_edits, ds_subset, iterations, chunk_metrics)
                chunk_metrics_false_case = []
                for chunk_metric in chunk_metrics:
                    if chunk_metric['case_id'] in false_case_ids:
                        chunk_metrics_false_case.append(chunk_metric)
                if len(chunk_metrics_false_case) > 0:
                    summary_false_case = summary_metrics(self.model_name.split('/')[-1], self.alg_name, ds_name, run_i, num_edits, ds_subset, iterations, chunk_metrics_false_case, suffix="false_cases")
                else:
                    summary_false_case = {}
                m_ppl = 0
                for ppl_check_data in ppl_check_datas:
                    text = ppl_check_data["Text"]
                    m_ppl += perplexity.perplexity(edited_model, self.tok, text)
                m_ppl /= len(ppl_check_datas)
                summary["m_ppl"] = m_ppl
                summary["m_ppl_pre"] = m_ppl_pre
                
                all_summary.append(summary)
                if len(summary_false_case)>0:
                    summary_false_case["m_ppl"] = m_ppl
                    summary_false_case["m_ppl_pre"] = m_ppl_pre
                    all_summary_false_case.append(summary_false_case)
                self.model = edited_model
        return all_summary, all_summary_false_case, edited_model, weights_copy
    
    def batch_edit_recursive_spread(self,
                   prompts: List[str],
                   target_new: List[str],
                   iterations: int,
                   ds_name: str,
                   num_edits,
                   ds_subset,
                   neighborhood_samples,
                   ground_truth: Optional[List[str]] = None,
                   target_neg: Optional[List[str]] = None,
                   rephrase_prompts: Optional[List[str]] = None,
                   locality_inputs: Optional[Dict] = None,
                   portability_inputs: Optional[Dict] = None,
                   sequential_edit=False,
                   verbose=False,
                   **kwargs
                   ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        print("SPREAD IT")
        if ds_name == "zsre":
            requests_zsre = copy.deepcopy(prompts)
            prompts = []
            for case_id, request in enumerate(requests_zsre):
                prompts.append(request['src'])
                ground_truth.append("<|endoftext|>")
                target_new.append(request["answers"][0])
                # ds_subject.append(request["subject"])
                rephrase_prompts.append([request["rephrase"]])
                ans_toks = self.tok.encode(request["loc_ans"], return_tensors="pt", add_special_tokens=False)[0]
                if ans_toks[0] == self.tok.bos_token_id or ans_toks[0] == self.tok.unk_token_id:
                    ans_toks = ans_toks[1:]
                neighbor_promts = []
                neighbor_gts = []
                for i in range(len(ans_toks)):
                    neighbor_promts.append(request["loc"] + "?" + self.tok.decode(ans_toks[:i]))
                    neighbor_gts.append(self.tok.decode(ans_toks[i]))
                locality_inputs['neighborhood']['prompt'].append(neighbor_promts)
                locality_inputs['neighborhood']['ground_truth'].append(neighbor_gts)
        assert self.hparams.batch_size == num_edits
        assert len(prompts) == len(target_new)
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        assert BatchEditor.is_batchable_method(self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'

        requests = _prepare_requests(prompts, target_new, ground_truth, target_neg, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)
        print(requests[0])

        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'
        
        ppl_check_file = os.path.join('data', 'ME-PPL_50.json')
        with open(ppl_check_file, 'r') as file:
            ppl_check_datas = json.load(file)
        
        m_ppl_pre = 0
        for ppl_check_data in ppl_check_datas:
            text = ppl_check_data["Text"]
            m_ppl_pre += perplexity.perplexity(self.model, self.tok, text)
        m_ppl_pre /= len(ppl_check_datas)
        
        pre_metrics = []
        requests_drop_lst = []
        for record_chunks in _chunks(requests, self.hparams.batch_size):
            for i, request in enumerate(record_chunks):
                metrics = {                      
                    "pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, None, test_generation=test_generation),
                }
                if 'locality' in metrics['pre'].keys():
                    for locality_key in request['locality'].keys():
                        neighbor_drop = []                       
                        for i_idxs, token_idxs in enumerate(metrics['pre']['locality'][f'{locality_key}_output']):
                            request['locality'][locality_key]["ground_truth"][i_idxs] = self.tok.decode(token_idxs).strip()
                            # uncomment this when filtering neighbors targets equal to ground truth 
                            if 'NEIGHBOR' in self.alg_name:
                                if request['locality'][locality_key]["ground_truth"][i_idxs] != request["ground_truth"]:                                    
                                    neighbor_drop.append(i_idxs)
                        
                        # uncomment this when filtering neighbors targets equal to ground truth
                        if 'NEIGHBOR' in self.alg_name:
                            neighborhood_samples[i] = [n for i_n, n in enumerate(neighborhood_samples[i]) if i_n not in neighbor_drop]   
                        # if len(neighborhood_samples[i]) <= 1:
                        #     requests_drop_lst.append(i)
                        # else:
                            record_chunks[i]['locality'][locality_key]['prompt'] = [n for i_n, n in enumerate(record_chunks[i]['locality'][locality_key]['prompt']) if i_n not in neighbor_drop]
                            record_chunks[i]['locality'][locality_key]['ground_truth'] = [n for i_n, n in enumerate(record_chunks[i]['locality'][locality_key]['ground_truth']) if i_n not in neighbor_drop]
                            metrics['pre']['locality'][f'{locality_key}_output'] = [n for i_n, n in enumerate(metrics['pre']['locality'][f'{locality_key}_output']) if i_n not in neighbor_drop]
                            metrics['pre']['locality'][f'{locality_key}_success'] = [n for i_n, n in enumerate(metrics['pre']['locality'][f'{locality_key}_success']) if i_n not in neighbor_drop]

                        #     del record_chunks[i]['locality'][locality_key]['prompt'][0]
                        #     del record_chunks[i]['locality'][locality_key]['ground_truth'][0]
                        #     del metrics['pre']['locality'][f'{locality_key}_output'][0]
                        #     del metrics['pre']['locality'][f'{locality_key}_success'][0]
                            record_chunks[i]["neighborhood_samples"] = [neighborhood_samples[i][0]]
                if  i not in requests_drop_lst:           
                    pre_metrics.append(metrics)
        
        requests = [n for i_n, n in enumerate(requests) if i_n not in requests_drop_lst]

        # if 'NEIGHBOR' in self.alg_name:
        # save_path = os.path.join('.', 'data', f'{ds_name}_random_subsets_{len(requests)}')
        # os.makedirs(save_path, exist_ok=True)
        # save_path = os.path.join(save_path, f'random_subsets_{len(requests)}.json')
        # with open(save_path, 'w') as file:
        #     json.dump(requests, file, indent=4)

        print(requests[0])
        
        all_summary = []
        all_summary_false_case = []
        false_case_ids = []
        deltas = None
        init_ppl_lst = None
        final_ppl_lst = None
        context_templates = None
        for run_i in range(1, iterations+1):
                        
            for record_chunks in _chunks(requests, self.hparams.batch_size):
                start = time()

                edited_model, weights_copy, init_ppl_lst, final_ppl_lst, context_templates, deltas = self.apply_algo(
                    self.model,
                    self.tok,
                    record_chunks,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    iterations=run_i,
                    deltas=deltas,
                    init_ppl_lst=init_ppl_lst,
                    final_ppl_lst=final_ppl_lst,
                    context_templates=context_templates,
                )
                exec_time = time() - start
                LOG.info(f"Execution editing took {exec_time}")
                original_v_num_grad_steps = self.hparams.v_num_grad_steps
                self.hparams.v_num_grad_steps = 2
                _deltas, _init_ppl_lst, _final_ppl_lst, _context_templates = execute_memit(
                    edited_model, 
                    self.tok, 
                    record_chunks, 
                    self.hparams,
                    skip_spread=True, 
                )
                self.hparams.v_num_grad_steps = original_v_num_grad_steps

                start = time()
                chunk_metrics = []
                for i, request in enumerate(record_chunks):
                    metrics = {
                        'case_id': i,
                        'time': exec_time,
                        'requested_rewrite': request,                        
                        'pre': pre_metrics[i]["pre"],
                        'post': compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, context_templates, test_generation=test_generation),
                    }

                    # metrics['requested_rewrite']['context_templates'] = context_templates
                    
                    metrics['post']['init_ppl'] = init_ppl_lst[i]
                    metrics['post']['final_ppl'] = final_ppl_lst[i]
                    metrics['post']['init_ppl_verify'] = _init_ppl_lst[i]
                    metrics['post'].update(compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, context_templates, 'ppl', test_generation=test_generation))
                    
                    if 'locality' in metrics['post'].keys():
                        for locality_key in request['locality'].keys():
                            locality_result = []
                            for ans, label in zip(metrics['post']['locality'][f'{locality_key}_output'], metrics['pre']['locality'][f'{locality_key}_output']):
                                locality_result.append(np.mean(np.array_equal(ans, label)))
                            metrics['post']['locality'][f'{locality_key}_acc'] = locality_result
                    
                    chunk_metrics.append(metrics)
                    
                    if run_i == 1:
                        if metrics['post']['rewrite_acc'][0] == 0.0:
                            false_case_ids.append(metrics['case_id'])

                LOG.info(f"Evaluation took {time() - start}")
                summary = summary_metrics(self.model_name.split('/')[-1], self.alg_name, ds_name, run_i, num_edits, ds_subset, iterations, chunk_metrics)
                chunk_metrics_false_case = []
                for chunk_metric in chunk_metrics:
                    if chunk_metric['case_id'] in false_case_ids:
                        chunk_metrics_false_case.append(chunk_metric)
                if len(chunk_metrics_false_case) > 0:
                    summary_false_case = summary_metrics(self.model_name.split('/')[-1], self.alg_name, ds_name, run_i, num_edits, ds_subset, iterations, chunk_metrics_false_case, suffix="false_cases")
                else:
                    summary_false_case = {}
                m_ppl = 0
                for ppl_check_data in ppl_check_datas:
                    text = ppl_check_data["Text"]
                    m_ppl += perplexity.perplexity(edited_model, self.tok, text)
                m_ppl /= len(ppl_check_datas)
                summary["m_ppl"] = m_ppl
                summary["m_ppl_pre"] = m_ppl_pre
                
                all_summary.append(summary)
                if len(summary_false_case)>0:
                    summary_false_case["m_ppl"] = m_ppl
                    summary_false_case["m_ppl_pre"] = m_ppl_pre
                    all_summary_false_case.append(summary_false_case)
                self.model = edited_model
        return all_summary, all_summary_false_case, edited_model, weights_copy

    def edit_requests(self,
             requests,
             sequential_edit=False,
             verbose=True,
             test_generation=False,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'
        all_metrics = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        else:
            for i, request in enumerate(tqdm(requests)):
                if self.alg_name == 'IKE':
                    assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                    metrics = {"pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''], request, self.hparams.device, pre_edit=True)}
                else:
                    metrics = {"pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)}
                all_metrics.append(metrics)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)

        def edit_func(request):
            if self.alg_name == 'IKE':
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                icl_examples = None
            return edited_model, weights_copy, icl_examples

        def edit_evaluation(all_metrics, request, edited_model, idx, test_generation, icl_examples, **kwargs):
            eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
            if self.alg_name == 'IKE':
                all_metrics[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples, request, self.hparams.device),
                })
            else:
                all_metrics[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation),
                })
                if "metric_kwargs" in kwargs:
                    all_metrics[idx].update(compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok,metric_kwargs=kwargs["metric_kwargs"][idx], device=self.hparams.device))
                if 'locality' in all_metrics[idx]['post'].keys():
                    for locality_key in request['locality'].keys():
                        locality_result = []
                        for ans, label in zip(all_metrics[idx]['post']['locality'][f'{locality_key}_output'], all_metrics[idx]['pre']['locality'][f'{locality_key}_output']):
                            locality_result.append(np.mean(np.equal(ans, label)))
                        all_metrics[idx]['post']['locality'][f'{locality_key}_acc'] = locality_result
                        all_metrics[idx]['post']['locality'].pop(f'{locality_key}_output')
                    all_metrics[idx]['pre'].pop('locality')

            if verbose:
                LOG.info(f"{idx} editing: {request['prompt']} -> {request['target_new']}  \n\n {all_metrics[idx]}")


        if sequential_edit:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
            if self.alg_name == 'WISE' and hasattr(self.hparams, 'save_path') and self.hparams.save_path:
                print("Start saving the WISE model!")
                edited_model.save(self.hparams.save_path)
            for i, request in enumerate(requests):
                edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, **kwargs)
        else:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
                edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, **kwargs)
                if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                    with torch.no_grad():
                        weights_copy()
                elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                    edited_model.unload()
                    del self.model.peft_config
                elif self.alg_name == 'MELO':
                    self.model = edited_model
                elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                    self.model = edited_model
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")


        if isinstance(edited_model, LORA):
            edited_model = edited_model.model
        if len(all_metrics) != 0:
            summary_metrics(all_metrics)

        return all_metrics, edited_model, weights_copy

    def normal_edit(
        self,
        prompts: List[str],
        target_new: List[str],
        sequential_edit=False,
    ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        assert BatchEditor.is_batchable_method(self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'

        requests = _prepare_requests(prompts, target_new, ground_truth)

        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'

        # print(f"[editor.py][batch_edit] `batch_size`={self.hparams.batch_size}")
        # for epc in range(epoch):
        #     print(f"[editor.py][batch_edit] `Epoch` = {epc+1}")
        #     for record_chunks in self._chunks(requests, self.hparams.batch_size):
        start = time()

        edited_model, weights_copy = self.apply_algo(
            self.model,
            self.tok,
            requests,  # record_chunks -> requests
            self.hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=False,
        )
        exec_time = time() - start
        LOG.info(f"Execution editing took {exec_time}")

        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        return None, edited_model, weights_copy
    
    def generate_edit(
        self,
        prompts: Union[str, List[str]],
        target_new: Union[str, List[str]],
        ground_truth: Optional[Union[str, List[str]]] = None,
        target_neg: Optional[Union[str, List[str]]] = None,
        rephrase_prompts: Optional[Union[str, List[str]]] = None,
        locality_inputs:  Optional[Dict] = None,
        portability_inputs: Optional[Dict] = None,
        sequential_edit=False,
        verbose=True,
        **kwargs
    ):
        eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        test_generation = kwargs.pop('test_generation', False)

        assert len(prompts) == len(target_new)

        if hasattr(self.hparams, 'batch_size'):
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'
        
        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = _prepare_requests(prompts, target_new, ground_truth, target_neg, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)
        
        def text_generate(
            model,
            model_name,
            hparams: HyperParams,
            tok: AutoTokenizer,
            query,
            device,
            eval_metric: str = 'token_em',
            test_generation = False
        ):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
            text = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tok.encode(text, return_tensors="pt").to(f"cuda:{device}")
            template_length = len(model_inputs[0])
            generated_ids = model.generate(
                input_ids=model_inputs,
                max_new_tokens=512
            )
            trimmed_generated_ids = generated_ids[0][template_length:]
            response = tok.decode(trimmed_generated_ids, skip_special_tokens=True)
            return response

        all_results = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            results = kwargs['pre_edit']
            all_results = results
        else:
            for i, request in enumerate(tqdm(requests)):
                results = {}
                results['pre'] = {}
                results['pre']['rewrite_ans'] = text_generate(self.model, self.model_name, self.hparams, self.tok, request['prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                results['pre']['rephrase_ans'] = text_generate(self.model, self.model_name, self.hparams, self.tok, request['rephrase_prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                por_results = []
                for pr in request['portability']['por_hop']['prompt']:
                    por_results.append(text_generate(self.model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
                if 'locality' in request.keys() and 'loc_hop' in request['locality'].keys():
                    loc_results = []
                    for pr in request['locality']['loc_hop']['prompt']:
                        loc_results.append(text_generate(self.model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
                    results['pre']['locality_ans'] = loc_results
                results['pre']['portability_ans'] = por_results
                all_results.append(results)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                json.dump(all_results, open(kwargs['pre_file'], 'w'), indent=4)

        def edit_func(request):
            if self.alg_name == 'IKE':
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                icl_examples = None
            return edited_model, weights_copy, icl_examples
        
        def post_edit_results(all_results, request, edited_model, idx, eval_metric, test_generation, icl_examples, **kwargs):
            if self.alg_name == 'IKE':
                all_results[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples, request, self.hparams.device),
                })
            else:
                results_post = {}
                results_post['rewrite_ans'] = text_generate(edited_model, self.model_name, self.hparams, self.tok, request['prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                results_post['rephrase_ans'] = text_generate(edited_model, self.model_name, self.hparams, self.tok, request['rephrase_prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                por_results = []
                for pr in request['portability']['por_hop']['prompt']:
                    por_results.append(text_generate(edited_model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
                if 'locality' in request.keys() and 'loc_hop' in request['locality'].keys():
                    loc_results = []
                    for pr in request['locality']['loc_hop']['prompt']:
                        loc_results.append(text_generate(edited_model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
                    results_post['locality_ans'] = loc_results
                results_post['portability_ans'] = por_results
                if test_generation:
                    if self.hparams.alg_name == 'GRACE':
                        results_post['fluency'] = test_generation_quality(model=edited_model,tok=self.tok,prefixes=request['prompt'] if isinstance(request['prompt'],list) else [request['prompt'],], max_out_len=100, vanilla_generation=True)
                    else:
                        results_post['fluency'] = test_generation_quality(model=edited_model,tok=self.tok,prefixes=request['prompt'] if isinstance(request['prompt'],list) else [request['prompt'],], max_out_len=100, vanilla_generation=False)
                all_results[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": results_post
                })
            if verbose:
                LOG.info(f"{idx} editing: {request['prompt']} -> {request['target_new']}")

        if sequential_edit:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
            for i, request in enumerate(requests):
                post_edit_results(all_results, request, edited_model, i, eval_metric, test_generation, icl_examples, **kwargs)
        else:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
                post_edit_results(all_results, request, edited_model, i, eval_metric, test_generation, icl_examples, **kwargs)
                if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                    with torch.no_grad():
                        weights_copy()
                elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                    edited_model.unload()
                    del self.model.peft_config
                elif self.alg_name == 'MELO':
                    self.model = edited_model
                elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                    self.model = edited_model
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        if isinstance(edited_model, LORA):
            edited_model = edited_model.model
        if len(all_results) != 0:
            summary_metrics(all_results)

        return all_results, edited_model, weights_copy

    def deep_edit(
        self,
        datasets
    ):
        metrics = self.apply_algo(datasets, self.hparams)
        return metrics


