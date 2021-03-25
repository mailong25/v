import os
from parlai.core.agents import create_agent

class Blender:
    
    def __init__(self, model_file, parlai_home, include_personas = False):
        opt  = {'init_opt': None, 'allow_missing_init_opts': False, 'task': 'blended_skill_talk', 'download_path': None, 
        'loglevel': 'info', 'datatype': 'train', 'image_mode': 'raw', 'hide_labels': False, 'multitask_weights': [1],
        'batchsize': 1, 'dynamic_batching': None, 'verbose': False, 'datapath': '/mnt/disks/blender/ParlAI/data', 
        'model': None, 'model_file': '/mnt/disks/blender/ParlAI/data/models/blender/blender_1Bdistill/model', 
        'init_model': None, 'dict_class': 'parlai.core.dict:DictionaryAgent', 'display_examples': False, 
        'display_prettify': False, 'display_add_fields': '', 'interactive_task': True, 'outfile': '', 
        'save_format': 'conversations', 'local_human_candidates_file': None, 'single_turn': False, 
        'log_keep_fields': 'all', 'image_size': 256, 'image_cropsize': 224, 'embedding_size': 300, 'n_layers': 2, 
        'ffn_size': 300, 'dropout': 0.0, 'attention_dropout': 0.0, 'relu_dropout': 0.0, 'n_heads': 2, 
        'learn_positional_embeddings': False, 'embeddings_scale': True, 'n_positions': None, 'n_segments': 0, 
        'variant': 'aiayn', 'activation': 'relu', 'output_scaling': 1.0, 'share_word_embeddings': True, 
        'n_encoder_layers': -1, 'n_decoder_layers': -1, 'model_parallel': False, 'beam_size': 1, 'beam_min_length': 1, 
        'beam_context_block_ngram': -1, 'beam_block_ngram': -1, 'beam_block_full_context': True, 
        'beam_length_penalty': 0.65, 'skip_generation': False, 'inference': 'greedy', 'topk': 10, 
        'topp': 0.9, 'beam_delay': 30, 'beam_block_list_filename': None, 'temperature': 1.0, 
        'compute_tokenized_bleu': False, 'interactive_mode': True, 'embedding_type': 'random', 
        'embedding_projection': 'random', 'fp16': False, 'fp16_impl': 'apex', 'force_fp16_tokens': False, 
        'optimizer': 'sgd', 'learningrate': 1, 'gradient_clip': 0.1, 'adam_eps': 1e-08, 'adafactor_eps': (1e-30, 0.001), 
        'momentum': 0, 'nesterov': True, 'nus': (0.7,), 'betas': (0.9, 0.999), 'weight_decay': None, 
        'rank_candidates': False, 'truncate': -1, 'text_truncate': None, 'label_truncate': None, 
        'history_reversed': False, 'history_size': -1, 'person_tokens': False, 'split_lines': False, 
        'use_reply': 'label', 'add_p1_after_newln': False, 'delimiter': '\n', 'history_add_global_end_token': None, 
        'special_tok_lst': None, 'gpu': -1, 'no_cuda': False, 'dict_file': None, 'dict_initpath': None, 
        'dict_language': 'english', 'dict_max_ngram_size': -1, 'dict_minfreq': 0, 'dict_maxtokens': -1, 
        'dict_nulltoken': '__null__', 'dict_starttoken': '__start__', 'dict_endtoken': '__end__', 
        'dict_unktoken': '__unk__', 'dict_tokenizer': 're', 'dict_lower': False, 'bpe_debug': False, 
        'dict_textfields': 'text,labels', 'bpe_vocab': None, 'bpe_merge': None, 'bpe_add_prefix_space': None, 
        'bpe_dropout': None, 'lr_scheduler': 'reduceonplateau', 'lr_scheduler_patience': 3, 'lr_scheduler_decay': 0.5, 
        'max_lr_steps': -1, 'invsqrt_lr_decay_gamma': -1, 'warmup_updates': -1, 'warmup_rate': 0.0001, 'update_freq': 1, 
        'display_partner_persona': True, 'include_personas': True, 'include_initial_utterances': False, 
        'safe_personas_only': False, 'parlai_home': '/mnt/disks/blender/ParlAI', 
        'override': {'task': 'blended_skill_talk', 'model_file': '/mnt/disks/blender/ParlAI/data/models/blender/blender_1Bdistill/model'}, 'starttime': 'Dec31_03-08'}

        opt['model_file'] = model_file
        opt['parlai_home'] = parlai_home
        opt['datapath'] = os.path.join(parlai_home,'data')
        opt['include_personas'] = include_personas
        opt['override']['model_file'] = model_file
        self.opt = opt
        
        self.bot_agent = create_agent(opt, requireModelExists=True)
        
    def predict(self,contexts):
        self.bot_agent.reset()
        contexts = contexts.split('\n')
        for context in contexts:
            self.bot_agent.observe({'id': 'localHuman',
                                    'episode_done': False,
                                    'label_candidates': None,
                                    'text': context})
        responses =  self.bot_agent.batch_act([self.bot_agent.observation])[0]
        
        return responses