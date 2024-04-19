import torch.nn as nn
import torch
from galvatron.core.pipeline import PipeSequential
from galvatron.core import mixed_precision_dtype, ModelInfo

def get_extended_attention_mask(attention_mask):
    # the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

# Use BertEmbeddings defined in bert_model
class BertEmbeddings_(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.embeddings = bert_model.bert.embeddings
    def forward(self, input_ids, token_type_ids, attention_mask):
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        return embedding_output, attention_mask

# Use BertEncoder defined in bert_model
class BertEncoder_(nn.Module):
    def __init__(self, bert_model, layer_idx_start, layer_idx_end):
        super().__init__()
        self.layer = bert_model.bert.encoder.layer[layer_idx_start:layer_idx_end]
        self.get_extended_attention_mask = get_extended_attention_mask
    def forward(self, hidden_states, attention_mask):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
            )
            hidden_states = layer_outputs[0]
        return hidden_states, attention_mask

# Use BertPooler defined in bert_model
class BertPooler_(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.pooler = bert_model.bert.pooler
    def forward(self, hidden_states, attention_mask):
        return hidden_states, self.pooler(hidden_states)

# Use BertPreTrainingHeads defined in bert_model
class BertPreTrainingHeads_(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.cls = bert_model.cls
    def forward(self, sequence_output, pooled_output):
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        return prediction_scores, seq_relationship_score

def construct_sequential_model(model, config):
    model_ = PipeSequential()
    model_.add_module('embeddings', BertEmbeddings_(model))
    for i in range(config.num_hidden_layers):
        enc = BertEncoder_(model, i, i + 1)
        setattr(enc, 'get_extended_attention_mask', get_extended_attention_mask)
        model_.add_module('encoder_%d'%i, enc)
    model_.add_module('pooler', BertPooler_(model))
    model_.add_module('cls', BertPreTrainingHeads_(model))
    return model_

class BertModelInfo(ModelInfo):
    def __init__(self, config, args):
        super(BertModelInfo, self).__init__()
        layernum_list = [config.num_hidden_layers]
        seq_len, hidden_size = config.max_position_embeddings, config.hidden_size
        mixed_precision = mixed_precision_dtype(args.mixed_precision)
        layer_shapes_list = [[[-1,seq_len,hidden_size], [-1,seq_len]]]
        layer_dtypes_list = [[mixed_precision, torch.long]]
        module_types = ['embed'] + ['bert_enc']*config.num_hidden_layers + ['pooler', 'cls']
        self.set_layernums(layernum_list)
        self.set_shapes(layer_shapes_list)
        self.set_dtypes(layer_dtypes_list)
        self.set_module_types(module_types)