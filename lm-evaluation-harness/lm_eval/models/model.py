import sys


from transformers import LlamaConfig,PreTrainedModel,LlamaForCausalLM
import torch
import torch.nn as nn
import re

class ActivationLayer(nn.Module):
    def __init__(self, hidden_size, update_layer,layer_type="all", add_type="ffn", is_llama=False):
        super().__init__()
        self.update_layer = update_layer
        self.layer_type = layer_type
        self.add_type = add_type
        if(is_llama):
            self.weight_type = torch.bfloat16
        else:
            self.weight_type = torch.float32
        if(self.layer_type=="all"):
            self.delta_vector = nn.ParameterDict({
                "activation_scaling": nn.Parameter(torch.ones(1, hidden_size)),
                "activation_bias":nn.Parameter(torch.zeros(1, hidden_size)),
            })
        elif(self.layer_type=="scaling"):
            self.delta_vector = nn.ParameterDict({
                "activation_scaling": nn.Parameter(torch.ones(1, hidden_size))
            })
        elif(self.layer_type=="bias"):
            self.delta_vector = nn.ParameterDict({
                "activation_bias":nn.Parameter(torch.zeros(1, hidden_size))
            })
        elif(self.layer_type=="ln"):
            self.delta_vector = nn.ParameterDict({
                "activation_ln": nn.LayerNorm(hidden_size),
                "activation_scaling": nn.Parameter(torch.ones(1, hidden_size)),
                "activation_bias":nn.Parameter(torch.zeros(1, hidden_size)),
            })
        # 为了跑通验证的代码
        # self.weight = torch.rand(1)
        self.delta_vector.to(self.weight_type)


    def forward(self, x, input_tensor=None):
        if(self.add_type == "res" or self.add_type =="res_with_attn" or self.add_type =="res_with_res"):
            hidden_states = self.update_layer(x, input_tensor)
        else:
            hidden_states = self.update_layer(x)

        if(self.layer_type=="all"):
            hidden_states = hidden_states * self.delta_vector["activation_scaling"].to(hidden_states.device)
            hidden_states = hidden_states + self.delta_vector["activation_bias"].to(hidden_states.device)
        elif(self.layer_type=="scaling"):
            hidden_states = hidden_states * self.delta_vector["activation_scaling"]
        elif(self.layer_type=="bias"):
            hidden_states = hidden_states + self.delta_vector["activation_bias"]
        elif(self.layer_type=="ln"):
            # input_tensor = hidden_states
            # hidden_states = self.delta_vector["activation_ln"](hidden_states)           # 先过 LN 再加残差
            hidden_states = hidden_states * self.delta_vector["activation_scaling"]
            hidden_states = hidden_states + self.delta_vector["activation_bias"]
            hidden_states = self.delta_vector["activation_ln"](hidden_states)
            # hidden_states = hidden_states + input_tensor
        if(self.add_type =="res_with_res"):
            hidden_states = hidden_states + x # 相当于加上残差
            pass
        return hidden_states

class ActivationLLama(nn.Module):
    _no_split_modules = ["LlamaDecoderLayer"]
    def __init__(self, base_model, add_type="ffn", layer_type="all", exclude_layers=[]):
        super().__init__()
        self.base_model = base_model
        self.model_type = "llama-7b"
        self.layer_type = layer_type
        self.add_type = add_type
        self.exclude_layers = exclude_layers
        if(exclude_layers):
            pattern_str = '|'.join(map(str, exclude_layers))
            pattern = re.compile(r'\b(?:' + pattern_str + r')\b')
        self.frozen_model()
        key_list = [key for key, _ in base_model.named_modules()]
        for key in key_list:
            if(exclude_layers):
                match = pattern.search(key)
                if(match):
                    continue
            if(self.check_update(key)):
                self.replace_layer(key)   
                
        self.device = self.base_model.device
        self.config = self.base_model.config
        print(self.print_trainable_parameters())


    # 判断某个 Module 是否需要更新
    def check_update(self, key):
        if(self.add_type=="ffn"):
            return self.match_substring(key)

    def generate(self, **args):
        return self.base_model.generate(**args)

    def tie_weights(self):
        return self.base_model.tie_weights()
        

    # 将 key 的 module 替换成对应的 layer
    def replace_layer(self, key):
        replaced_module = self.base_model.get_submodule(key)
        parent_key = ".".join(key.split(".")[:-1])
        parent_module = self.base_model.get_submodule(parent_key)
        replaced_name_last = key.split(".")[-1]
        new_module = ActivationLayer(
                        hidden_size = self.base_model.config.hidden_size,
                        update_layer = replaced_module,
                        layer_type = self.layer_type,
                        add_type = self.add_type,
                        is_llama=True)
        setattr(parent_module, replaced_name_last, new_module)

    def print_trainable_parameters(self):
        total_parameters = 0
        trainable_parameters = 0 
        for name, param in self.base_model.named_parameters():
            total_parameters+=param.numel()
            if(param.requires_grad):
                trainable_parameters+=param.numel()
    
        return {
            "total_para:": total_parameters,
            "trainable_para: ":trainable_parameters,
            "trainable%:" : f"{100 * trainable_parameters / total_parameters:.4f}"
                }


    def frozen_model(self):
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False
    

    def match_substring(self, input_string):
        pattern = r'down_proj'
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False

    def forward(self, input_ids, attention_mask=None, labels=None):
        # input_ids = input_ids.to(self.base_model.model.device)
        # attention_mask = attention_mask.to(self.base_model.model.device)
        # labels = labels.to(self.base_model.model.device)
        # print(input_ids.device)
        output = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return  output
    
    def load_model(self, save_path):
        save_dict = torch.load(save_path)
        save_key_list = save_dict.keys()
        key_list = [key for key, _ in self.base_model.state_dict().items()]
        for key in key_list:
            if key in save_key_list:
                new_module = save_dict[key]
                new_module.requires_grad = True
                parent_key = ".".join(key.split(".")[:-1])
                # parent_module = self.base_model.get_submodule(parent_key)
                replaced_name_last = key.split(".")[-1]
                if("activation_ln" in key):
                    if("weight" in replaced_name_last):
                        self.base_model.get_submodule(parent_key).weight.data = new_module
                    elif("bias" in replaced_name_last):
                        self.base_model.get_submodule(parent_key).bias.data = new_module
                else:
                    self.base_model.get_submodule(parent_key)[replaced_name_last] = new_module

    def get_save_dict(self):
        state_dict = self.base_model.state_dict()
        save_dict = {k: state_dict[k] for k in state_dict if "activation_" in k}
        return save_dict

    def save_model(self, save_path):
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_path)


