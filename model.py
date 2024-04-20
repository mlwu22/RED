import os
import torch
import torch.nn as nn
import re


target_dict = {}
total_parameter = 0

class ClassifierHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout()
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ActivationLayer(nn.Module):
    def __init__(self, hidden_size, update_layer, layer_type="all", op_position="ffn", is_llama=False):
        super().__init__()
        self.update_layer = update_layer
        self.layer_type = layer_type
        self.op_position = op_position
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

        
        self.weight = torch.rand(1)
        self.delta_vector.to(self.weight_type)

    def forward(self, x, input_tensor=None):
        if(self.op_position == "res" or self.op_position =="res_with_attn" or self.op_position =="res_with_res"):
            hidden_states = self.update_layer(x, input_tensor)
        else:
            hidden_states = self.update_layer(x)

        if(self.layer_type=="all"):
            hidden_states = hidden_states * self.delta_vector["activation_scaling"]
            hidden_states = hidden_states + self.delta_vector["activation_bias"]
        elif(self.layer_type=="scaling"):
            hidden_states = hidden_states * self.delta_vector["activation_scaling"]
        elif(self.layer_type=="bias"):
            hidden_states = hidden_states + self.delta_vector["activation_bias"]
        elif(self.layer_type=="ln"):
            hidden_states = hidden_states * self.delta_vector["activation_scaling"]
            hidden_states = hidden_states + self.delta_vector["activation_bias"]
            hidden_states = self.delta_vector["activation_ln"](hidden_states)
        if(self.op_position =="res_with_res"):
            hidden_states = hidden_states + x 
        return hidden_states

class ActivationModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.model_type = "t5-base"
        self.frozen_model()
        key_list = [key for key, _ in base_model.named_modules()]
        for key in key_list:
            if(self.check_update(key)):
                self.replace_layer(key)    

    def check_update(self, key):
        check_list = ["wo"]
        for name in check_list:
            if(name in key):
                return True
        return False

    def replace_layer(self, key):
        replaced_module = self.base_model.get_submodule(key)
        parent_key = ".".join(key.split(".")[:-1])
        parent_module = self.base_model.get_submodule(parent_key)
        replaced_name_last = key.split(".")[-1]
        new_module = ActivationLayer(
                        hidden_size = self.base_model.config.d_model,
                        update_layer = replaced_module)
        setattr(parent_module, replaced_name_last, new_module)


    def print_trainable_parameters(self):
        total_parameters = 0
        trainable_parameters = 0 
        for name, param in self.base_model.named_parameters():
            total_parameters+=param.numel()
            if(param.requires_grad):
                trainable_parameters+=param.numel()
        
        base_model_total_parameters = total_parameters - trainable_parameters
        return trainable_parameters/base_model_total_parameters, trainable_parameters


    def frozen_model(self):
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.base_model(input_ids, attention_mask, labels = labels)
    
    def generate(self, **args):
        return self.base_model.generate(**args)
    
    def get_save_dict(self):
        state_dict = self.base_model.state_dict()
        save_dict = {k: state_dict[k] for k in state_dict if "activation_" in k}
        return save_dict

    def save_model(self, save_path):
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_path)

    def load_model(self, save_path):
        save_dict = torch.load(save_path)
        save_key_list = save_dict.keys()
        key_list = [key for key, _ in self.base_model.state_dict().items()]
        for key in key_list:
            if key in save_key_list:
                new_module = save_dict[key]
                new_module.requires_grad = True
                parent_key = ".".join(key.split(".")[:-1])
                replaced_name_last = key.split(".")[-1]
                self.base_model.get_submodule(parent_key)[replaced_name_last] = new_module

class ActivationModelRoberta(nn.Module):
    def __init__(self, base_model, num_labels, layer_type="all", op_position="ffn", exclude_layers=[]):
        super().__init__()
        self.base_model = base_model
        self.model_type = "roberta-base"
        self.layer_type = layer_type
        self.op_position = op_position
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
        self.classifier = ClassifierHead(self.base_model.config.hidden_size, num_labels) 
        self.print_trainable_parameters()


    def check_update(self, key):
        if(self.op_position=="ffn"):
            return self.match_substring(key)
        elif(self.op_position=="ffn_attn"):
            return self.match_substring_ffn_attn(key)
        elif(self.op_position=="attn"):
            return self.match_substring_attn(key)
        elif(self.op_position=="res"):
            return self.match_substring_res(key)
        elif(self.op_position=="res_with_attn"):
            return self.match_substring_res_with_attn(key)


    def replace_layer(self, key):
        replaced_module = self.base_model.get_submodule(key)
        parent_key = ".".join(key.split(".")[:-1])
        parent_module = self.base_model.get_submodule(parent_key)
        replaced_name_last = key.split(".")[-1]
        new_module = ActivationLayer(
                        hidden_size = self.base_model.config.hidden_size,
                        update_layer = replaced_module,
                        layer_type = self.layer_type,
                        op_position = self.op_position)
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
    

    def forward(self, input_ids, attention_mask):
        hidden = self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(hidden)
        return  {"logits":logits}
    
    
    def get_save_dict(self):
        state_dict = self.base_model.state_dict()
        save_dict = {k: state_dict[k] for k in state_dict if "activation_" in k}
        return save_dict

    def save_model(self, save_path):
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_path)
        save_parent = "/".join(save_path.split("/")[:-1])
        save_classifier = os.path.join(save_parent,"classifier.pth")
        classifier_dict = self.classifier.state_dict()
        torch.save(classifier_dict, save_classifier)

    def load_model(self, save_path):
        save_dict = torch.load(save_path)
        save_key_list = save_dict.keys()
        key_list = [key for key, _ in self.base_model.state_dict().items()]
        for key in key_list:
            if key in save_key_list:
                new_module = save_dict[key]
                new_module.requires_grad = True
                parent_key = ".".join(key.split(".")[:-1])
                replaced_name_last = key.split(".")[-1]
                if("activation_ln" in key):
                    if("weight" in replaced_name_last):
                        self.base_model.get_submodule(parent_key).weight.data = new_module
                    elif("bias" in replaced_name_last):
                        self.base_model.get_submodule(parent_key).bias.data = new_module
                else:
                    self.base_model.get_submodule(parent_key)[replaced_name_last] = new_module

        save_parent = "/".join(save_path.split("/")[:-1])
        save_classifier = os.path.join(save_parent,"classifier.pth")
        classifier_dict = torch.load(save_classifier)
        key_list = [key for key, _ in self.classifier.state_dict().items()]
        classifier_state_dict = self.classifier.state_dict()
        for key in key_list:
            classifier_state_dict[key] = classifier_dict[key]
        self.classifier.load_state_dict(classifier_state_dict)

    def load_model_without_head(self, save_path):
        save_dict = torch.load(save_path)
        save_key_list = save_dict.keys()
        key_list = [key for key, _ in self.base_model.state_dict().items()]
        for key in key_list:
            if key in save_key_list:
                new_module = save_dict[key]
                new_module.requires_grad = True
                parent_key = ".".join(key.split(".")[:-1])
                replaced_name_last = key.split(".")[-1]
                if("activation_ln" in key):
                    if("weight" in replaced_name_last):
                        self.base_model.get_submodule(parent_key).weight.data = new_module
                    elif("bias" in replaced_name_last):
                        self.base_model.get_submodule(parent_key).bias.data = new_module
                else:
                    self.base_model.get_submodule(parent_key)[replaced_name_last] = new_module


    def match_substring(self, input_string):
        pattern = r'\d+\.output.dense'
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False
        
    def match_substring_ffn_attn(self, input_string):
        pattern = r'.output.dense'
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False
        
    def match_substring_attn(self, input_string):
        pattern = r'[^\d].output.dense'
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False
    
    def match_substring_res(self, input_string):
        pattern = r'\d+\.output$'
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False
    
    def match_substring_res_with_attn(self, input_string):
        pattern = r'.output$'
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False

        
class ActivationModelGPT2(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.model_type = "gpt2-base"
        self.frozen_model()
        key_list = [key for key, _ in base_model.named_modules()]
        for key in key_list:
            if(self.check_update(key)):
                self.replace_layer(key)   
        self.print_trainable_parameters()


    def check_update(self, key):
        return self.match_substring(key)

    def generate(self, **args):
        return self.base_model.generate(**args)

    def replace_layer(self, key):
        replaced_module = self.base_model.get_submodule(key)
        parent_key = ".".join(key.split(".")[:-1])
        parent_module = self.base_model.get_submodule(parent_key)
        replaced_name_last = key.split(".")[-1]
        new_module = ActivationLayer(
                        hidden_size = self.base_model.config.hidden_size,
                        update_layer = replaced_module)
        setattr(parent_module, replaced_name_last, new_module)


    def print_trainable_parameters(self):
        total_parameters = 0
        trainable_parameters = 0 
        for name, param in self.base_model.named_parameters():
            total_parameters+=param.numel()
            if(param.requires_grad):
                trainable_parameters+=param.numel()
        
        base_model_total_parameters = total_parameters - trainable_parameters
        return trainable_parameters/base_model_total_parameters, trainable_parameters


    def frozen_model(self):
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = output.logits
        return  {"logits":logits}
    
    
    def get_save_dict(self):
        state_dict = self.base_model.state_dict()
        save_dict = {k: state_dict[k] for k in state_dict if "activation_" in k}
        return save_dict

    def save_model(self, save_path):
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_path)   

    def load_model(self, save_path):
        save_dict = torch.load(save_path)
        save_key_list = save_dict.keys()
        key_list = [key for key, _ in self.base_model.state_dict().items()]
        for key in key_list:
            if key in save_key_list:
                new_module = save_dict[key]
                new_module.requires_grad = True
                parent_key = ".".join(key.split(".")[:-1])
                replaced_name_last = key.split(".")[-1]
                self.base_model.get_submodule(parent_key)[replaced_name_last] = new_module

    def match_substring(self, input_string):
        pattern = r'mlp.c_proj'
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False
        

class ActivationLLama(nn.Module):
    _no_split_modules = ["LlamaDecoderLayer"]
    def __init__(self, base_model, op_position="ffn", layer_type="all", exclude_layers=[]):
        super().__init__()
        self.base_model = base_model
        self.model_type = "llama-7b"
        self.layer_type = layer_type
        self.op_position = op_position
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

        print(self.print_trainable_parameters())



    def check_update(self, key):
        if(self.op_position=="ffn"):
            return self.match_substring(key)

    def generate(self, **args):
        return self.base_model.generate(**args)


    def replace_layer(self, key):
        replaced_module = self.base_model.get_submodule(key)
        parent_key = ".".join(key.split(".")[:-1])
        parent_module = self.base_model.get_submodule(parent_key)
        replaced_name_last = key.split(".")[-1]
        new_module = ActivationLayer(
                        hidden_size = self.base_model.config.hidden_size,
                        update_layer = replaced_module,
                        layer_type = self.layer_type,
                        op_position = self.op_position,
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


class ActivationModelRobertaMore(nn.Module):
    def __init__(self, base_model, num_labels, op_position="ffn", layer_type="all", exclude_layers=[]):
        super().__init__()
        self.base_model = base_model
        self.model_type = "roberta-base"
        self.layer_type = layer_type
        self.op_position = op_position
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
        self.classifier = ClassifierHead(self.base_model.config.hidden_size, num_labels) 
        self.print_trainable_parameters()

    def check_update(self, key):
        if(self.op_position=="ffn"):
            return self.match_substring(key)
        elif(self.op_position=="ffn_attn"):
            return self.match_substring_ffn_attn(key)
        elif(self.op_position=="attn"):
            return self.match_substring_attn(key)
        elif(self.op_position=="res"):
            return self.match_substring_res(key)
        elif(self.op_position=="res_with_attn"):
            return self.match_substring_res_with_attn(key)
        elif(self.op_position=="more"):
            return self.match_more(key)

    def replace_layer(self, key):
        replaced_module = self.base_model.get_submodule(key)
        parent_key = ".".join(key.split(".")[:-1])
        parent_module = self.base_model.get_submodule(parent_key)
        replaced_name_last = key.split(".")[-1]
        new_module = ActivationLayer(
                        hidden_size = self.base_model.config.hidden_size,
                        update_layer = replaced_module,
                        layer_type = self.layer_type,
                        op_position = self.op_position)
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
    
    def forward(self, input_ids, attention_mask):
        hidden = self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(hidden)
        return  {"logits":logits}
        
    def get_save_dict(self):
        state_dict = self.base_model.state_dict()
        save_dict = {k: state_dict[k] for k in state_dict if "activation_" in k}
        return save_dict

    def save_model(self, save_path):
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_path)
        save_parent = "/".join(save_path.split("/")[:-1])
        save_classifier = os.path.join(save_parent,"classifier.pth")
        classifier_dict = self.classifier.state_dict()
        torch.save(classifier_dict, save_classifier)

    def load_model(self, save_path):
        save_dict = torch.load(save_path)
        save_key_list = save_dict.keys()
        key_list = [key for key, _ in self.base_model.state_dict().items()]
        for key in key_list:
            if key in save_key_list:
                new_module = save_dict[key]
                new_module.requires_grad = True
                parent_key = ".".join(key.split(".")[:-1])
                replaced_name_last = key.split(".")[-1]
                if("activation_ln" in key):
                    if("weight" in replaced_name_last):
                        self.base_model.get_submodule(parent_key).weight.data = new_module
                    elif("bias" in replaced_name_last):
                        self.base_model.get_submodule(parent_key).bias.data = new_module
                else:
                    self.base_model.get_submodule(parent_key)[replaced_name_last] = new_module

        save_parent = "/".join(save_path.split("/")[:-1])
        save_classifier = os.path.join(save_parent,"classifier.pth")
        classifier_dict = torch.load(save_classifier)
        key_list = [key for key, _ in self.classifier.state_dict().items()]
        classifier_state_dict = self.classifier.state_dict()
        for key in key_list:
            classifier_state_dict[key] = classifier_dict[key]
        self.classifier.load_state_dict(classifier_state_dict)

    def load_model_without_head(self, save_path):
        save_dict = torch.load(save_path)
        save_key_list = save_dict.keys()
        key_list = [key for key, _ in self.base_model.state_dict().items()]
        for key in key_list:
            if key in save_key_list:
                new_module = save_dict[key]
                new_module.requires_grad = True
                parent_key = ".".join(key.split(".")[:-1])
                replaced_name_last = key.split(".")[-1]
                if("activation_ln" in key):
                    if("weight" in replaced_name_last):
                        self.base_model.get_submodule(parent_key).weight.data = new_module
                    elif("bias" in replaced_name_last):
                        self.base_model.get_submodule(parent_key).bias.data = new_module
                else:
                    self.base_model.get_submodule(parent_key)[replaced_name_last] = new_module


    def match_substring(self, input_string):
        pattern = r'\d+\.output.dense'
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False
        
    def match_substring_ffn_attn(self, input_string):
        pattern = r'.output.dense'
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False
        
    def match_substring_attn(self, input_string):
        pattern = r'[^\d].output.dense'
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False
    
    def match_substring_res(self, input_string):
        pattern = r'\d+\.output$'
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False
    
    def match_substring_res_with_attn(self, input_string):
        pattern = r'.output$'
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False
        
    def match_more(self, input_string):
        pattern = r'.output.dense|key|query|value'
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False
        
class ActivationModelGPT2More(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.model_type = "gpt2-base"
        self.frozen_model()
        key_list = [key for key, _ in base_model.named_modules()]
        for key in key_list:
            if(self.check_update(key)):
                self.replace_layer(key)   
        self.print_trainable_parameters()

    def check_update(self, key):
        return self.match_substring(key)

    def generate(self, **args):
        return self.base_model.generate(**args)


    def replace_layer(self, key):
        replaced_module = self.base_model.get_submodule(key)
        parent_key = ".".join(key.split(".")[:-1])
        parent_module = self.base_model.get_submodule(parent_key)
        replaced_name_last = key.split(".")[-1]
        if("c_attn" in key):
            new_module = ActivationLayer(
                        hidden_size = self.base_model.config.hidden_size*3,
                        update_layer = replaced_module)
        else: 
            new_module = ActivationLayer(
                        hidden_size = self.base_model.config.hidden_size,
                        update_layer = replaced_module)
        setattr(parent_module, replaced_name_last, new_module)


    def print_trainable_parameters(self):
        total_parameters = 0
        trainable_parameters = 0 
        for name, param in self.base_model.named_parameters():
            total_parameters+=param.numel()
            if(param.requires_grad):
                trainable_parameters+=param.numel()
        
        base_model_total_parameters = total_parameters - trainable_parameters
        return trainable_parameters/base_model_total_parameters, trainable_parameters


    def frozen_model(self):
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = output.logits
        return  {"logits":logits}
    
    
    def get_save_dict(self):
        state_dict = self.base_model.state_dict()
        save_dict = {k: state_dict[k] for k in state_dict if "activation_" in k}
        return save_dict

    def save_model(self, save_path):
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_path)   

    def load_model(self, save_path):
        save_dict = torch.load(save_path)
        save_key_list = save_dict.keys()
        key_list = [key for key, _ in self.base_model.state_dict().items()]
        for key in key_list:
            if key in save_key_list:
                new_module = save_dict[key]
                new_module.requires_grad = True
                parent_key = ".".join(key.split(".")[:-1])
                replaced_name_last = key.split(".")[-1]
                self.base_model.get_submodule(parent_key)[replaced_name_last] = new_module

    def match_substring(self, input_string):
        pattern = r'c_attn|c_proj'
        match = re.search(pattern, input_string)
        if match:
            return True
        else:
            return False