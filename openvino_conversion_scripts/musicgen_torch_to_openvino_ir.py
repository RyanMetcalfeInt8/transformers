from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy
import numpy as np
import random
import os
import torch
import sys
from consolidate_ir_bins import consolidate_ir_bins_in_folder


import torch.nn as nn
class InitialCrossAttnProducer(nn.Module):
    def __init__(self, embed_dim: int,
                       num_heads: int,
                       enc_to_dec_proj
        ):
        super().__init__()
        self.k_proj_list = nn.ModuleList()
        self.v_proj_list = nn.ModuleList()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.enc_to_dec_proj = enc_to_dec_proj

    def forward(self, encoder_hidden_states, encoder_attention_mask):

        encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        encoder_hidden_states = encoder_hidden_states * encoder_attention_mask[..., None]

        bsz = 2
        initial_kv = []
        for i in range(0, len(self.k_proj_list)):
            key_states = self._shape(self.k_proj_list[i](encoder_hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj_list[i](encoder_hidden_states), -1, bsz)
            initial_kv.append((key_states, value_states, ))

        ret_dict = {}

        for layeri in range(len(initial_kv)):
            past_key_tuple = initial_kv[layeri]
            ret_dict["present." + str(layeri) + ".encoder.key"] = past_key_tuple[0]
            ret_dict["present." + str(layeri) + ".encoder.value"] = past_key_tuple[1]

        return ret_dict

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

def generate_initial_cross_attn_kv_producer(model, dst_folder="."):
    print("generating initial_cross_attn_kv_producer")

    with torch.no_grad():
        initial_cross_attn_producer = InitialCrossAttnProducer(model.config.decoder.hidden_size,
                                                               model.config.decoder.num_attention_heads,
                                                               model.enc_to_dec_proj)

        musicgen_decoder = model.decoder.model.decoder
        for layer in musicgen_decoder.layers:
            initial_cross_attn_producer.k_proj_list.append(layer.encoder_attn.k_proj)
            initial_cross_attn_producer.v_proj_list.append(layer.encoder_attn.v_proj)

        #import numpy as np
        #encoder_hidden_states =  torch.from_numpy(np.load('encoder_hidden_states.npy'))
        #print("encoder_hidden_states.shape = ", encoder_hidden_states.shape)
        #attention_mask =  torch.from_numpy(np.load('attention_mask.npy'))

        encoder_hidden_states_padded = torch.zeros(2, 64, 768)
        #encoder_hidden_states_padded[:, 0:encoder_hidden_states.shape[1], :] = encoder_hidden_states

        attention_mask_padded = torch.zeros(2,64, dtype=torch.int64)
        #attention_mask_padded[:, 0:attention_mask.shape[1]] = attention_mask

        encoder_hidden_states = encoder_hidden_states_padded
        attention_mask = attention_mask_padded

        initial_cross_attn_producer.eval()

        dummy_inputs = {}
        dummy_inputs["encoder_hidden_states"] = encoder_hidden_states
        dummy_inputs["encoder_attention_mask"] = attention_mask
        import openvino
        from openvino.tools.ovc import convert_model
        ov_model = convert_model(initial_cross_attn_producer, example_input=dummy_inputs)

        ov_model.inputs[0].get_tensor().set_names({"encoder_hidden_states"})
        ov_model.inputs[1].get_tensor().set_names({"attention_mask"})

        openvino.runtime.save_model(ov_model, os.path.join(dst_folder, "initial_cross_attn_kv_producer.xml"), compress_to_fp16=True)

        print("done generating initial_cross_attn_kv_producer.xml")
        
import torch.nn as nn
class MusicGenWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, decoder_input_ids, attention_mask, encoder_outputs, decoder_attention_mask=None, past_key_length_tens=None, past_key_values=None ):

        return_only_new_kv = False
        if past_key_values is not None and past_key_length_tens is not None:
            return_only_new_kv = True

        print("setting return_only_new_kv to ", return_only_new_kv)
        ret = self.model(decoder_input_ids=decoder_input_ids,
                         attention_mask=attention_mask,
                         encoder_outputs=encoder_outputs,
                         decoder_attention_mask=decoder_attention_mask,
                         past_key_length_tens=past_key_length_tens,
                         past_key_values=past_key_values,
                         return_only_new_kv=return_only_new_kv,
                         return_dict=True)


        ret_dict = {"logits": ret.logits}


        for layeri in range(len(ret.past_key_values)):
            past_key_tuple = ret.past_key_values[layeri]
            ret_dict["present." + str(layeri) + ".decoder.key"] = past_key_tuple[0]
            ret_dict["present." + str(layeri) + ".decoder.value"] = past_key_tuple[1]

            #if past_key_values == None:
            #    ret_dict["present." + str(layeri) + ".encoder.key"] = past_key_tuple[2]
            #    ret_dict["present." + str(layeri) + ".encoder.value"] = past_key_tuple[3]

        return ret_dict

def generate_decode_kvcache(model, dst_folder="."):

    with torch.no_grad():
        model.config.torchscript = False
        model.config.return_dict = True
        model.eval()

        b = 2

        num_codebooks = model.config.decoder.num_codebooks

        past_key_num_tokens = 64
        decoder_input_ids = torch.ones(num_codebooks*2, 1, dtype=torch.int64)
        attention_mask = torch.zeros(b, 64, dtype=torch.int64)
        encoder_outputs = ( torch.randn(b, 64, 768), )
        past_key_values = []

        num_attention_heads = model.config.decoder.num_attention_heads
        num_hidden_layers = model.config.decoder.num_hidden_layers

        #num_hidden_layers = 1
        for i in range(num_hidden_layers):
            past_key_values.append( (torch.randn(b, num_attention_heads, past_key_num_tokens, 64),torch.randn(b, num_attention_heads, past_key_num_tokens, 64),torch.randn(b, num_attention_heads, 64, 64),torch.randn(b, num_attention_heads, 64, 64)) )

        decoder_attention_mask = torch.zeros(b,past_key_num_tokens+1, dtype=torch.int64)
        decoder_attention_mask[:, -1] = 1

        past_key_length_tens = torch.zeros(1, dtype=torch.int64)
        past_key_length_tens[0] = 0

        dummy_inputs = {}
        dummy_inputs["decoder_input_ids"] = decoder_input_ids
        dummy_inputs["attention_mask"] = attention_mask
        dummy_inputs["encoder_outputs"] = encoder_outputs
        dummy_inputs["decoder_attention_mask"] = decoder_attention_mask
        dummy_inputs["past_key_length_tens"] = past_key_length_tens
        dummy_inputs["past_key_values"] = past_key_values

        wrapper = MusicGenWrapper(model)

        import openvino
        from openvino.tools.ovc import convert_model
        ov_model = convert_model(wrapper, example_input=dummy_inputs)

        ov_model.inputs[0].get_tensor().set_names({"decoder_input_ids"})
        ov_model.inputs[1].get_tensor().set_names({"encoder_attention_mask"})
        ov_model.inputs[2].get_tensor().set_names({"encoder_hidden_states"})
        ov_model.inputs[3].get_tensor().set_names({"decoder_attention_mask"})
        ov_model.inputs[4].get_tensor().set_names({"past_key_length_tens"})

        print(type(ov_model.inputs[4].get_tensor()))

        past_key_vali = 0
        layeri = 0
        while past_key_vali < len(ov_model.inputs) - 5:
            base_index = past_key_vali + 5
            ov_model.inputs[base_index + 0].get_tensor().set_names({"past_key_values." + str(layeri) + ".decoder.key"})
            ov_model.inputs[base_index + 1].get_tensor().set_names({"past_key_values." + str(layeri) + ".decoder.value"})
            ov_model.inputs[base_index + 2].get_tensor().set_names({"past_key_values." + str(layeri) + ".encoder.key"})
            ov_model.inputs[base_index + 3].get_tensor().set_names({"past_key_values." + str(layeri) + ".encoder.value"})

            past_key_vali = past_key_vali + 4
            layeri = layeri + 1



        ov_model.validate_nodes_and_infer_types()
        openvino.runtime.save_model(ov_model, os.path.join(dst_folder, "musicgen_decoder.xml"))

        '''
        for param in ov_model.get_parameters():
            print(f"Parameter: {param.friendly_name}, Shape: {param.get_partial_shape()}")

        name_to_shape = dict()
        name_to_shape["encoder_attention_mask"] = attention_mask.shape
        name_to_shape["decoder_input_ids"] = decoder_input_ids.shape
        name_to_shape["encoder_hidden_states"] = encoder_outputs[0].shape
        name_to_shape["decoder_attention_mask"] = decoder_attention_mask.shape
        name_to_shape["past_key_length_tens"] = past_key_length_tens.shape
        for i in range(num_hidden_layers):
            name_to_shape["past_key_values." + str(i) + ".decoder.key"] = past_key_values[0][0].shape
            name_to_shape["past_key_values." + str(i) + ".decoder.value"] = past_key_values[0][1].shape
            name_to_shape["past_key_values." + str(i) + ".encoder.key"] = past_key_values[0][2].shape
            name_to_shape["past_key_values." + str(i) + ".encoder.value"] = past_key_values[0][3].shape

        ov_model.reshape(name_to_shape)
        openvino.runtime.save_model(ov_model, os.path.join(dst_folder, "musicgen_decoder_reshaped.xml"), compress_to_fp16=True)
        '''

        core = openvino.Core()
        ov_model = core.read_model(os.path.join(dst_folder, "musicgen_decoder.xml"))
        import nncf
        from nncf import compress_weights, CompressWeightsMode
        patterns = ['.*embed_tokens.*', '.*embed_positions.*']
        ignored_scope = nncf.IgnoredScope(patterns=patterns)
        #ignored_scope=None

        ov_model_compressed = compress_weights(ov_model, ignored_scope=ignored_scope)
        openvino.save_model(ov_model_compressed, os.path.join(dst_folder, "musicgen_decoder_int8.xml"))

        #ov_model_compressed = compress_weights(ov_model, ignored_scope=ignored_scope, mode=CompressWeightsMode.INT4_ASYM, ratio=0.75)
        #openvino.save_model(ov_model_compressed, "musicgen_decoder_int4.xml")

        print("done saving kv cache model")

def generate_decode_nonkvcache(model, dst_folder="."):

    with torch.no_grad():
        model.config.torchscript = False
        model.config.return_dict = True
        model.eval()

        num_codebooks = model.config.decoder.num_codebooks

        decoder_input_ids = torch.ones(num_codebooks*2, 254, dtype=torch.int64)
        attention_mask = torch.zeros(2, 64, dtype=torch.int64)
        encoder_outputs = ( torch.randn(2, 64, 768), )
        past_key_values = []


        num_attention_heads = model.config.decoder.num_attention_heads

        num_hidden_layers = model.config.decoder.num_hidden_layers

        #num_hidden_layers = 1
        for i in range(num_hidden_layers):
            past_key_values.append( (torch.randn(2, num_attention_heads, 64, 64),torch.randn(2, num_attention_heads, 64, 64)) )


        dummy_inputs = {}
        dummy_inputs["decoder_input_ids"] = decoder_input_ids
        dummy_inputs["attention_mask"] = attention_mask
        dummy_inputs["encoder_outputs"] = encoder_outputs
        dummy_inputs["past_key_values"] = past_key_values

        wrapper = MusicGenWrapper(model)

        import openvino
        from openvino.tools.ovc import convert_model
        ov_model = convert_model(wrapper, example_input=dummy_inputs)

        ov_model.inputs[0].get_tensor().set_names({"decoder_input_ids"})
        ov_model.inputs[1].get_tensor().set_names({"encoder_attention_mask"})
        ov_model.inputs[2].get_tensor().set_names({"encoder_hidden_states"})


        past_key_vali = 0
        layeri = 0
        while past_key_vali < len(ov_model.inputs) - 3:
            base_index = past_key_vali + 3
            ov_model.inputs[base_index + 0].get_tensor().set_names({"past_key_values." + str(layeri) + ".encoder.key"})
            ov_model.inputs[base_index + 1].get_tensor().set_names({"past_key_values." + str(layeri) + ".encoder.value"})

            past_key_vali = past_key_vali + 2
            layeri = layeri + 1



        ov_model.validate_nodes_and_infer_types()
        openvino.runtime.save_model(ov_model, os.path.join(dst_folder, "musicgen_decoder_nonkv.xml"), compress_to_fp16=True)

        core = openvino.Core()
        ov_model = core.read_model(os.path.join(dst_folder, "musicgen_decoder_nonkv.xml"))

        import nncf
        from nncf import compress_weights, CompressWeightsMode
        patterns = ['.*embed_tokens.*', '.*embed_positions.*']
        ignored_scope = nncf.IgnoredScope(patterns=patterns)

        ov_model_compressed = compress_weights(ov_model, ignored_scope=ignored_scope)
        openvino.save_model(ov_model_compressed, os.path.join(dst_folder, "musicgen_decoder_nonkv_int8.xml"))


        #ov_model_compressed = compress_weights(ov_model, ignored_scope=ignored_scope, mode=CompressWeightsMode.INT4_ASYM)
        #openvino.save_model(ov_model_compressed, "musicgen_decoder_int4.xml")

        print("done saving nonkv cache model!")


if __name__ == "__main__":

    def create_directory(dir_path):
        if os.path.exists(dir_path):
            print(f"Error: Directory '{dir_path}' already exists.", file=sys.stderr)
            sys.exit(1)
        else:
            os.makedirs(dir_path)
    
    model_id="facebook/musicgen-small"
    #model_id="facebook/musicgen-stereo-small"
    #model_id="facebook/musicgen-medium"
    #model_id="facebook/musicgen-stereo-medium"
    
    
    ov_ir_output_folder = model_id.split("/")[-1] + "-openvino"
    create_directory(ov_ir_output_folder)

    model = MusicgenForConditionalGeneration.from_pretrained(model_id)    
    
    generate_initial_cross_attn_kv_producer(model, dst_folder=ov_ir_output_folder)
    
    intermediate_dir=os.path.join(ov_ir_output_folder, "intermediate")
    generate_decode_kvcache(model, dst_folder=intermediate_dir)
    generate_decode_nonkvcache(model, dst_folder=intermediate_dir)
   
    
    #okay, combine some bins
    input_ir_files=[ 
        os.path.join(intermediate_dir, "musicgen_decoder_nonkv.xml"),
        os.path.join(intermediate_dir, "musicgen_decoder.xml")]     
    consolidate_ir_bins_in_folder(input_ir_files=input_ir_files, output_folder=ov_ir_output_folder, combined_bin_name="musicgen_decoder_combined.bin")
    
    input_ir_files=[ 
        os.path.join(intermediate_dir, "musicgen_decoder_nonkv_int8.xml"),
        os.path.join(intermediate_dir, "musicgen_decoder_int8.xml")]     
    consolidate_ir_bins_in_folder(input_ir_files=input_ir_files, output_folder=ov_ir_output_folder, combined_bin_name="musicgen_decoder_int8_combined.bin")