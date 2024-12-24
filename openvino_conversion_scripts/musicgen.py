from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy
import numpy as np
#set seed...
import random
import os
import torch
my_seed=1
random.seed(my_seed)
torch.manual_seed(my_seed)
os.environ["PYTHONHASHSEED"] = str(my_seed)

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

audio_to_continue = torch.from_numpy(np.load("audio_values.npy")).squeeze(0).squeeze(0)
print("audio_to_continue shape = ", audio_to_continue.shape)
inputs = processor(
    audio=audio_to_continue,
    text=["80s pop track with bassy drums and synth"],
    #padding=True,
    return_tensors="pt",
)

print("type(model.decoder.model.decoder) = ", type(model.decoder.model.decoder))

#todo: this should be in separate py or function
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
        
    def forward(self, encoder_hidden_states, attention_mask):
    
        print("forward!")
        encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
        
        encoder_hidden_states = encoder_hidden_states * attention_mask[..., None]
        
        bsz = 2
        initial_kv = ()
        for i in range(0, len(self.k_proj_list)):
            key_states = self._shape(self.k_proj_list[i](encoder_hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj_list[i](encoder_hidden_states), -1, bsz)
            initial_kv += (key_states, value_states, )
            
        print("returning from InitialCrossAttnProducer forward")

        return initial_kv
        
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

def generate_initial_cross_attn_kv_producer(model):
    print("generating initial_cross_attn_kv_producer")
    

    initial_cross_attn_producer = InitialCrossAttnProducer(model.config.decoder.hidden_size, 
                                                           model.config.decoder.num_attention_heads,
                                                           model.enc_to_dec_proj)

    musicgen_decoder = model.decoder.model.decoder
    for layer in musicgen_decoder.layers:
        print("type(layer)", type(layer))
        print("type(self.initial_cross_attn_producer.k_proj_list)", type(initial_cross_attn_producer.k_proj_list))
        initial_cross_attn_producer.k_proj_list.append(layer.encoder_attn.k_proj)
        initial_cross_attn_producer.v_proj_list.append(layer.encoder_attn.v_proj)
        
    import numpy as np    
    encoder_hidden_states =  torch.from_numpy(np.load('encoder_hidden_states.npy'))
    attention_mask =  torch.from_numpy(np.load('attention_mask.npy'))
    
    encoder_hidden_states_padded = torch.zeros(2, 64, 768)
    encoder_hidden_states_padded[:, 0:encoder_hidden_states.shape[1], :] = encoder_hidden_states
    
    attention_mask_padded = torch.zeros(2,64, dtype=torch.int64)
    attention_mask_padded[:, 0:attention_mask.shape[1]] = attention_mask
    
    encoder_hidden_states = encoder_hidden_states_padded
    attention_mask = attention_mask_padded
    
    initial_cross_attn_producer.eval()
    output_names=[]            
    for l in range(0, 24):
        for i in range(0, 2):
            output_names.append("new_key_value_" + str(l) + "_" + str(i))
            
    print("done appending k/v")            
    torch.onnx.export(
                initial_cross_attn_producer,
                (encoder_hidden_states, attention_mask),
                "initial_cross_attn_kv_producer.onnx",
                input_names=["encoder_hidden_states", "attention_mask"],
                output_names=output_names
            )
    print("done generating initial_cross_attn_kv_producer.onnx")
    
def test(model):
    
    initial_cross_attn_producer = InitialCrossAttnProducer(model.config.decoder.hidden_size, 
                                                           model.config.decoder.num_attention_heads,
                                                           model.enc_to_dec_proj)
    musicgen_decoder = model.decoder.model.decoder
    for layer in musicgen_decoder.layers:
        initial_cross_attn_producer.k_proj_list.append(layer.encoder_attn.k_proj)
        initial_cross_attn_producer.v_proj_list.append(layer.encoder_attn.v_proj)
    
    import numpy as np    
    encoder_hidden_states =  torch.from_numpy(np.load('encoder_hidden_states.npy'))
    attention_mask =  torch.from_numpy(np.load('attention_mask.npy'))
    
    encoder_hidden_states_padded = torch.zeros(2, 64, 768)
    encoder_hidden_states_padded[:, 0:encoder_hidden_states.shape[1], :] = encoder_hidden_states
    
    attention_mask_padded = torch.zeros(2,64, dtype=torch.int64)
    attention_mask_padded[:, 0:attention_mask.shape[1]] = attention_mask
    
    encoder_hidden_states = encoder_hidden_states_padded
    attention_mask = attention_mask_padded
    
    print("encoder_hidden_states shape = ", encoder_hidden_states.shape)
    print("attention_mask shape = ", attention_mask.shape)
    
    res = initial_cross_attn_producer(encoder_hidden_states, attention_mask)
    
    first_key = res[0]
    print("first_key.shape = ", first_key.shape)
    print("first 10 of first key = ", first_key[0,0,0,0:10])
                                                           
    
import torch.nn as nn
class MusicGenWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, decoder_input_ids, attention_mask, encoder_outputs, decoder_attention_mask=None, past_key_length_tens=None, past_key_values=None ):
        
        return_only_new_kv = False
        if past_key_values is not None and past_key_length_tens is not None:
            return_only_new_kv = True
        
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

def generate_decode_kvcache(model):
   
    with torch.no_grad():
        model.config.torchscript = False
        model.config.return_dict = True
        model.eval()
        
        past_key_num_tokens = 64
        decoder_input_ids = torch.ones(8, 1, dtype=torch.int64)
        attention_mask = torch.zeros(2, 64, dtype=torch.int64)
        encoder_outputs = ( torch.randn(2, 64, 768), )
        past_key_values = []
        for i in range(24):
            past_key_values.append( (torch.randn(2, 16, past_key_num_tokens, 64),torch.randn(2, 16, past_key_num_tokens, 64),torch.randn(2, 16, 64, 64),torch.randn(2, 16, 64, 64)) )
        
        decoder_attention_mask = torch.zeros(2,past_key_num_tokens+1, dtype=torch.int64)
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
        openvino.runtime.save_model(ov_model, "musicgen_decoder.xml")
        
        from nncf import compress_weights, CompressWeightsMode
        #ov_model_compressed = compress_weights(ov_model)
        #openvino.save_model(ov_model_compressed, "musicgen_decoder_int8.xml")
        
        ov_model_compressed = compress_weights(ov_model, mode=CompressWeightsMode.INT4_ASYM)
        openvino.save_model(ov_model_compressed, "musicgen_decoder_int4.xml")
        
        print("done saving kv cache model")
        
def generate_decode_nonkvcache(model):
   
    with torch.no_grad():
        model.config.torchscript = False
        model.config.return_dict = True
        model.eval()
        
        
        decoder_input_ids = torch.ones(8, 254, dtype=torch.int64)
        attention_mask = torch.zeros(2, 64, dtype=torch.int64)
        encoder_outputs = ( torch.randn(2, 64, 768), )
        past_key_values = []
        for i in range(24):
            past_key_values.append( (torch.randn(2, 16, 64, 64),torch.randn(2, 16, 64, 64)) )

        
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
        openvino.runtime.save_model(ov_model, "musicgen_decoder_nonkv.xml")
        
        #from nncf import compress_weights, CompressWeightsMode
        #ov_model_compressed = compress_weights(ov_model)
        #openvino.save_model(ov_model_compressed, "musicgen_decoder_int8.xml")
        
        #ov_model_compressed = compress_weights(ov_model, mode=CompressWeightsMode.INT4_ASYM)
        #openvino.save_model(ov_model_compressed, "musicgen_decoder_int4.xml")
        
        print("done saving nonkv cache model!")

        
    
def manual_decode(model):    
    import numpy as np    
    encoder_hidden_states =  torch.from_numpy(np.load('encoder_hidden_states.npy'))
    attention_mask =  torch.from_numpy(np.load('attention_mask.npy'))
    decoder_input_ids = torch.from_numpy(np.load('decoder_input_ids.npy'))
    
    encoder_hidden_states_padded = torch.zeros(2, 64, 768)
    encoder_hidden_states_padded[:, 0:encoder_hidden_states.shape[1], :] = encoder_hidden_states
    
    attention_mask_padded = torch.zeros(2,64, dtype=torch.int64)
    attention_mask_padded[:, 0:attention_mask.shape[1]] = attention_mask
    
    encoder_hidden_states = encoder_hidden_states_padded
    attention_mask = attention_mask_padded
    
    past_key_num_tokens=1004
    decoder_attention_mask = torch.zeros(2,past_key_num_tokens+1, dtype=torch.int64)
    decoder_attention_mask[:, -1] = 1
    decoder_attention_mask.numpy().tofile("decoder_attention_mask_works.raw")
    
    past_key_length_tens = torch.zeros(1, dtype=torch.int64)
    past_key_length_tens[0] = 0
    
    past_key_values = []
    for i in range(24):
        #decode past key vals
        this_layer_tuple = ()
        for ii in range(2):
           this_layer_tuple += ( torch.zeros(2, 16, past_key_num_tokens, 64), )
        
        #encode past key vals
        for ii in range(2):
            past_key_val = torch.zeros(2, 16, 64, 64)
            past_key_val_slice = torch.from_numpy(np.load("past_key_value_" + str(i) + "_" + str(ii + 2) + ".npy"))
            past_key_val[:,:,0:past_key_val_slice.shape[2],:] = past_key_val_slice
            this_layer_tuple += ( past_key_val, )
            
        past_key_values.append(this_layer_tuple)
        
    #past_key_values = []
    #for i in range(24):
    #    past_key_values.append( (torch.zeros(2, 16, past_key_num_tokens, 64),torch.zeros(2, 16, past_key_num_tokens, 64),torch.zeros(2, 16, 64, 64),torch.zeros(2, 16, 64, 64)) )
    

    encoder_outputs = (encoder_hidden_states, None, None)
    
    with torch.no_grad():
        model.config.torchscript = False
        model.config.return_dict = True
        model.eval()
        
        ret = model(decoder_input_ids=decoder_input_ids, 
                    attention_mask=attention_mask,
                    encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                    decoder_attention_mask=decoder_attention_mask,
                    past_key_length_tens=past_key_length_tens
                    )
    

generate_initial_cross_attn_kv_producer(model)
generate_decode_kvcache(model)
generate_decode_nonkvcache(model)

print("done saving all IRs")
import sys
sys.exit(0)
if False:
    test(model)
    import sys
    sys.exit()
elif False:
    test_decode(model)
else:
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
    
   
    #np.save("audio_values.npy", audio_values)

    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())