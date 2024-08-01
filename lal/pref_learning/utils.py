import re
import torch

rs_feature_aspects = ['gt_reward', 'speed', 'height', 'distance_to_bottle', 'distance_to_cube']
mw_feature_aspects = ['height', 'velocity', 'distance']

def remove_special_characters(input_string):
    # Define the pattern to match special characters (non-alphanumeric and non-whitespace)
    pattern = r'[^a-zA-Z0-9\s]'
    # Use re.sub() to replace the special characters with an empty string
    cleaned_string = re.sub(pattern, '', input_string)
    cleaned_string = cleaned_string.strip()
    cleaned_string += "."
    return cleaned_string

def get_lang_embed(nlcomp, model, device, tokenizer, preprocessed=False, lang_model=None):
    if preprocessed:
        assert lang_model is not None
        inputs = tokenizer(nlcomp, return_tensors="pt")
        lang_outputs = lang_model(**inputs)
        embedding = lang_outputs.last_hidden_state

        # Average across the sequence to get a sentence-level embedding
        embedding = torch.mean(embedding, dim=1, keepdim=False)
        lang_embed = model.lang_encoder(embedding.to(device)).squeeze(0).detach().cpu()

    else:
        # First tokenize the NL comparison and get the embedding
        inputs = tokenizer(nlcomp, return_tensors="pt")
        # move inputs to the device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        hidden_states = model.lang_encoder(**inputs).last_hidden_state
        lang_embed = torch.mean(hidden_states, dim=1, keepdim=False).squeeze(0).detach().cpu()

    return lang_embed