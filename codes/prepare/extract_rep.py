import fire
from glob import glob
import json
import numpy as np
import os
from pathlib import Path
import shutil
import time
from tqdm import tqdm
import torch
curr_dir = os.path.dirname(os.path.realpath(__file__))
import sys
import gc
sys.path.insert(0, os.path.join(curr_dir, ".."))
from model_utils import ModelLoader, FeatExtractor
from utils import read_lst, load_dct, write_to_file, save_dct

def log_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")

def force_cleanup():
    """Aggressive memory cleanup"""
    for _ in range(3):  # Multiple cleanup passes
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def save_batch_array(array_list, save_path, mode='wb'):
    """Save arrays efficiently"""
    if not array_list:
        return
    try:
        array = np.concatenate(array_list, 0)
        with open(save_path, mode) as f:
            np.save(f, array)
        del array
        force_cleanup()
    except Exception as e:
        print(f"Error saving batch array: {e}")
        force_cleanup()
        raise

def process_batch(batch_items, encoder, model_name, rep_type, fbank_dir, task_cfg, offset, mean_pooling, span, utt_id_dct=None):
    """Process a batch of items and return their representations"""
    batch_rep_dct = {}
    batch_transformed_fbank = []
    batch_truncated_fbank = []
    batch_quantized_features = []
    batch_quantized_indices = []
    batch_labels = []

    # Match original dictionary accumulation
    batch_quantized_features_dct = {}
    batch_discrete_indices_dct = {}

    for item in batch_items:
        if span == "frame" or span == "utt":
            time_stamp_lst = None
            utt_id, wav_fn = item.split("\t")
        elif ".lst" in str(utt_id_dct):  # all-words with samples saved as lst
            utt_id, wav_fn, start_time, end_time, wrd = item.split(",")
            time_stamp_lst = [[start_time, end_time, wrd]]
        else:
            utt_id = item
            wav_fn = utt_id_dct[utt_id][0]
            time_stamp_lst = utt_id_dct[utt_id][1:]
            if time_stamp_lst:
                batch_labels.extend([x[2] for x in time_stamp_lst])  # Collect labels from timestamps

        extract_obj = FeatExtractor(
            encoder,
            utt_id,
            wav_fn,
            rep_type,
            model_name,
            fbank_dir,
            task_cfg,
            offset=offset,
            mean_pooling=mean_pooling,
        )
        getattr(extract_obj, model_name.split("_")[0])()

        if rep_type == "local":
            extract_obj.extract_local_rep(
                batch_rep_dct, 
                batch_transformed_fbank,
                batch_truncated_fbank
            )
        elif rep_type == "contextualized":
            #print(f'timestamp {time_stamp_lst}')
            extract_obj.extract_contextualized_rep(batch_rep_dct, time_stamp_lst, batch_labels)
        elif rep_type == "quantized":
            extract_obj.extract_quantized_rep(
                batch_quantized_features,
                batch_quantized_indices,
                batch_quantized_features_dct,
                batch_discrete_indices_dct
            )
        #print("\nCurrent batch statistics:")
        #for layer_num in batch_rep_dct:
        #    print(f"Layer {layer_num}: {len(batch_rep_dct[layer_num])} representations")
        
        del extract_obj
        force_cleanup()
    return (batch_rep_dct, batch_transformed_fbank, batch_truncated_fbank,
            batch_quantized_features, batch_quantized_indices, batch_labels,
            batch_quantized_features_dct, batch_discrete_indices_dct)

def save_rep(
    model_name,
    ckpt_pth,
    save_dir,
    utt_id_fn,
    model_type="pretrained",
    rep_type="contextualized",
    dict_fn=None,
    fbank_dir=None,
    offset=False,
    mean_pooling=False,
    span="frame",
    pckg_dir=None,
    batch_size=50, #batch_size parameter
):
    """
    Extract layer-wise representations from the model

    ckpt_pth: path to the model checkpoint
    save_dir: directory where the representations are saved
    utt_id_fn: identifier for utterances
    model_type: pretrained or finetuned
    rep_type: contextualized or local or quantized
    dict_fn: path to dictionary file in case of finetuned models
    fbank_dir: directory that has filterbanks stored
    offset: span representation attribute
    mean_pooling: span representation attribute
    span: frame | phone | word
    """
    assert rep_type in ["local", "quantized", "contextualized"]
    print("Initial memory usage:")
    log_memory_usage()

    model_obj = ModelLoader(ckpt_pth, model_type, pckg_dir, dict_fn)
    encoder, task_cfg = getattr(model_obj, model_name.split("_")[0])()
    del model_obj
    force_cleanup()
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    # added to help with memory constraints
    temp_dir = os.path.join(save_dir, "temp_batches")
    Path(temp_dir).mkdir(exist_ok=True)
    print(utt_id_fn)
    if ".tsv" in utt_id_fn:
        utt_id_lst = read_lst(utt_id_fn)
        utt_id_dct = None
    elif ".lst" in utt_id_fn:
        utt_id_lst = read_lst(utt_id_fn)
        label_lst_fn = utt_id_fn.replace("word_segments_", "labels_")
        new_label_lst_fn = os.path.join(save_dir, "..", f'labels_{save_dir.split("/")[-1]}.lst')
        if not os.path.exists(new_label_lst_fn):
            shutil.copy(label_lst_fn, new_label_lst_fn)
        utt_id_dct = ".lst" # marker for lst files
    else:
        utt_id_dct = load_dct(utt_id_fn)
        utt_id_lst = list(utt_id_dct.keys())
        label_lst_fn = os.path.join(
            save_dir, "..", f'labels_{save_dir.split("/")[-1]}.lst')
    #print(label_lst_fn)

    #Initialize for quantized type
    quantized_features_dct = {}
    discrete_indices_dct = {}
    # Initialize structure for incremental layer files
    if rep_type != "quantized":
        layer_files = {}
    # Process in batches
    start = time.time()
    batches = [utt_id_lst[i:i + batch_size] for i in range(0, len(utt_id_lst), batch_size)]
    # Setup temporary files
    temp_files = {
        'transformed': os.path.join(temp_dir, "transformed_fbank.npy"),
        'truncated': os.path.join(temp_dir, "truncated_fbank.npy"),
        'features': os.path.join(temp_dir, "features.npy"),
        'indices': os.path.join(temp_dir, "indices.npy")
    }        
    labels_lst=[]
    for batch_idx, batch_items in enumerate(tqdm(batches)):
        (batch_rep_dct, batch_transformed_fbank, batch_truncated_fbank,
         batch_quantized_features, batch_quantized_indices, batch_labels,
         batch_quantized_features_dct, batch_discrete_indices_dct) = process_batch(
            batch_items, encoder, model_name, rep_type, fbank_dir, task_cfg,
            offset, mean_pooling, span, utt_id_dct
        )
        labels_lst.extend(batch_labels)
    	# Save batch results to temporary files
        if rep_type != "quantized":
                for key, value in batch_rep_dct.items():
                    value_array = np.vstack(value)
                    #print(f'value {value_array.shape}')
                    if key not in layer_files:
                        layer_files[key] = os.path.join(save_dir, f"layer_{key}.npy")
                        np.save(layer_files[key], value_array)
                    else:
                        existing_array = np.load(layer_files[key])
                        combined_array = np.vstack([existing_array, value_array])
                        total_reps = combined_array.shape[0]
                        np.save(layer_files[key], combined_array)
                        # Append to existing file
                        #value_array = np.array(value, dtype=object)
                        #with open(layer_files[key], 'ab') as f:
                        #    np.save(f, value_array)
                    del value_array
                del batch_rep_dct
                force_cleanup()

        # Save batch results to temporary files
        if rep_type == "local":
            if batch_transformed_fbank:
                save_batch_array(batch_transformed_fbank, temp_files['transformed'], 
                                   'ab' if batch_idx > 0 else 'wb')
            if batch_truncated_fbank:
                save_batch_array(batch_truncated_fbank, temp_files['truncated'],
                                   'ab' if batch_idx > 0 else 'wb')

        elif rep_type == "quantized":
            if batch_quantized_features:
                save_batch_array(batch_quantized_features, temp_files['features'],
                                   'ab' if batch_idx > 0 else 'wb')
            if batch_quantized_indices:
                save_batch_array(batch_quantized_indices, temp_files['indices'],
                                   'ab' if batch_idx > 0 else 'wb')
        # Clear batch data
        
        del batch_transformed_fbank, batch_truncated_fbank
        del batch_quantized_features, batch_quantized_indices
        force_cleanup()
        # Clear batch memory
        gc.collect()
        torch.cuda.empty_cache()
    
    log_memory_usage()
    #print(label_lst_fn)
    if span in ["phone", "word"]:
        write_to_file("\n".join(labels_lst), label_lst_fn)

    if rep_type == "local":
        if os.path.exists(temp_files['truncated']):
            if "avhubert" not in model_name:
                shutil.move(temp_files['truncated'], 
                          os.path.join(fbank_dir, "all_features.npy"))
                sfx = ""
            else:
                sfx = "_by4"
            
        if os.path.exists(temp_files['transformed']):
            shutil.move(temp_files['transformed'],
                       os.path.join(fbank_dir, f"all_features_downsampled{sfx}.npy"))

    elif rep_type == "quantized":
        if os.path.exists(temp_files['features']):
            shutil.move(temp_files['features'], os.path.join(save_dir, "features.npy"))
        if os.path.exists(temp_files['indices']):
            shutil.move(temp_files['indices'], os.path.join(save_dir, "indices.npy"))
        save_dct(os.path.join(save_dir, "quantized_features.pkl"), quantized_features_dct)
        save_dct(os.path.join(save_dir, "discrete_indices.pkl"), discrete_indices_dct)

    # Final cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    force_cleanup() 
    print("Final memory usage:")
    log_memory_usage()
    print("%s representations saved to %s" % (rep_type, save_dir))

    print("Time required: %.1f mins" % ((time.time() - start) / 60))

def combine(
    model_name,
    save_dir,
    subfname="all_words_200instances",
    layer_num=-1,
):
    """
    Combine all extracted contextualzed word embeddings into a single pkl file
    """
    embedding_dir = os.path.join(
        save_dir,
        model_name,
        "librispeech",
        subfname)
    num_splits = len(glob(os.path.join(embedding_dir, "*", "layer_0.npy")))
    num_layers = len(glob(os.path.join(embedding_dir, "0", "layer_*.npy")))
    
    labels_lst = read_lst(os.path.join(embedding_dir, "labels_0.lst"))
    for split_num in range(1, num_splits):
        labels_lst_1 = read_lst(os.path.join(embedding_dir, f"labels_{split_num}.lst"))
        labels_lst.extend(labels_lst_1)

    if layer_num == -1:
        layer_lst = np.arange(num_layers)
        print("Combining representations for all layers")
    else:
        layer_lst = [layer_num]
        print(f"Combining representations for layer {layer_num}")
    for layer_num in layer_lst:
        print(layer_num)
        rep_mat = np.load(os.path.join(embedding_dir, "0", f"layer_{layer_num}.npy"))
        for split_num in tqdm(range(1, num_splits)):
            rep_mat_1 = np.load(os.path.join(embedding_dir, str(split_num), f"layer_{layer_num}.npy"))
            rep_mat = np.concatenate((rep_mat, rep_mat_1), axis=0)
        np.save(os.path.join(embedding_dir, f"layer_{layer_num}.npy"), rep_mat)
   
    write_to_file("\n".join(labels_lst), os.path.join(embedding_dir, "labels.lst"))


if __name__ == "__main__":
    fire.Fire({
        "save_rep": save_rep,
        "combine": combine, 
    })
