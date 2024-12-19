import numpy as np
import os
import sys

import fairseq
import soundfile as sf
import torch
import torch.nn.functional as F
import librosa

from prepare_utils import PER_LAYER_TRANSFORM_DCT

curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(curr_dir, ".."))
from utils import load_dct


class ModelLoader:
    def __init__(self, ckpt_pth, model_type, pckg_dir=None, dict_fn=None):
        """
        load model from ckpt_pth
        """
        self.ckpt_pth = ckpt_pth
        self.model_type = model_type
        self.pckg_dir = pckg_dir
        self.dict_fn = dict_fn

    def wavlm(self):
        sys.path.insert(0, self.pckg_dir)
        from WavLM import WavLM, WavLMConfig

        ckpt = torch.load(self.ckpt_pth)
        cfg = WavLMConfig(ckpt["cfg"])
        encoder = WavLM(cfg)
        encoder.load_state_dict(ckpt["model"])
        return encoder, cfg

    def avhubert(self):
        from argparse import Namespace

        fairseq.utils.import_user_module(Namespace(user_dir=self.pckg_dir))
        models, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.ckpt_pth]
        )
        encoder = models[0]
        return encoder, task.cfg

    def fairseq_model_loader(self):
        if self.model_type == "finetuned":
            assert self.dict_fn
            model, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [self.ckpt_pth],
                arg_overrides={"data": self.dict_dn},
            )
            model = model[0]
            encoder = model.w2v_encoder._modules["w2v_model"]
        else:
            (
                encoder,
                _,
                task,
            ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.ckpt_pth])
            encoder = encoder[0]
        task_cfg = task.cfg
        return encoder, task_cfg

    def wav2vec(self):
        encoder, task_cfg = self.fairseq_model_loader()
        return encoder, task_cfg

    def xlsr53(self):
        encoder, task_cfg = self.fairseq_model_loader()
        return encoder, task_cfg

    def xlsr128(self):
        encoder, task_cfg = self.fairseq_model_loader()
        return encoder, task_cfg

    def hubert(self):
        encoder, task_cfg = self.fairseq_model_loader()
        return encoder, task_cfg

    def fastvgs(self):
        sys.path.append(self.pckg_dir)
        from models import w2v2_model

        args = load_dct(f"{self.ckpt_pth}/args.pkl")
        weights = torch.load(f"{self.ckpt_pth}/best_bundle.pth")
        model = w2v2_model.Wav2Vec2Model_cls(args)
        model.carefully_load_state_dict(
            weights["dual_encoder"]
        )  # will filter out weights that don't belong to w2v2
        return model, args

    def fastvgs_coco(self):
        self.fastvgs()

    def fastvgs_places(self):
        self.fastvgs()

    def fastvgs_plus_coco(self):
        self.fastvgs()
    
    def randominit(self):
        """
        Uses pre-trained model ckpt to obtain model arguments and task config
        """
        import fairseq.models.wav2vec.wav2vec2 as w2v
        _, args, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.ckpt_pth])
        task_cfg = task.cfg
        cfg_cls = w2v.Wav2Vec2Config(**args['model'])
        encoder = w2v.Wav2Vec2Model(cfg_cls)
        return encoder, task_cfg


class DataLoader:
    def __init__(
        self,
        wav_fn,
        task_cfg=None,
    ):
        audio, fs = sf.read(wav_fn)
        # the audio may be at a different sampling rate, so we force a certain one
        resampled = librosa.resample(audio, orig_sr=fs, target_sr=16000)
        self.audio = resampled.astype(audio.dtype)
        self.fs = 16000
        self.task_cfg = task_cfg

    def stacker(self, feats, stack_order):
        """
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % stack_order != 0:
            res = stack_order - len(feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, stack_order, feat_dim)).reshape(
            -1, stack_order * feat_dim
        )
        return feats

    def avhubert(self):
        # 26-dim logfbank as input features
        from python_speech_features import logfbank

        audio_feats = logfbank(self.audio, samplerate=self.fs).astype(
            np.float32
        )  # [T, F]
        in_data = self.stacker(audio_feats, self.task_cfg.stack_order_audio)
        # [T/stack_order_audio, stack_order_audio*F]
        in_data = torch.from_numpy(in_data.astype(np.float32))
        if self.task_cfg.normalize:
            with torch.no_grad():
                in_data = F.layer_norm(in_data, in_data.shape[1:])
        in_data = torch.unsqueeze(in_data, 0)
        return in_data  # BxTxF

    def wavlm(self):
        in_data = torch.from_numpy(np.expand_dims(self.audio, 0).astype("float32"))
        if self.task_cfg.normalize:
            in_data = F.layer_norm(in_data, in_data.shape)
        return in_data

    def fairseq_indata(self):
        in_data = torch.from_numpy(np.expand_dims(self.audio, 0).astype("float32"))
        if self.task_cfg.normalize:
            in_data = F.layer_norm(in_data, in_data.shape)
        return in_data

    def wav2vec(self):
        in_data = self.fairseq_indata()
        return in_data

    def xlsr128(self):
        in_data = self.fairseq_indata()
        return in_data

    def xlsr53(self):
        in_data = self.fairseq_indata()
        return in_data

    def hubert(self):
        in_data = self.fairseq_indata()
        return in_data

    def randominit(self):
        in_data = self.fairseq_indata()
        return in_data

    def fastvgs(self):
        assert self.fs == 16000
        in_data = (self.audio - np.mean(self.audio)) / np.std(self.audio)
        in_data = torch.from_numpy(np.expand_dims(in_data, 0).astype("float32"))
        return in_data

    def fastvgs_coco(self):
        self.fastvgs()

    def fastvgs_plus_coco(self):
        self.fastvgs()

    def fastvgs_places(self):
        self.fastvgs()


class FeatExtractor:
    def __init__(
        self,
        encoder,
        utt_id,
        wav_fn,
        rep_type,
        model_name,
        fbank_dir=None,
        task_cfg=None,
        offset=False,
        mean_pooling=False,
    ):
        data_obj = DataLoader(wav_fn, task_cfg)
        self.task_cfg = task_cfg
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_data = getattr(data_obj, self.model_name.split("_")[0])().to(
            self.device
        )
        self.encoder = encoder
        self.encoder.eval()
        self.encoder.to(self.device)
        self.offset = offset
        self.mean_pooling = mean_pooling
        self.rep_type = rep_type
        if self.rep_type == "local":
            self.fbank_dir = fbank_dir
            self.fbank = np.load(os.path.join(fbank_dir, utt_id + ".npy"))
        self.contextualized_features = {}
        self.local_features = {}
        self.utt_id = utt_id

    def avhubert(self):
        # model only has a projection layer before the transformer module
        with torch.no_grad():
            # Specify output_layer if you want to extract feature of an intermediate layer
            _, all_features, in_rep, _ = self.encoder.extract_finetune(
                source={"video": None, "audio": self.in_data.transpose(1, 2)},
                output_layer=None,
                padding_mask=None,
            )
        self.contextualized_features[0] = (
            in_rep.transpose(1, 2).squeeze(0).cpu().numpy()
        )
        layer_num = 1
        for layer_rep, _ in all_features:
            self.contextualized_features[layer_num] = layer_rep.squeeze(1).cpu().numpy()
            layer_num += 1
        self.stride_sec = 40 / 1000
        self.n_frames = len(in_rep.transpose(1, 2).squeeze(0))

    def wavlm(self):
        if self.rep_type == "contextualized":
            with torch.no_grad():
                rep, layer_results = self.encoder.extract_features(
                    self.in_data,
                    output_layer=self.encoder.cfg.encoder_layers,
                    ret_layer_results=True,
                    ret_conv=True,
                )[0]
            layer_num = 0
            for layer_rep, _ in layer_results:
                self.contextualized_features[layer_num] = (
                    layer_rep.transpose(0, 1).squeeze(0).cpu().numpy()
                )
                layer_num += 1
            self.n_frames = self.contextualized_features[0].shape[0]
        elif self.rep_type == "local":
            with torch.no_grad():
                self.local_features = self.encoder.feature_extractor(self.in_data)[1]
            self.n_frames = self.local_features[0].shape[-1]
        self.stride_sec = 20 / 1000

    def fairseq_extractor(self):
        with torch.no_grad():
        #get local features
            #features, intermediate_features = self.encoder.forward_features(self.in_data)
            #in_rep, intermediate_features = self.encoder.feature_extractor(self.in_data)
            
            # Get features directly from feature_extractor for wav2vec model
            # Need to access .conv_layers directly to get intermediate features
            x = self.in_data.unsqueeze(1)  # BxT -> BxCxT
            intermediate_features = []
        
            # Manually track intermediate features through conv layers
            for i, conv in enumerate(self.encoder.feature_extractor.conv_layers):
                x = conv(x)
                #print(f"After conv layer {i}: {x.shape}")
                intermediate_features.append(x)
            
            features = x  # Final output is the last layer's output
            
            # process features through layer norm if needed
            in_rep = features.transpose(1, 2)
            in_rep = self.encoder.layer_norm(in_rep)
        if self.rep_type == "contextualized":
            if "hubert" in self.model_name:
                # HuBERT specific path
                num_layers = len(self.encoder.encoder.layers) 
                features, padding_mask = self.encoder.extract_features(
                    self.in_data,
                    output_layer=num_layers
                )
                x = features
                for layer_idx, layer in enumerate(self.encoder.encoder.layers):
                    x, _ = layer(x, padding_mask)
                    self.contextualized_features[layer_idx] = x.squeeze(0).detach().cpu().numpy()
                
            else:
            #encoder_out = self.encoder(self.in_data, features_only=True, mask=False)
                encoder_out = self.encoder(
                source=self.in_data,
            padding_mask=None,
            mask=False,
            features_only=True
        )
                
        
                if hasattr(encoder_out, "layer_results"):
                    for layer_num, layer_rep in enumerate(encoder_out["layer_results"]):
                        self.contextualized_features[layer_num] = (
                    layer_rep[0].squeeze(1).cpu().numpy()
                )
        if self.rep_type == "quantized" and "hubert" not in self.model_name:
            self.z_discrete, self.indices = self.encoder.quantize(self.in_data)
        if self.rep_type == "local":
            self.local_features = intermediate_features
        #self.n_frames = len(in_rep.transpose(1, 2).squeeze(0))
        self.n_frames = features.size(2) # get time dimension
        self.stride_sec = 20 / 1000

    def wav2vec(self):
        self.fairseq_extractor()

    def xlsr53(self):
        self.fairseq_extractor()

    def xlsr128(self):
        self.fairseq_extractor()

    def hubert(self):
        self.fairseq_extractor()

    def fastvgs(self):
        if "fastvgs_plus" in self.model_name:
            num_layers = 13
        else:
            num_layers = self.task_cfg.layer_use + 2
        with torch.no_grad():
            encoder_out = self.encoder(
                source=self.in_data,
                padding_mask=None,
                mask=False,
                features_only=True,
                superb=True,
                tgt_layer=None,
            )
        if self.rep_type == "contextualized":
            for layer_num, layer_rep in enumerate(encoder_out["hidden_states"]):
                if layer_num < num_layers:
                    self.contextualized_features[layer_num] = (
                        layer_rep.squeeze(0).cpu().numpy()  # TxD
                    )
        self.n_frames = self.contextualized_features[0].shape[0]
        self.stride_sec = 20 / 1000

    def randominit(self):
        self.fairseq_extractor()

    def transform_rep(self, kernel_size, stride, layer_rep):
        """
        Transform local z representations to match the fbank features' stride and receptive field
        layer_rep: torch.cuda.FloatTensor # B*C*T
        """
        #print('transformation debug')
        #print(layer_rep.shape)
        layer_rep = torch.transpose(layer_rep, 1, 0)  # 512 x 1 x num_frames
        
        #print(layer_rep.shape)
        weight = (
            torch.from_numpy(np.ones([1, 1, kernel_size]) / kernel_size)
            .type(torch.cuda.FloatTensor)
            .to(self.device)
        )
        transformed_rep = F.conv1d(layer_rep, weight, stride=stride)
        #print(transformed_rep.shape)
        transformed_rep = torch.transpose(transformed_rep, 1, 0)

        # check averaging
        mean_vec1 = torch.mean(layer_rep[:, :, :kernel_size], axis=-1)
        mean_vec2 = torch.mean(layer_rep[:, :, stride : stride + kernel_size], axis=-1)
        out_vec1 = transformed_rep[:, :, 0]
        out_vec2 = transformed_rep[:, :, 1]
        diff1 = torch.mean(mean_vec1 - out_vec1)
        diff2 = torch.mean(mean_vec2 - out_vec2)
        
        assert torch.mean(mean_vec1 - out_vec1) < 2e-7
        assert torch.mean(mean_vec2 - out_vec2) < 2e-7
        return torch.transpose(transformed_rep, 1, 2).squeeze(0).cpu().numpy()

    def extract_local_rep(self, rep_dct, transformed_fbank_lst, truncated_fbank_lst):
        for layer_num in range(1, len(self.local_features) + 1):
            _ = rep_dct.setdefault(layer_num, [])
            if layer_num == len(self.local_features):
                #print(self.local_features.size())
                #print('huh')
                curr_layer_rep = (
                    self.local_features[layer_num - 1]
                    .transpose(1, 2)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
                rep_dct[layer_num].append(curr_layer_rep)
                num_samples_last = self.local_features[layer_num - 1].shape[-1]
            else:
                transformed_rep = self.transform_rep(
                    PER_LAYER_TRANSFORM_DCT[layer_num]["kernel"],
                    PER_LAYER_TRANSFORM_DCT[layer_num]["stride"],
                    self.local_features[layer_num - 1],
                )
                rep_dct[layer_num].append(transformed_rep)
                num_samples_rest = transformed_rep.shape[0]
        fbank = (
            torch.from_numpy(self.fbank)
            .type(torch.cuda.FloatTensor)
            .to(self.device)
            .unsqueeze(0)
        )
        if "avhubert" in self.model_name:
            kernel, stride = 1, 4
            num_samples_last = self.contextualized_features[0].shape[0]
        else:
            truncated_fbank_lst.append(self.fbank.T[:num_samples_rest])         
            assert num_samples_rest < (self.fbank.shape[1] + 1)
            kernel = PER_LAYER_TRANSFORM_DCT[len(self.local_features)]["kernel"]
            stride = PER_LAYER_TRANSFORM_DCT[len(self.local_features)]["stride"]
        transformed_fbank = self.transform_rep(kernel, stride, fbank)
        assert num_samples_last < (transformed_fbank.shape[0] + 1)
        transformed_fbank_lst.append(transformed_fbank[:num_samples_last])

    def update_dct(self, indices, rep_array, rep_dct, key):
        _ = rep_dct.setdefault(key, [])
        rep_array_masked = rep_array[indices]
        if self.mean_pooling:
            rep_array_masked = np.expand_dims(np.mean(rep_array_masked, 0), 0)
        rep_dct[key].append(rep_array_masked)

    def get_segment_idx(self, start_time, end_time, len_utt):
        start_id = int(np.floor(float(start_time) / self.stride_sec))
        end_id = int(np.ceil(float(end_time) / self.stride_sec))
        if self.offset:
            offset = int(np.floor((end_id - start_id + 1) / 3))
        else:
            offset = 0
        start_id += offset
        end_id -= offset
        if end_id == start_id:
            end_id += 1
        if end_id == len_utt + 1:
            end_id = len_utt
        assert end_id > start_id

        return np.arange(start_id, end_id)

    def extract_contextualized_rep(self, rep_dct, time_stamp_lst=None, label_lst=None):
        if self.model_name == "fastvgs_coco":
            num_layers = 9
        else:
            num_layers = len(self.contextualized_features)
            
        for layer_num in range(num_layers):
            c_rep = self.contextualized_features[layer_num]
            
            #print(f"Layer {layer_num} c_rep shape: {c_rep.shape}")
            if time_stamp_lst:
                layer_words = []  # Collect all words for this layer
                #print(f"Number of words in utterance: {len(time_stamp_lst)}")
                for start_time, end_time, token in time_stamp_lst:
                    indices = self.get_segment_idx(start_time, end_time, len(c_rep))
                    # Get representation for each word
                    word_rep = np.mean(c_rep[indices], axis=0, keepdims=True)
                    #print(f"Word {token} indices: {len(indices)}")
                     #Update the representation dictionary
                    #if layer_num not in rep_dct:
                    #    rep_dct[layer_num] = []
                    #rep_dct[layer_num].append(word_rep)
                    layer_words.append(word_rep)
                    #if layer_num == 0 and label_lst is not None:
                    #    label_lst.append(token)
                # After processing all words in utterance for this layer
                if layer_words:
                    if layer_num not in rep_dct:
                        rep_dct[layer_num] = []
                    rep_dct[layer_num].extend(layer_words)
                    #print(f"\nLayer {layer_num} total representations: {len(rep_dct[layer_num])}")  
            else:
                self.update_dct(np.arange(0, len(c_rep)), c_rep, rep_dct, layer_num)

    def extract_quantized_rep(
        self,
        quantized_features,
        quantized_indices,
        quantized_features_dct,
        discrete_indices_dct,
    ):
        idx_lst = np.arange(0, self.n_frames)
        z_discrete = self.z_discrete.squeeze(0).detach().cpu().numpy()[idx_lst]
        indices = self.indices.squeeze(0).detach().cpu().numpy()[idx_lst]
        quantized_features.append(z_discrete)
        quantized_indices.append(indices)
        assert self.utt_id not in quantized_features_dct
        quantized_features_dct[self.utt_id] = z_discrete
        discrete_indices_dct[self.utt_id] = indices

    def save_rep_to_file(self, rep_dct, out_dir):
        """Save representations to file with detailed error logging"""
        try:
            print(f"\nStarting save_rep_to_file:")
            print(f"Output directory: {out_dir}")
            print(f"Directory exists: {os.path.exists(out_dir)}")
        
            nframes = []
            for layer_num, rep_lst in rep_dct.items():
                print(f"\nProcessing layer {layer_num}")
                print(f"Number of representations: {len(rep_lst)}")
            
                try:
                
                    # First, calculate total size and gather frame info
                    if layer_num == 1:
                        for rep in rep_lst:
                            nframes.append(rep.shape[0])
                
                    # Get total size for preallocating array
                    total_rows = sum(rep.shape[0] for rep in rep_lst)
                    feature_dim = rep_lst[0].shape[1]  # Assuming all reps have same feature dim
                    print(f"Total size will be: ({total_rows}, {feature_dim})")
                
                    # Create memory-mapped array for results
                    out_fn = os.path.join(out_dir, "layer_" + str(layer_num) + ".npy")
                    print(f"Creating memmap file: {out_fn}")
                    rep_mat = np.memmap(out_fn, dtype=rep_lst[0].dtype, mode='w+', 
                                  shape=(total_rows, feature_dim))
                
                    # Copy data in chunks
                    current_row = 0
                    chunk_size = 100  # Process 100 representations at a time
                    for i in range(0, len(rep_lst), chunk_size):
                        chunk = rep_lst[i:i + chunk_size]
                        chunk_data = np.concatenate(chunk, 0)
                        chunk_rows = chunk_data.shape[0]
                        rep_mat[current_row:current_row + chunk_rows] = chunk_data
                        current_row += chunk_rows
                        print(f"Processed chunk {i//chunk_size + 1}/{(len(rep_lst) + chunk_size - 1)//chunk_size}")
                
                    # Ensure data is written to disk
                    rep_mat.flush()
                    del rep_mat  # Close memmap file
                    #rep_mat = np.concatenate(rep_lst, 0)
                    #print(f"Concatenated shape: {rep_mat.shape}")
                
                    #if layer_num == 1:
                    #    for rep in rep_lst:
                    #        nframes.append(rep.shape[0])
                
                    #out_fn = os.path.join(out_dir, "layer_" + str(layer_num) + ".npy")
                    #print(f"Saving to: {out_fn}")
                    #np.save(out_fn, rep_mat)
                    #print(f"Successfully saved layer {layer_num}")
                
                except Exception as e:
                    print(f"\nError processing layer {layer_num}:")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    if len(rep_lst) > 0:
                        print(f"First rep shape: {rep_lst[0].shape}")
                    raise
        
            # Save nframes
            frames_fn = os.path.join(out_dir, "n_frames.txt")
            print(f"\nSaving n_frames to: {frames_fn}")
            print(f"Number of frames: {len(nframes)}")
        
            with open(frames_fn, 'w') as f:
                for n in nframes:
                    f.write(f'{n}\n')
            print("Successfully saved n_frames.txt")
        
        except Exception as e:
            print(f"\nError in save_rep_to_file:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Output directory: {out_dir}")
            print(f"Current working directory: {os.getcwd()}")
            raise  # Re-raise the exception after logging
