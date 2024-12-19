dataset=$1
data_dir=$2
alignment_text_grid_dir=$4 # add in an additional argument in case alignment for nonstandard datasets exist in some other folder
alignment_data_dir=$3

mkdir -p $alignment_data_dir
if [ "$dataset" = "librispeech" ]; then
    #echo "Step 1: Downloading and extracting Librispeech alignment files"
    wget -P $alignment_data_dir https://zenodo.org/record/2619474/files/librispeech_alignments.zip
    unzip -qq $alignment_data_dir/librispeech_alignments.zip -d $alignment_data_dir
    rm $alignment_data_dir/librispeech_alignments.zip
    
    audio_ext=flac
    echo "Step 2: Saving alignments (originally in TextGrid format) in dictionary format"
    # for data_split in dev-clean dev-other train-clean-100 train-clean-360 train-other-500 test-clean test-other
    for data_split in dev-clean train-clean-100 train-clean-360 train-other-500
    do	
        echo $data_split
        audio_dir="${data_dir}/${data_split}"
        python codes/prepare/read_librispeech_alignments.py read save_data $alignment_text_grid_dir $data_split $audio_dir $audio_ext $alignment_data_dir
        # rm -r $alignment_data_dir/$data_split
    done

    echo "Step 3: Combining all train-clean-* splits into train-clean and all train-* splits into train"
    for data_split in train-clean train
    do
        echo $data_split
        python codes/prepare/read_librispeech_alignments.py combine $alignment_data_dir $data_split phone
        python codes/prepare/read_librispeech_alignments.py combine $alignment_data_dir $data_split word
    done
elif [ "$dataset" = "libriTTS" ]; then
    echo "assumption: You already git cloned the alignments at https://github.com/kan-bayashi/LibriTTSLabel?tab=readme-ov-file" 
    audio_ext=wav
    echo "Step 2: Saving alignments (originally in TextGrid format) in dictionary format"
    # for data_split in dev-clean dev-other train-clean-100 train-clean-360 train-other-500 test-clean test-other
    for data_split in dev-clean train-clean-100 train-clean-360 train-other-500 test-clean
    do	
        echo $data_split
        audio_dir="${data_dir}/${data_split}"
        python codes/prepare/read_librispeech_alignments.py read save_data $alignment_text_grid_dir $data_split $audio_dir $audio_ext $alignment_data_dir
        # rm -r $alignment_data_dir/$data_split
    done

    echo "Step 3: Combining all train-clean-* splits into train-clean and all train-* splits into train"
    for data_split in train-clean train
    do
        echo $data_split
        python codes/prepare/read_librispeech_alignments.py combine $alignment_data_dir $data_split phone
        python codes/prepare/read_librispeech_alignments.py combine $alignment_data_dir $data_split word
    done
fi
