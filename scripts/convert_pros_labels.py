import json
import os


def load_alignments(alignment_file):
    """
    Load the alignment file and structure it for easy lookup
    Returns a dict: {word: {utt_id: [(start_time, end_time), ...]}}
    """
    with open(alignment_file, 'r') as f:
        alignments = json.load(f)

    # Restructure for easier lookup
    structured_alignments = {}
    for word, instances in alignments.items():
        structured_alignments[word] = {}
        for instance in instances:
            utt_id = instance[0]
            wav_path = instance[1]
            wav_name = os.path.basename(wav_path)
            if utt_id not in structured_alignments[word]:
                structured_alignments[word][utt_id] = []
            structured_alignments[word][utt_id].append((float(instance[2]), float(instance[3])))

    return structured_alignments


def find_matching_word_data(word_data, target_word):
    """Find the specific word instance that matches both time and word."""
    matches = []
    for data in word_data:
        if target_word == 'mister':
            if data['word'].lower() in [target_word.lower(), 'mr']:
                matches.append(data)
        elif data['word'].lower() == target_word.lower():
            matches.append(data)
    print(matches)
    return matches[0] if matches else None

def process_text_file(input_file, order_json, alignment_file):
    """
    Process the input text file and create two label files containing the 3rd and 4th column values,
    ordered according to a JSON file

    Args:
        input_file (str): Path to the input text file
        order_json (str): Path to JSON file containing utterance order
    """
    # Load alignments
    alignments = load_alignments(alignment_file)

    # Read the ordering JSON
    with open(order_json, 'r') as f:
        order_data = json.load(f)
    print(len(order_data.keys()))
    # First pass: collect all data from text file
    data_dict = {}
    current_file = None

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('<file>'):
                current_file = line.split('\t')[1].replace('.txt', '.wav')
                data_dict[current_file] = []
            elif line and current_file:
                parts = line.split('\t')
                if parts[3] == 'NA' or parts[4] == 'NA':
                    continue
                if len(parts) >= 5:
                    word = parts[0].strip("'")
                    data_dict[current_file].append({
                        'word': word,
                        'prominence': float(parts[3]),
                        'boundary': float(parts[4])
                    })

    # Create output files by iterating through the dictionary
    out_dir = "/home/eduardo/PycharmProjects/layerwise-analysis/results/libriTTS/wav2vec_small/libriTTS_dev-clean_sample1/contextualized/word_level/"
    count = 0
    line_written = 0
    with open(f'{out_dir}prominence_labels_0.lst', 'w') as f3, open(f'{out_dir}boundary_labels_0.lst', 'w') as f4:
        for key in order_data.keys():

            wav_name = os.path.basename(order_data[key][0])
            print(count)
            print(len(order_data[key][1:]))
            # Process each word specification (starting from index 1)
            for word_spec in order_data[key][1:]:
                start_time = word_spec[0]
                target_word = word_spec[2].lower()
                count+=1
                # Use alignment information to verify this is the correct instance

                if (target_word in alignments and
                        wav_name.replace('.wav', '') in alignments[target_word].keys()):
                    # Find the matching time in alignments
                    for align_start, align_end in alignments[target_word][wav_name.replace('.wav', '')]:
                        if abs(align_start - start_time) < 0.01:

                            # Find matching word in our data
                            word_match = find_matching_word_data(
                                data_dict[wav_name],
                                target_word
                            )
                            if word_match:
                                line_written+= 1
                                print(line_written)
                                # Write to labels3.lst
                                #f3.write(f"{key}\n")
                                f3.write(f"{word_match['prominence']}\n")

                                # Write to labels4.lst
                                #f4.write(f"{key}\n")
                                f4.write(f"{word_match['boundary']}\n")
                            else:
                                f3.write(f"{0}\n")
                                f4.write(f"{0}\n")
                                print('problem')
                else:
                    print('problem')
            #wav_name = wav_name.replace(".wav", ".txt")
            # values3 = [str(entry['prominence']) for entry in data_dict[wav_name]]
            # values4 = [str(entry['boundary']) for entry in data_dict[wav_name]]
            # f3.write(f"{values3}\n")
            # f.write(f"{values4}\n")


# Example usage
if __name__ == "__main__":
    input_file = "/home/eduardo/Downloads/dev.txt"  # Replace with your input file path
    order_json = "/home/eduardo/PycharmProjects/layerwise-analysis/data_samples/libriTTS/word_level/dev-clean_segments_sample1_0.json"  # Replace with your JSON order file path
    alignment_file = "/home/eduardo/libriTTS/train-clean-100/LibriTTS/alignments/alignment_word_dev-clean.json"
    process_text_file(input_file, order_json, alignment_file)
