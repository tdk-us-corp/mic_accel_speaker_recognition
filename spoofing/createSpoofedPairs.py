def process_verification_pairs(input_file_path, output_file_path, new_keyword):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    new_pairs = []
    for line in lines:
        parts = line.strip().split(' ')
        label, audio1, audio2 = parts[0], parts[1], parts[2]

        if label == '1':  # Positive label
            new_audio2 = audio2.replace('/mic/', f'/{new_keyword}/')
            new_pair = f"0 {audio1} {new_audio2}"
            new_pairs.append(new_pair)

    with open(output_file_path, 'w') as file:
        file.writelines(lines)  # Write the original lines first
        for pair in new_pairs:
            file.write(pair + '\n')

# Usage
input_file_path = 'verif_pairs/SeparateEars_4s+_1234.txt'

output_file_path = input_file_path.split('/')[1].replace('.txt', '_spoofed.txt')
# output_file_path = 'Spoofed_pairs.txt'
new_keyword = 'spoofed_mic'  # Replace with the actual keyword you want to use
process_verification_pairs(input_file_path, output_file_path, new_keyword)
