import os

def modify_chunks(chunk):
    new_chunk = ''
    for ch in chunk:
        new_chunk = os.path.join(new_chunk, ch)

    return new_chunk

    
        

        
output_dir = '/mnt/009-Audio/Internships/AccelAuthentification/training_data/verif_pairs/trimmed_paper_pairs'

for root, dir, files in os.walk('/mnt/009-Audio/Internships/AccelAuthentification/training_data/verif_pairs/paper_pairs'):
    for file in files:
        if file.endswith('.txt'):
            print(f'Trimming file {file}')
            input_file = os.path.join(root, file)
            new_lines = []

            with open(input_file, 'r') as file:
                for line in file.readlines():
                    chunks = line.split(' ')

                    chunks[1] = modify_chunks(chunks[1].split('/')[-4:])
                    chunks[2] = modify_chunks(chunks[2].split('/')[-4:])

                    # print(chunks[1], chunks[2], '\n')
                    new_lines.append(f'{chunks[0]} {chunks[1]} {chunks[2]}')

            base_name = input_file.split('/')[-1]
            new_file = os.path.join(output_dir, base_name)

            with open(new_file, 'w') as file:
                for line in new_lines:
                    file.write(line)


