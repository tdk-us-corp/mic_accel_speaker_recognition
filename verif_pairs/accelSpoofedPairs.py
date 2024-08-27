def modify_file_contents(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        # Replace 'mic' with 'acc' in the paths
        line = line.replace('/mic/', '/acc/')
        line = line.replace('/spoofed_mic/', '/spoofed_acc/')
        
        # Add 'ACCEL_' prefix to the .wav filenames
        parts = line.split()
        modified_parts = []
        for part in parts:
            if '.wav' in part:
                path_parts = part.split('/')
                filename = 'ACCEL_' + path_parts[-1]
                path_parts[-1] = filename
                modified_part = '/'.join(path_parts)
                modified_parts.append(modified_part)
            else:
                modified_parts.append(part)
        modified_line = ' '.join(modified_parts)


        # modified_line = line
        modified_lines.append(modified_line + '\n')

# 
    with open(output_file, 'w') as file:
        file.writelines(modified_lines)

# Use the function
input_file = 'SeparateEars_4s+_1234_spoofed.txt'
output_file = input_file[:-4] + '_accel.txt'
modify_file_contents(input_file, output_file)

# 3
# 5
# None