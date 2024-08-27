def modify_file_contents(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    modified_lines = lines.copy()
    for line in lines:
        # Replace 'mic' with 'acc' in the paths
        # line = line.replace('/mic/', '/acc/')
        
        # Add 'ACCEL_' prefix to the .wav filenames
        parts = line.split()

        if parts[0] == '1':
            parts[-1] = parts[-1].replace("/acc/", "/spoofed_acc/")
            parts[0] = '0'

            modified_line = ' '.join(parts)
            # print(modified_line)
            modified_lines.append(modified_line + '\n')

    with open(output_file, 'w') as file:
        file.writelines(modified_lines)


input_file = 'cleaned_split1_4+_accel.txt'
output_file = input_file[:-4] + '_spoofed.txt'
modify_file_contents(input_file, output_file)

print("Spoofed pairs have been added")

# 3
# 5
# None