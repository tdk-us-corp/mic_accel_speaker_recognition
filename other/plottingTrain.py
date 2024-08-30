#
# Copyright (c) [2024] TDK U.S.A. Corporation
#
import matplotlib.pyplot as plt
import re

def parse_file(filename):
    epochs = []
    train_losses = []
    valid_losses = []
    error_rates = []

    pattern = re.compile(r'epoch: (\d+), lr: ([\de\.-]+) - train loss: ([\de\.-]+) - valid loss: ([\de\.-]+), valid ErrorRate: ([\de\.-]+)')
    
    with open(filename, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if not match:
                print(f"Could not parse line: {line.strip()}")
                continue  # Skip lines that do not conform to expected format

            epoch = int(match.group(1))
            lr = float(match.group(2))
            train_loss = float(match.group(3))
            valid_loss = float(match.group(4))
            error_rate = float(match.group(5))


            epochs.append(epoch)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            error_rates.append(error_rate)

    return epochs, train_losses, valid_losses, error_rates

# Replace 'data.txt' with the path to your text file
epochs, train_losses, valid_losses, error_rates = parse_file('aug_mfcc13_64size/train_log.txt')

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, valid_losses, label='Validation Loss')
plt.title('Training and Validation Loss by Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_and_validation_loss.png')
plt.close()

# Plot error rate
plt.figure(figsize=(10, 5))
plt.plot(epochs, error_rates, label='Validation Error Rate')
plt.title('Validation Error Rate by Epoch')
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.legend()
plt.grid(True)
plt.savefig('error_rate.png')
plt.close()
