import matplotlib.pyplot as plt
import numpy as np
import re

def get_log():
	log = []
	with open('log.txt', 'r') as f:
		log = [line.strip() for line in f.readlines()]

	return log

def extract_info(line):
	regex = re.compile(r'\[\d\d:\d\d:\d\d] Epoch [-+]?[0-9]+/[-+]?[0-9]+: train loss: (\d*\.?\d+), train accuracy: (\d*\.?\d+)%, test loss: (\d*\.?\d+), test accuracy: (\d*\.?\d+)%')
	return regex.match(line).groups()


def main():
	log = get_log()

	data = np.zeros((4, len(log)))
	data[1] *= 100
	data[3] *= 100
	
	for line_num, line in enumerate(log):
		for index, met in enumerate(extract_info(line)):
			data[index, line_num] = met


	plt.plot(data[0], color = 'g', label = 'Training Loss') 
	plt.plot(data[2], color = 'r', label = 'Test Loss') 
	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Loss Curve')
	plt.savefig('classification_loss.png')

	plt.figure()

	plt.plot(data[1], color = 'g', label = 'Training BA') 
	plt.plot(data[3], color = 'r', label = 'Test BA') 
	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('Balanced Accuracy')
	plt.title('Balanced Accuracy Curve')
	plt.savefig('classification_ba.png')




if __name__ == '__main__':
	main()