import matplotlib.pyplot as plt
import numpy as np


def main():
	# segmentation_cross_validation_distribution()
	# segmentation_cross_validation_results()
	# segmentation_comparison_distribution()
	# classification_cross_validation_distribution()
	# classification_cross_validation_results()
	classification_comparison_distribution()



def segmentation_cross_validation_distribution():
	datasets = ('AISD', 'PhysioNet-ICH', 'BHSD', 'Seg-CQ500', 'Total')
	sample_counts = [('Original', np.array([397, 36, 192, 51, 676]), '#7FB2D5CC'), ('Augmented', np.array([397, 288, 0, 255, 940]), '#B3D1E7CC')]

	# plt.figure(figsize=(8, 4.8))
	# plt.rcParams.update({'font.family': 'Times New Roman'})
	plt.rcParams.update({'font.size': 14})
	plt.rcParams.update({'font.weight': 500})

	fig, ax = plt.subplots(figsize=(8, 4))

	bottom = np.zeros(5)

	for i, (sample, sample_count, color) in enumerate(sample_counts):
		p = ax.bar(datasets, sample_count, 0.6, label=sample, bottom=bottom, color=color)
		bottom += sample_count

		labels = [v if v > 0 else '' for v in sample_count]

		ax.bar_label(p, labels=labels, label_type='center')
		
		if i == len(sample_counts) - 1:
			labels = np.array([sample_count[1] for sample_count in sample_counts])
			labels = np.sum(labels, axis=0)
			ax.bar_label(p, labels=labels)

	for lab in ax.get_xticklabels():
		if lab.get_text() == 'Total':
			lab.set_fontweight('bold')

	ax.legend()
	ax.set_ylim([0,1800])
	plt.savefig('../test/segmentation/cross_validation_distribution.png', bbox_inches='tight', dpi=300)
	plt.show()



def segmentation_cross_validation_results():
	metrics = ('Dice', 'IOU', 'Precision', 'Recall')
	values = {'AISD': (0.5793, 0.3848, 0.6616, 0.5282), 'PhysioNet-ICH': (0.5060, 0.3562, 0.7750, 0.4783), 'BHSD': (0.6406, 0.4712, 0.7321, 0.5977), 'Seg-CQ500': (0.6910, 0.5034, 0.7027, 0.6796), 'Overall': (0.6434, 0.4507, 0.7053, 0.5983)}
	# colors = {'AISD': '#045275', 'PhysioNet-ICH': '#7CCBA2', 'BHSD': '#E9E29C', 'Seg-CQ500': '#EEB479', 'Overall': '#CF597E'}
	colors = {'AISD': '#88CEE6', 'PhysioNet-ICH': '#F6C8A8', 'BHSD': '#E89DA0', 'Seg-CQ500': '#B2D3A4', 'Overall': '#B696B6'}

	x = np.arange(len(metrics))
	width = 0.15
	multiplier = 0

	plt.rcParams.update({'font.size': 14})
	plt.rcParams.update({'font.weight': 500})

	fig, ax = plt.subplots(layout='constrained', figsize=(8, 3.6))

	for dataset, value in values.items():
		offset = width * multiplier
		rects = ax.bar(x + offset, [num*100 for num in value], width-0.03, label=dataset, color=colors[dataset])
		ax.bar_label(rects, padding=5, fontsize=10, rotation=45)
		multiplier += 1

	ax.set_xticks(x + 2*width, metrics)

	ax.legend(ncol=3)
	ax.set_ylabel('%', rotation=0)
	ax.set_ylim([30,105])
	ax.yaxis.set_label_coords(0.02,0.92)

	plt.savefig('../test/segmentation/cross_validation_results.png', bbox_inches='tight', dpi=300)
	plt.show()



def segmentation_comparison_distribution():
	metrics = ('AISD', 'PhysioNet-ICH', 'BHSD', 'Seg-CQ500', 'Normal', 'Total')
	values = {'Training (original)': (345, 0, 192, 51, 200, 788), 'Training (augmented)': (345, 0, 192, 0, 0, 437), 'Testing': (52, 36, 0, 0, 0, 88)}
	colors = {'Training (original)': '#7FB2D5CC', 'Training (augmented)': '#B3D1E7CC', 'Testing': '#F47F72CC'}

	x = np.arange(len(metrics))
	offsets = {'Training (original)': (0, 0, 0.15, 0.15, 0.15, 0), 'Training (augmented)': (0, 0, 0.15, 0.15, 0.15, 0), 'Testing': (0.3, 0.15, 0.6, 0.6, 0.6, 0.3)}
	widths = {'Training (original)': (0.3, 0, 0.6, 0.6, 0.6, 0.3), 'Training (augmented)': (0.3, 0, 0.6, 0.6, 0.6, 0.3), 'Testing': (0.3, 0.6, 0, 0, 0, 0.3)}
	multiplier = 1

	plt.rcParams.update({'font.size': 14})
	plt.rcParams.update({'font.weight': 500})

	fig, ax = plt.subplots(layout='constrained', figsize=(8, 4))

	bottom = np.zeros(6)

	for dataset, value in values.items():
		if dataset == 'Training (augmented)':
			offset = [off * multiplier for off in offsets[dataset]]
			rects = ax.bar(x + offset, value, np.array(widths[dataset])-0.03, bottom=bottom, label=dataset, color=colors[dataset])
			labels = [v if v > 0 else '' for v in value]
			ax.bar_label(rects, labels=labels, label_type='center')

			labels = [values['Training (original)'][i] + values['Training (augmented)'][i] if values['Training (augmented)'][i] > 0 else '' for i in range(len(values['Training (augmented)']))]
			ax.bar_label(rects, labels=labels)

		else:
			offset = np.array([off * multiplier for off in offsets[dataset]])
			indices = [i for i in range(len(widths[dataset])) if widths[dataset][i] > 0]
			rects = ax.bar(x[indices] + offset[indices], np.array(value)[indices], np.array(widths[dataset])[indices]-0.03, label=dataset, color=colors[dataset])
			labels = [v for v in value if v > 0]
			ax.bar_label(rects, labels=labels, label_type='center')
			# multiplier += 1

			bottom += value

	ax.set_xticks(x + 0.5*0.3, metrics)

	for lab in ax.get_xticklabels():
		if lab.get_text() == 'Total':
			lab.set_fontweight('bold')
	ax.legend()
	ax.set_ylim([0,1380])
	plt.savefig('../test/segmentation/comparison_distribution.png', bbox_inches='tight', dpi=300)
	plt.show()



def classification_cross_validation_distribution():
	metrics = ('AISD+', 'BGD-ISD', 'PhysioNet-ICH', 'RSNA', 'CQ500', 'Total')
	values = {'Normal': (148, 148, 46, 148, 286, 776), 'Ischemic': (345, 431, 0, 0, 0, 776), 'Ischemic (augmented)': (0, 0, 0, 0, 0, 0), 'Hemorrhagic': (0, 0, 36, 463, 205, 704), 'Hemorrhagic (augmented)': (0, 0, 72, 0, 0, 72)}
	colors = {'Normal': '#7FB2D5CC', 'Ischemic': '#BFBCDACC', 'Ischemic (augmented)': '#DAD7EACC', 'Hemorrhagic': '#F47F72CC', 'Hemorrhagic (augmented)': '#F6B3ACCC'}

	x = np.arange(len(metrics))
	offsets = {'Normal': (0,0,0,0,0,-0.15), 'Ischemic': (0.3,0.3,0,0,0,0.15), 'Ischemic (augmented)': (0.3,0.3,0,0,0,0.15), 'Hemorrhagic': (0,0,0.3,0.3,0.3,0.45), 'Hemorrhagic (augmented)': (0,0,0.3,0,0,0.45)}
	widths = {'Normal': (0.3,0.3,0.3,0.3,0.3,0.3), 'Ischemic': (0.3,0.3,0,0,0,0.3), 'Ischemic (augmented)': (0.3,0.3,0,0,0,0.3), 'Hemorrhagic': (0,0,0.3,0.3,0.3,0.3), 'Hemorrhagic (augmented)': (0,0,0.3,0,0,0.3)}
	multiplier = 1

	plt.rcParams.update({'font.size': 14})
	plt.rcParams.update({'font.weight': 500})

	fig, ax = plt.subplots(layout='constrained', figsize=(8, 4))

	bottom = np.zeros(6)

	for dataset, value in values.items():
		if dataset == 'Ischemic (augmented)':
			offset = [off * multiplier for off in offsets[dataset]]
			rects = ax.bar(x + offset, value, np.array(widths[dataset])-0.03, bottom=bottom, label=dataset, color=colors[dataset])
			labels = [v if v > 0 else '' for v in value]
			ax.bar_label(rects, labels=labels, label_type='center')

			labels = [values['Ischemic'][i] + values['Ischemic (augmented)'][i] if values['Ischemic (augmented)'][i] > 0 else '' for i in range(len(values['Ischemic (augmented)']))]
			ax.bar_label(rects, labels=labels, fontsize=12)
			bottom = np.zeros(6)

		elif dataset == 'Hemorrhagic (augmented)':
			offset = [off * multiplier for off in offsets[dataset]]
			rects = ax.bar(x + offset, value, np.array(widths[dataset])-0.03, bottom=bottom, label=dataset, color=colors[dataset])
			labels = [v if v > 0 else '' for v in value]
			ax.bar_label(rects, labels=labels, label_type='center')

			labels = [values['Hemorrhagic'][i] + values['Hemorrhagic (augmented)'][i] if values['Hemorrhagic (augmented)'][i] > 0 else '' for i in range(len(values['Hemorrhagic (augmented)']))]
			ax.bar_label(rects, labels=labels, fontsize=12)
			bottom = np.zeros(6)

		else:
			offset = np.array([off * multiplier for off in offsets[dataset]])
			indices = [i for i in range(len(widths[dataset])) if widths[dataset][i] > 0]
			rects = ax.bar(x[indices] + offset[indices], np.array(value)[indices], np.array(widths[dataset])[indices]-0.03, label=dataset, color=colors[dataset])
			labels = [v for v in value if v > 0]
			ax.bar_label(rects, labels=labels, label_type='center')
			# multiplier += 1

			if dataset == 'Ischemic' or dataset == 'Hemorrhagic':
				bottom += value

	ax.set_xticks(x + 0.5*0.3, metrics)

	for lab in ax.get_xticklabels():
		if lab.get_text() == 'Total':
			lab.set_fontweight('bold')
	ax.legend()
	ax.set_ylim([0,820])
	plt.savefig('../test/classification/cross_validation_distribution.png', bbox_inches='tight', dpi=300)
	plt.show()



def classification_cross_validation_results():
	metrics = ('AUC', 'Dice', 'Recall', 'Specificity')
	values = {'AISD+': (0.9625, 0.9660, 0.9812, 0.9020), 'BGD-ISD': (0.9577, 0.8947, 0.8630, 0.9901), 'AIS overall': (0.9600, 0.9161, 0.8987, 0.9821), 'PhysioNet-ICH': (0.8393, 0.6154, 0.6667, 0.6667), 'RSNA': (0.9457, 0.9079, 0.9542, 0.7571), 'CQ500': (0.9481, 0.9258, 0.9077, 0.9132), 'ICH overall': (0.9319, 0.8868, 0.8226, 0.8001)}
	colors = {'AISD+': '#88CEE6', 'BGD-ISD': '#D3D3D3', 'AIS overall': '#E6CECF', 'PhysioNet-ICH': '#F6C8A8', 'RSNA': '#E89DA0', 'CQ500': '#B2D3A4', 'ICH overall': '#B696B6'}

	x = np.arange(len(metrics))
	width = 0.1
	multiplier = 0

	plt.rcParams.update({'font.size': 14})
	plt.rcParams.update({'font.weight': 500})

	fig, ax = plt.subplots(layout='constrained', figsize=(8, 3.6))

	for dataset, value in values.items():
		offset = width * multiplier
		if dataset in ['PhysioNet-ICH', 'RSNA', 'CQ500', 'ICH overall']:
			offset += 0.05
		rects = ax.bar(x + offset, [num*100 for num in value], width-0.03, label=dataset, color=colors[dataset])
		ax.bar_label(rects, padding=5, fontsize=10, rotation=45)
		multiplier += 1

	ax.set_xticks(x + 0.32, metrics)

	ax.legend(ncol=3)
	ax.set_ylabel('%', rotation=0)
	ax.set_ylim([50,140])
	ax.yaxis.set_label_coords(0.02,0.92)

	plt.savefig('../test/classification/cross_validation_results.png', bbox_inches='tight', dpi=300)
	plt.show()




def classification_comparison_distribution():
	metrics = ('AISD', 'RSNA', 'BGD-ISD', 'CQ500')
	values = {'Normal': (0, 1725, 450, 286), 'Ischemic': (345, 0, 431, 0), 'Ischemic (augmented)': (1380, 0, 0, 0), 'Hemorrhagic': (0, 1725, 0, 205), 'Hemorrhagic (augmented)': (0, 0, 0, 0)}
	colors = {'Normal': '#7FB2D5CC', 'Ischemic': '#BFBCDACC', 'Ischemic (augmented)': '#DAD7EACC', 'Hemorrhagic': '#F47F72CC', 'Hemorrhagic (augmented)': '#F6B3ACCC'}

	x = np.arange(len(metrics))
	offsets = {'Normal': (0,0,0,0), 'Ischemic': (0.15,0,0.3,0), 'Ischemic (augmented)': (0.15,0,0.3,0), 'Hemorrhagic': (0,0.3,0.3,0.3), 'Hemorrhagic (augmented)': (0,0.3,0,0)}
	widths = {'Normal': (0,0.3,0.3,0.3), 'Ischemic': (0.3,0,0.3,0), 'Ischemic (augmented)': (0.3,0,0,0), 'Hemorrhagic': (0,0.3,0,0.3), 'Hemorrhagic (augmented)': (0,0,0,0)}
	multiplier = 1

	plt.rcParams.update({'font.size': 14})
	plt.rcParams.update({'font.weight': 500})

	fig, ax = plt.subplots(layout='constrained', figsize=(8, 4.8))

	bottom = np.zeros(4)

	for dataset, value in values.items():
		if dataset == 'Ischemic (augmented)':
			offset = [off * multiplier for off in offsets[dataset]]
			rects = ax.bar(x + offset, value, np.array(widths[dataset])-0.03, bottom=bottom, label=dataset, color=colors[dataset])
			labels = [v if v > 0 else '' for v in value]
			ax.bar_label(rects, labels=labels, label_type='center')

			labels = [values['Ischemic'][i] + values['Ischemic (augmented)'][i] if values['Ischemic (augmented)'][i] > 0 else '' for i in range(len(values['Ischemic (augmented)']))]
			ax.bar_label(rects, labels=labels, fontsize=12)
			bottom = np.zeros(4)

		elif dataset == 'Hemorrhagic (augmented)':
			offset = [off * multiplier for off in offsets[dataset]]
			rects = ax.bar(x + offset, value, np.array(widths[dataset])-0.03, bottom=bottom, label=dataset, color=colors[dataset])
			labels = [v if v > 0 else '' for v in value]
			ax.bar_label(rects, labels=labels, label_type='center')

			labels = [values['Hemorrhagic'][i] + values['Hemorrhagic (augmented)'][i] if values['Hemorrhagic (augmented)'][i] > 0 else '' for i in range(len(values['Hemorrhagic (augmented)']))]
			ax.bar_label(rects, labels=labels, fontsize=12)
			bottom = np.zeros(4)

		else:
			offset = np.array([off * multiplier for off in offsets[dataset]])
			indices = [i for i in range(len(widths[dataset])) if widths[dataset][i] > 0]
			rects = ax.bar(x[indices] + offset[indices], np.array(value)[indices], np.array(widths[dataset])[indices]-0.03, label=dataset, color=colors[dataset])
			labels = [v for v in value if v > 0]
			ax.bar_label(rects, labels=labels, label_type='center')
			# multiplier += 1

			if dataset == 'Ischemic' or dataset == 'Hemorrhagic':
				bottom += value

	ax.set_xticks(x + 0.5*0.3, metrics)

	for lab in ax.get_xticklabels():
		if lab.get_text() == 'BGD-ISD' or lab.get_text() == 'CQ500':
			lab.set_fontweight('bold')
	ax.legend()
	ax.set_ylim([0,1800])

	plt.savefig('../test/classification/comparison_distribution.png', bbox_inches='tight', dpi=300)
	plt.show()



if __name__ == '__main__':
	main()