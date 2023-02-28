from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score


"""
	Utility functions for evaluating the model performance
"""


def eval_deep(log, loader):

	data_size = len(loader.dataset.indices)
	batch_size = loader.batch_size
	if data_size%batch_size == 0:
		size_list = [batch_size] * (data_size//batch_size)
	else:
		size_list = [batch_size] * (data_size // batch_size) + [data_size%batch_size]

	assert len(log) == len(size_list)

	accuracy, f1_macro, f1_micro, precision, recall = 0, 0, 0, 0, 0

	prob_log, label_log = [], []

	for batch, size in zip(log, size_list):
		pred_y, y = batch[0].data.cpu().numpy().argmax(axis=1), batch[1].data.cpu().numpy().tolist()
		prob_log.extend(batch[0].data.cpu().numpy()[:, 1].tolist())
		label_log.extend(y)

		accuracy += accuracy_score(y, pred_y) * size
		f1_macro += f1_score(y, pred_y, average='macro') * size
		f1_micro += f1_score(y, pred_y, average='micro') * size
		precision += precision_score(y, pred_y, zero_division=0) * size
		recall += recall_score(y, pred_y, zero_division=0) * size

	auc = roc_auc_score(label_log, prob_log)
	ap = average_precision_score(label_log, prob_log)

	return accuracy/data_size, f1_macro/data_size, f1_micro/data_size, precision/data_size, recall/data_size, auc, ap


def eval_hin(log):

	pred_y, y = log[0].data.cpu().numpy().argmax(axis=1), log[1].data.cpu().numpy().tolist()
	prob_log = log[0].data.cpu().numpy()[:, 1].tolist()
	label_log = y

	accuracy = accuracy_score(y, pred_y)
	f1_macro = f1_score(y, pred_y, average='macro')
	f1_micro = f1_score(y, pred_y, average='micro')
	precision = precision_score(y, pred_y, zero_division=0)
	recall = recall_score(y, pred_y, zero_division=0)

	auc = roc_auc_score(label_log, prob_log)
	ap = average_precision_score(label_log, prob_log)

	return accuracy, f1_macro, f1_micro, precision, recall, auc, ap


def eval_shallow(pred_y, prob_y, y):

	f1_macro = f1_score(y, pred_y, average='macro')
	f1_micro = f1_score(y, pred_y, average='micro')
	accuracy = accuracy_score(y, pred_y)
	precision = precision_score(y, pred_y, zero_division=0)
	recall = recall_score(y, pred_y, zero_division=0)
	auc = roc_auc_score(y, prob_y)
	ap = average_precision_score(y, prob_y)

	return accuracy, f1_macro, f1_micro, precision, recall, auc, ap