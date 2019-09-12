import torch
import torch.nn as nn

class SSTask():
	def __init__(self, ext, head, criterion, optimizer, scheduler,
				 su_tr_loader, su_te_loader, tu_tr_loader, tu_te_loader):
		self.test_static = None
		self.ext = ext
		self.head = head
		self.criterion = criterion
		self.optimizer = optimizer
		self.scheduler = scheduler

		self.su_tr_epoch_counter = 1
		self.tu_tr_epoch_counter = 1

		self.su_tr_loader = su_tr_loader
		self.su_te_loader = su_te_loader
		self.tu_tr_loader = tu_tr_loader
		self.tu_te_loader = tu_te_loader

		self.reset_su()
		self.reset_tu()
			
	def reset_su(self):
		self.su_tr_iter_counter = 1
		self.su_tr_loader_iterator = iter(self.su_tr_loader)

	def reset_tu(self):
		self.tu_tr_iter_counter = 1
		self.tu_tr_loader_iterator = iter(self.tu_tr_loader)

	def assign_test(self, function):
		self.test_static = function

	def test(self):
		model = nn.Sequential(self.ext, self.head)
		test_su = self.test_static(self.su_te_loader, model)
		test_tu = self.test_static(self.tu_te_loader, model)
		return (test_su + test_tu) / 2, test_su, test_tu

	def train_batch(self):
		su_tr_inputs, su_tr_labels = next(self.su_tr_loader_iterator)
		tu_tr_inputs, tu_tr_labels = next(self.tu_tr_loader_iterator)
		self.su_tr_iter_counter += 1
		self.tu_tr_iter_counter += 1

		us_tr_inputs = torch.cat((su_tr_inputs, tu_tr_inputs))
		us_tr_labels = torch.cat((su_tr_labels, tu_tr_labels))
		us_tr_inputs, us_tr_labels = us_tr_inputs.cuda(), us_tr_labels.cuda()

		self.optimizer.zero_grad()
		outputs = self.ext(us_tr_inputs)
		outputs = self.head(outputs)
		loss = self.criterion(outputs, us_tr_labels)
		loss.backward()
		self.optimizer.step()
		
		if self.su_tr_iter_counter > len(self.su_tr_loader):
			self.su_tr_epoch_counter += 1
			self.reset_su()
		if self.tu_tr_iter_counter > len(self.tu_tr_loader):
			self.tu_tr_epoch_counter += 1
			self.reset_tu()
