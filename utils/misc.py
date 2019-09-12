import os

def write_to_txt(name, content):
	with open(name, 'w') as text_file:
		text_file.write(content)

def my_makedir(name):
	try:
		os.makedirs(name)
	except OSError:
		pass

def print_args(opt):
	for arg in vars(opt):
		print('%s %s' % (arg, getattr(opt, arg)))

def mean(ls):
	return sum(ls) / len(ls)

def print_nparams(model):
	nparams = sum([param.nelement() for param in model.parameters()])
	print('number of parameters: %d' % (nparams))
