from torch import nn

class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)

def extractor_from_layer3(net):
	layers = [net.conv1, net.layer1, net.layer2, net.layer3, net.bn, net.relu]
	return nn.Sequential(*layers)

def linear_on_layer3(classes, width, pool):
	layers = [nn.AvgPool2d(pool), ViewFlatten(), nn.Linear(64 * width, classes)]
	return nn.Sequential(*layers)
