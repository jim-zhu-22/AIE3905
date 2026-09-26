import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from material_recognition.model import build_model
from material_recognition.train import run_epoch


class FrozenBackboneTest(unittest.TestCase):
    def test_head_training_preserves_batchnorm_and_updates_classifier(self):
        torch.set_num_threads(1)
        model = build_model(pretrained=False)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        before_bn = model.bn1.running_mean.clone()
        before_fc = model.fc.weight.detach().clone()
        data = TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
        optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.1)
        run_epoch(model, DataLoader(data, batch_size=2), nn.CrossEntropyLoss(), torch.device("cpu"), optimizer)
        self.assertTrue(torch.equal(before_bn, model.bn1.running_mean))
        self.assertFalse(torch.equal(before_fc, model.fc.weight.detach()))


if __name__ == "__main__":
    unittest.main()
