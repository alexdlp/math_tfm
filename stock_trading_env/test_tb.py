import torch
from torch.utils.tensorboard import SummaryWriter

# Setup
writer = SummaryWriter(log_dir='./tb_logs')

# Log example data
for n_iter in range(100):
    writer.add_scalar('Loss/train', 0.5 ** n_iter, n_iter)
    writer.add_scalar('Accuracy/train', 0.5 + 0.5 * n_iter, n_iter)

writer.close()