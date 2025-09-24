import os
import torch
import shutil

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
	assert path is not None, f"Checkpoint save path should not be None type."
	os.makedirs(path, exist_ok=True)
	torch.save(state, os.path.join(path, filename))
	if is_best:
		shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'model_best.pth.tar'))