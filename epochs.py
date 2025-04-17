from tqdm import tqdm
from utils import *
from fft_utils import historgram_match

def train_epoch(model, optimizer, criterion, dataloader, scheduler=None, device='cuda', verbose=True, metric_hook=None, **kwargs):
    model.train()
    losses, total = 0, 0
    data_iter = dataloader if not verbose else tqdm(
        dataloader, leave=False, desc='Train', dynamic_ncols=True
    )
    timer = Timer()
    for sample in data_iter:
        input, target = sample['data'], sample['label']

        if device != 'cpu':
            input = torch.from_numpy(np.array(input))
            input_his_aug, source_data, target_data = historgram_match(input)
            input, target = input.to(device), target.to(device)
            input_his_aug = input_his_aug.to(device)

        read_time = timer.update()
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        output_his_aug = model(input_his_aug)
        loss_his_aug = criterion(output_his_aug, target)
        loss_his_aug.backward()
        optimizer.step()
        batch_time = timer.update()

        if metric_hook is not None:
            metric_hook(output, target)

        losses += loss.item() * target.shape[0]
        total += target.shape[0]
        losses += loss_his_aug.item() * target.shape[0]
        total += target.shape[0]

        if scheduler is not None:
            scheduler.step()

        if verbose:
            data_iter.set_postfix(loss=f'{loss.item():.5f}',
                                  rt=f'{read_time:.3f}',
                                  bt=f'{batch_time:.3f}')
    if verbose:
        data_iter.close()
    return losses / total


def val_epoch(model, criterion, dataloader, device='cuda', verbose=True, metric_hook=None):
    model.eval()
    losses, total = 0, 0
    data_iter = dataloader if not verbose else tqdm(
        dataloader, leave=False, desc='Validate', dynamic_ncols=True)
    for sample in data_iter:
        input, target = sample['data'], sample['label']
        if device != 'cpu':
            input, target = input.to(device), target.to(device)
        output = model(input)
        loss = criterion(output, target)
        losses += loss.item() * target.shape[0]
        total += target.shape[0]
        if metric_hook is not None:
            metric_hook(output, target)
        if verbose:
            data_iter.set_postfix(loss=f'{loss.item():.5f}')
    if verbose:
        data_iter.close()
    return losses / total


def test_epoch(model, dataloader, device='cuda', verbose=True, metric_hook=None):
    model.eval()
    data_iter = dataloader if not verbose else tqdm(
        dataloader, leave=False, desc='Test', dynamic_ncols=True)
    for sample in data_iter:
        input, target = sample['data'], sample['label']
        if device != 'cpu':
            input, target = input.to(device), target.to(device)
        output = model(input)
        if metric_hook is not None:
            metric_hook(output, target)
    if verbose:
        data_iter.close()

