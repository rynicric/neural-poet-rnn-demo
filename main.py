import os
import visdom
import torch
import torchnet
import numpy as np
from tqdm import tqdm
from utils.data import build_data
from utils.model import Poet as OurModel
from utils.misc import config_hook
from utils.misc import keep_seed
from utils.misc import load_model_cache


def _composer(model, start_words, comp_maxwds, ix2word, word2ix, prefix_words=None):
    ''' Compose a complete poem base on start_words.

    The creation will be based on the mood of the given prefix_words which is
    not a part of the generated poem.

    Args:
      - model (torch.nn.Module): model for peom composer
      - start_words (str): the beginning of the poem
      - comp_maxwds (int): maximum length for poetry creation
      - ix2word (dict): mapping for encoding to character, encode -> word
      - word2ix (dict): mapping for character to encoding, word -> encode
      - prefix_words (str): used to control the mood only

    Returns:
      - (list): generated verse
    '''
    # Set device.
    device = next(model.parameters()).device

    rst = list(start_words)
    start_words_len = len(start_words)
    # Set the first character as `<START>`.
    input_ = torch.Tensor([word2ix['<START>']]).view(1, 1).long().to(device)
    hidden = None

    if prefix_words:
        for wd in prefix_words:
            output, hidden = model(input_, hidden)
            input_ = input_.data.new([word2ix[wd]]).view(1, 1)

    for i in range(comp_maxwds):
        output, hidden = model(input_, hidden)

        if i < start_words_len:
            wd = rst[i]
            input_ = input_.data.new([word2ix[wd]]).view(1, 1)
        else:
            top_idx = output.data[0].topk(1)[1][0].item()
            wd = ix2word[top_idx]
            rst.append(wd)
            input_ = input_.data.new([top_idx]).view(1, 1)
        if wd == '<EOP>':
            del rst[-1]
            break

    return rst


def _composer_acro(model, start_words, comp_maxwds, ix2word, word2ix, prefix_words=None):
    ''' Compose a acrostic poem base on start_words.

    Each word in start_words is now the first word in each sentence. The
    creation will be based on the mood of the given prefix_words which is not
    a part of the generated poem.

    Args:
      - model (torch.nn.Module): model for peom composer
      - start_words (str): the first word at the begining of every sentences
      - comp_maxwds (int): maximum length for poetry creation
      - ix2word (dict): mapping for encoding to character, encode -> word
      - word2ix (dict): mapping for character to encoding, word -> encode
      - prefix_words (str): used to control the mood only

    Returns:
      - (list): generated verse
    '''
    # Set device.
    device = next(model.parameters()).device

    # Set the first character as `<START>`.
    input_ = torch.Tensor([word2ix['<START>']]).view(1, 1).long().to(device)
    hidden = None

    if prefix_words:
        for wd in prefix_words:
            output, hidden = model(input_, hidden)
            input_ = input_.data.new([word2ix[wd]]).view(1, 1)

    rst = []
    pre_word = '<START>'
    start_words_iter = iter(list(start_words))
    for _ in range(comp_maxwds):
        output, hidden = model(input_, hidden)

        top_idx = output.data[0].topk(1)[1][0].item()
        wd = ix2word[top_idx]
        # Insert one element of `start_words`.
        if pre_word in {u'。', u'！', '<START>'}:
            try:
                wd = next(start_words_iter)
                input_ = input_.data.new([word2ix[wd]]).view(1, 1)
            except StopIteration:
                break
        else:
            input_ = input_.data.new([word2ix[wd]]).view(1, 1)
        rst.append(wd)
        pre_word = wd

    return rst


def train(config, **kwargs):
    ''' Provide command line interface to train a poetry model. '''

    # Set config, device, visualization, and specify random seed.
    opt = config_hook(config, **kwargs)
    keep_seed(opt.seed)
    panels = visdom.Visdom(env=opt.vis_env)
    device = (torch.device(opt.device) \
              if torch.cuda.is_available() else torch.device('cpu'))

    if not os.path.exists(opt.checkpoints):
        os.makedirs(opt.checkpoints)

    # Handle data.
    data, word2ix, ix2word = build_data(opt)
    data = torch.from_numpy(data)
    dataloader = torch.utils.data.DataLoader(data,
                                             shuffle=True,
                                             batch_size=opt.batch_size,
                                             num_workers=opt.num_workers)

    # Handle model, optimizer, and criterion.
    model = OurModel(len(word2ix), 128, 256)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = torch.nn.CrossEntropyLoss()

    if opt.model_cache:
        model_cache = torch.load(opt.model_cache)
        state_dict = load_model_cache(model.state_dict(), model_cache)
        model.load_state_dict(state_dict)
    model.to(device)

    # Training.
    loss_meter = torchnet.meter.AverageValueMeter()
    for epoch in range(opt.epoch):
        loss_meter.reset()
        for idx, data_ in tqdm(enumerate(dataloader), desc='Training'):
            data_ = data_.long().transpose(1, 0).contiguous().to(device)
            input_, target = data_[:-1, :], data_[1:, :]

            optimizer.zero_grad()
            output, _ = model(input_)

            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())

            # Visualization
            if idx % opt.vis_frequency == 0:
                # Loss.
                panels.line(Y=np.array([loss_meter.value()[0]]), \
                            X=np.array([epoch * len(dataloader) + idx]),
                            win='trian_loss',
                            update=(None if epoch * len(dataloader) + idx == 0 else 'append'))

                examples = [[ix2word[w] for w in data_[:, i].tolist()] \
                    for i in range(data_.shape[1])][:4]
                textboard = ('</br>' * 2).join([''.join(e) for e in examples])
                panels.text(textboard, win='origin_poem_examples')

                comps = []
                for word in list(opt.start_words):
                    comp = _composer(model, word, opt.comp_maxwds, ix2word, word2ix)
                    comps.append(comp)
                textboard = ('</br>' * 2).join([''.join(e) for e in comps])
                panels.text(textboard, win='composer_poem_examples')

        if not opt.backupoints or epoch in tuple(opt.backupoints):
            saving_name = f'{model.__class__.__name__}_{epoch}.pth'
            saving_name = os.path.join(opt.checkpoints, saving_name)
            torch.save(model.state_dict(), saving_name)


def inference(config, **kwargs):
    ''' Provide command line interface to compose a corresponding poem. '''

    # Set config, device.
    opt = config_hook(config, **kwargs)
    device = (torch.device(opt.device) \
              if torch.cuda.is_available() else torch.device('cpu'))

    # Handle data.
    _, word2ix, ix2word = build_data(opt)

    # Handle model, optimizer, and criterion.
    model = OurModel(len(word2ix), 128, 256)
    state_dict = torch.load(opt.model_path, map_location=(lambda s, l: s))
    model.load_state_dict(state_dict)
    model.to(device)

    if opt.start_words.isprintable():
        start_words = opt.start_words
        prefix_words = opt.prefix_words if opt.prefix_words else None
    else:
        start_words = opt.start_words.encode('ascii', 'surrogateescape') \
            .decode('utf8')
        prefix_words = opt.prefix_words.encode('ascii', 'surrogateescape') \
            .decode('utf8') if opt.prefix_words else None

    start_words = start_words.replace(',', u'，').replace('.', u'。')
    start_words = start_words.replace('?', u'？').replace('!', u'！')

    comper = _composer_acro if opt.acrostic else _composer
    rst = comper(model, start_words, opt.comp_maxwds, ix2word, word2ix, prefix_words)
    return ''.join(rst)


if __name__ == '__main__':
    import fire
    fire.Fire()
