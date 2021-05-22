import os
import re
import json
import numpy as np


def _parse_raw_data(directory='data/chinese-poetry/json/',
                    category='poet.tang',
                    author=None,
                    constrain=None):
    ''' Processing source data files to get a list of the poetry content.

    If the author is provided, it only returns the poetry by this author.
    While constrain is set, searching only for sentences with special length.

    Args:
      - path (str, optional): the location of the json files
      - category (str, optional): limit the category of poem
      - author ([type], optional): limit the category of poem
      - constrain ([type], optional): limit the length of poem

    Returns:
      - corpus (list): a list of the poetry content

    Ref:
    https://github.com/justdark/pytorch-poetry-gen/blob/master/dataHandler.py
    '''
    def parse_sentence(para):
        rst, _ = re.subn(r'（[^（）]*）', '', para)
        rst, _ = re.subn(r'（[^（）]*）', '', rst)
        rst, _ = re.subn(r'（.*$', '', rst)
        rst, _ = re.subn(r'）', '', rst)

        rst, _ = re.subn(r'｛[^｛｝]*｝', '', rst)
        rst, _ = re.subn(r'{[^{}]*}', '', rst)
        rst, _ = re.subn(r'}', '', rst)

        rst, _ = re.subn(r'[［］\[\]《》/]', '', rst)
        rst, _ = re.subn(r'。。', '。', rst)

        rst = [i for i in rst if i not in set('1234567890-')]
        rst = ''.join(rst)
        return rst

    def handle_json(file):
        with open(file) as f:
            data = json.loads(f.read())

        for poetry in data:
            # Limit the author.
            if author is not None and poetry.get('author') != author:
                continue

            para = poetry.get('paragraphs')
            para = ''.join(para)
            para = parse_sentence(para)

            # Limit the length.
            if constrain is not None:
                sents = re.split(r'[，。！？]', para)
                sentslens = [len(s) for s in sents][:-1]

                if any(np.asarray(sentslens) - constrain):
                    continue

            if not para:
                continue

            yield para

    corpus = []
    file_list = os.listdir(directory)
    file_list = [f for f in file_list if f.endswith('.json')]
    file_list = [f for f in file_list if f.startswith(category)]
    for file in file_list:
        corpus.extend(handle_json(os.path.join(directory, file)))
    return corpus


def slice_sequence(seqs,
                   maxlen=None,
                   dtype='int32',
                   truncating='post',
                   padding='pre',
                   value=0.):
    ''' Slice each sequence to the same length for easy training.

    If maxlen is provided, any sequence longer than maxlen is truncated to
    maxlen. Truncation happens off either the end of the sequence (default) or
    the beginning. It also supports pre-padding (default) and post-padding
    when the sequence shorter than maxlen.

    Args:
      - seqs (list): list of lists where each element is a sequence
      - maxlen (int, optional): maximum length
      - dtype (str, optional): type to cast the resulting sequence
      - truncating (str, optional): 'pre' or 'post', remove values from sequence
            larger than maxlen either in the beginning or in the ending
      - padding (str, optional): 'pre' or 'post', pad either before or after each sequence
      - value ([type], optional): float, value to pad the sequences to the desired value

    Returns:
      - (numpy.ndarray): numpy array with dimension [num_of_sequences, maxlen]
    '''
    seqlen_lst = [len(s) for s in seqs]
    if maxlen is None:
        maxlen = np.max(seqlen_lst)

    rst = (np.ones((len(seqs), maxlen)) * value).astype(dtype)

    for idx, seq in enumerate(seqs):
        # For truncating.
        if truncating == 'pre':
            trunc = seq[-1 * maxlen:]
        elif truncating == 'post':
            trunc = seq[:maxlen]
        else:
            raise ValueError(f'Truncating type `{truncating}` not understood')
        # For padding.
        if padding == 'pre':
            rst[idx, -len(trunc):] = trunc
        elif padding == 'post':
            rst[idx, :len(trunc)] = trunc
        else:
            raise ValueError(f'Padding type `{padding}` not understood')
    return rst


def build_data(opt):
    ''' Create applicable data types and build text mapping relationships.

    Args:
      -  opt (NestedNamespace): the config arguments, including `data_path`,
            `category`, `author`, `constrain`, `maxlen`

    Returns:
      - data (numpy.ndarray): a poem set, with dimension [num_of_poem, words]
      - wrod2ix (function): mapping for character to encoding, word -> encode
      - ix2word (function): mapping for encoding to character, word -> encode
    '''

    # Try to extract from the cache, otherwise build from raw data.
    cache_path = os.path.join(opt.data_path, 'chinese-poetry', opt.category + '.npz')
    if os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True)
        data = cache['data']
        word2ix = cache['word2ix'].item()
        ix2word = cache['ix2word'].item()
        return data, word2ix, ix2word

    # Processing raw data.
    data = _parse_raw_data(category=opt.category,
                           author=opt.author,
                           constrain=opt.constrain)

    # Encode words.
    words = {wd for sent in data for wd in sent}
    word2ix = {wd: idx for idx, wd in enumerate(words)}
    word2ix['<EOP>'] = len(word2ix)  # end
    word2ix['<START>'] = len(word2ix)  # start
    word2ix['</s>'] = len(word2ix)  # space bar
    ix2word = {idx: wd for wd, idx in list(word2ix.items())}

    # Add start and end identifiers.
    for idx, datum in enumerate(data):
        data[idx] = ['<START>'] + list(datum) + ['<EOP>']

    # Data to encoding.
    data = [[word2ix[wd] for wd in sent] for sent in data]

    # Slice the same length for training.
    data = slice_sequence(data, maxlen=opt.maxlen, value=word2ix['</s>'])

    # Save cache.
    np.savez_compressed(cache_path, data=data, word2ix=word2ix, ix2word=ix2word)
    return data, word2ix, ix2word
