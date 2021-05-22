Neural Poet RNN Demo
====================

> Fake it till you make it.

This is the personal implementation of the Neural Poet Case of the book<br>
_[PyTorch: Introduction and Practice][chenyuntc]_ by Yun Chen.

[chenyuntc]: https://github.com/chenyuntc/pytorch-book

Thanks to the author for using Chinese poetry as a breakthrough point to lead
us into the fantastical circle of natural language. This case introduces the
knowledge of natural language processing and deformed CharRNN. By collecting
tens of thousands of Tang or Song poems, we can train a small network
that can write poems.


Installation
---------------------------------------

We provide instructions on how to install dependencies via pip.<br>
First, it is recommended for you to create a new environment:

```sh
conda create -n POET python=3.6.9
conda activate POET
```

Then, clone the repo and build the environment.

```sh
git clone https://github.com/rynicric/neural-poet-rnn-demo.git
pip install -r requirements.txt
```

Hint, please modify and install a suitable version of Pytorch
for your device as appropriate.


Data Preparation
----------------------------------------

The data for this repo are from the project called [chinese-poetry][cpoetry],
and<br>thanks to the team for providing massive data support.

[cpoetry]: https://github.com/chinese-poetry/chinese-poetry

Please download the [dataset][cpoetry-data] and organize them as follows.<br>
Of course, you can also directly use the processed and adapted<br>
data `poet.tang.npz` and `poet.song.np` provided in this repo:

[cpoetry-data]: https://github.com/chinese-poetry/chinese-poetry/tree/master/json

```txt
neural-poet-rnn-demo/
├── checkpoints/
├── data/
│   ├── chinese-poetry/
│   │   ├── json/
│   │   │   ├── poet.song.0.json
│   │   │   ├── poet.song.10000.json
│   │   ├── poet.song.npz
│   │   ├── poet.tang.npz
├── utils/
...
```


Training
----------------------------------------

To train the baseline Peot for the Tang poems on a single GPU, run:

```sh
python -m visdom.server &
python -m main train -config="opt.tang.yaml"
```

The default options are supported by the parameter `-config`, which can be<br>
modified as needed or provided directly on the command line as a param.

For example, if you want to use Song poems as training samples, run:

```sh
python -m visdom.server &
python -m main train -config="opt.tang.yaml" \
    -category="poet.song" \
    -checkpoints="checkpoints/song"
```


Inference
----------------------------------------

We provide baseline Poet models trained on Tang or Song poems. You can<br>
take a nap and utilize the provided models and processed data directly<br>
to complete your creation instead of training from scratch.

Please jump to [releases][releases] to download the models and dataset.

[releases]: https://github.com/rynicric/neural-poet-rnn-demo/releases

Hint, the param `-prefix_words` is not an integral part of the poem and<br>
is used to control the emotions (意境) of the generated poem.

```sh
python -m main inference -config="opt.tang.yaml" \
    -model-path="checkpoints/tang/Poet_199.pth" \
    -start_words="深度學習" \
    -prefix_words="海底魚兮天上鳥，高可射兮低可釣。" \
    -comp_maxwds=32
```

> 深度學習而不測，彼此悟之憎不足。我有我性不可奈，何物不兮勿相爲？

```sh
python -m main inference -config="opt.tang.yaml" \
    -model-path="checkpoints/tang/Poet_199.pth" \
    -start_words="春江花月夜，與君共長眠。" \
    -prefix_words="江流天地外，山色有無中。" \
    -comp_maxwds=48
```

> 春江花月夜，與君共長眠。人間有美女，采采復何嗟。<br>
> 流落人不見，暮鶯鳴且鮮。客心隨日度，春夢入樓前。


```sh
python -m main inference -config="opt.tang.yaml" \
    -category="poet.song" \
    -model-path="checkpoints/song/Poet_199.pth" \
    -start_words="深度學習" \
    -prefix-words="江流天地外，山色有無中。" \
    -acrostic=True
```

> 深谷無人到，幽居有此翁。<br>
> 度溪疑有水，漱石不潺湲。<br>
> 學道無人到，詩情有道逢。<br>
> 習池供茗飲，清興入詩筒。

Would u wanna compose?

🥳
