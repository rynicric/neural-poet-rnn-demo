from torch import nn


class Poet(nn.Module):
    ''' Poet Model.

    Following the development of natural language processing, using word
    vectors and incorporating LSTM to CharRNN. By collecting tens of thousands
    of Tang poems, you can train a small network that can write poems.
    '''
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        seq_len, batch_size = x.size()

        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size,
                             self.hidden_dim).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size,
                             self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden

        # x.shape: [seq_len, batch_size]
        x = self.embeddings(x)
        # x.shape: [seq_len, batch_size, embeding_dim]
        x, hidden = self.lstm(x, (h_0, c_0))
        # x.shape: [seq_len, batch_size, hidden_dim]
        x = self.linear(x.view(seq_len * batch_size, -1))
        # x.shape: [seq_len * batch_size, vocab_size]

        return x, hidden
