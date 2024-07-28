import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size),
                                          device=device, requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):

        batch_size, hidden_size, _ = static_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)  # (batch, seq_len)
        return attns


class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
                                          device=device, requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.encoder_attn = Attention(hidden_size)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):
        # if last_hh is not None:
        #     last_hh = last_hh.detach()
        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        
        rnn_out = rnn_out.squeeze(1)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh) 

        # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out)
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # (B, 1, num_feats)

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)  # (B, num_feats, seq_len)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)

        return probs, last_hh


class DRL4TSP(nn.Module):
    """Defines the main Encoder, Decoder, and Pointer combinatorial models.

    Parameters
    ----------
    static_size: int
        Defines how many features are in the static elements of the model
        (e.g. 2 for (x, y) coordinates)
    dynamic_size: int > 1
        Defines how many features are in the dynamic elements of the model
        (e.g. 2 for the VRP which has (load, demand) attributes. The TSP doesn't
        have dynamic elements, but to ensure compatility with other optimization
        problems, assume we just pass in a vector of zeros.
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    update_fn: function or None
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element.
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidlines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
    dropout: float
        Defines the dropout rate for the decoder
    """

    def __init__(self, static_size, dynamic_size, hidden_size,
                 update_fn=None, mask_fn=None, num_layers=1, dropout=0.):
        super(DRL4TSP, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.update_fn = update_fn
        self.mask_fn = mask_fn

        # Define the encoder & decoder models
        # EDIT: self.static_encoder = Encoder(static_size, hidden_size)
        # self.static_encoder = Encoder(static_size + 1, hidden_size)
        # self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        # self.decoder = Encoder(static_size + 1, hidden_size)
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Encoder(static_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        # EDIT: Used as a proxy initial state in the decoder when not specified
        # self.x0 = torch.zeros((1, static_size + 1, 1), requires_grad=True, device=device)
        self.x0 = torch.zeros((1, static_size, 1), requires_grad=True, device=device)

    def forward(self, static, dynamic, decoder_input=None, last_hh=None, indices = None, mask_first = None, mask_second = None, last_ptr = None, tour_logp = None):
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the VRP, this can be
            things like the (load, demand) of each city. If there are no dynamic
            elements, this can be set to None
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        """
        batch_size, input_size, sequence_size = static.size()
        if mask_first is None:
            mask_first = torch.ones(batch_size, sequence_size, device=device)
        if mask_second is None:
            mask_second = torch.ones(batch_size, sequence_size, device=device)
        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1, -1)
        # Always use a mask - if no function is provided, we don't update it
        # mask = torch.ones(batch_size, sequence_size, device=device)

        # Structures for holding the output sequences
        # tour_idx, tour_logp = [], []
        # max_steps = sequence_size if self.mask_fn is None else 1000
        max_steps = 1
        
        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        # EDIT: static_hidden = self.static_encoder(static)
        if True:
            if indices == None:
                indices = torch.arange(0, sequence_size, device=device).unsqueeze(0).repeat(batch_size, 1)
            # print(indices)
            # Should be edited
            # update_static_input = torch.concat((indices.unsqueeze(1),static),dim=1)
            # static_hidden = self.static_encoder(update_static_input)
            update_static_input = static
            static_hidden = self.static_encoder(update_static_input)
            dynamic_hidden = self.dynamic_encoder(dynamic)
            
            for _ in range(max_steps):
                decoder_hidden = self.decoder(decoder_input)
    
                probs, last_hh = self.pointer(static_hidden,
                                              dynamic_hidden,
                                              decoder_hidden, last_hh)

                first_probs = F.softmax(probs + mask_first.log(), dim=1)
                if self.training:
                    m = torch.distributions.Categorical(first_probs)
                    first_ptr = m.sample()
                    # print(f"second_ptr:{second_ptr}")
                    # print(mask_second.sum(dim=1)[0])
                    while not torch.gather(mask_first, 1, first_ptr.data.unsqueeze(1)).byte().all():
                        first_ptr = m.sample() # 检查下sample的有无被mask掉
                else:
                    prob, first_ptr = torch.max(first_probs, 1)  # Greedy
                first_logp = m.log_prob(first_ptr)
                mask_first[torch.arange(first_ptr.size(0)), first_ptr] = 0

                second_probs = F.softmax(probs + mask_second.log(), dim=1)
                if self.training:
                    m = torch.distributions.Categorical(second_probs)
                    second_ptr = m.sample()
                    # print(f"second_ptr:{second_ptr}")
                    # print(mask_second.sum(dim=1)[0])
                    while not torch.gather(mask_second, 1, second_ptr.data.unsqueeze(1)).byte().all():
                        second_ptr = m.sample() # 检查下sample的有无被mask掉
                else:
                    prob, second_ptr = torch.max(second_probs, 1)  # Greedy
                mask_second[torch.arange(second_ptr.size(0)), second_ptr] = 0
                # second_logp = F.log_softmax(probs,dim=1)[torch.arange(second_ptr.size(0)), second_ptr]
                second_logp = m.log_prob(second_ptr)
                chosen_points = torch.stack((first_ptr, second_ptr), dim=1)
                # print(chosen_points)
                indices = utils.apply_batch_permutation_pytorch(chosen_points, indices)
            logp = second_logp + first_logp
            tour_logp.append(logp.unsqueeze(1))
            decoder_input = torch.gather(update_static_input, 2,
                                         second_ptr.view(-1, 1, 1)
                                         .expand(-1, input_size, 1)).detach()

        return decoder_input, tour_logp, last_hh, mask_first, mask_second, second_ptr, indices

if __name__ == '__main__':
    raise Exception('Cannot be called from main')
