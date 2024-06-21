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
        # print(decoder_hidden.size())
        # print(static_hidden.size())

        # decoder_hidden b,hidden_size,seq_len
        batch_size, hidden_size, _ = static_hidden.size()

        # hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)]
        hidden = decoder_hidden.transpose(1,2)
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

    #decoder_hidden 从 B,2,hidden_size 变成了 B,n,2,hidden_size
    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):
        # rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        decoder_hidden = decoder_hidden.reshape((decoder_hidden.size(0), -1, self.hidden_size))
        rnn_out, last_hh = self.gru(decoder_hidden, last_hh)
        rnn_out = rnn_out.squeeze(1)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh) 

        # Given a summary of the output, find an input context 根据rnn结果分配注意力
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out)
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # (B, 1, num_feats)

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)  # (B, num_feats, seq_len)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)
        # (B, 20+1)

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
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Encoder(static_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        # Used as a proxy initial state in the decoder when not specified 默认的decoder_input (batch_size, num_points, 2, 1)
        # self.x0 = torch.zeros((1, static_size, 1), requires_grad=True, device=device)

    def forward(self, static, dynamic, decoder_input=None, last_hh=None):
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

        if decoder_input is None:
            self.x0 = torch.arange(0, sequence_size,device=device)
            decoder_input = self.x0.unsqueeze(0).repeat(batch_size, 1)
            # self.x0 = torch.zeros((1, sequence_size, 2, 1), requires_grad=True, device=device)
            # decoder_input = self.x0.expand(batch_size, -1, -1, -1)

        # 现在默认的decoder_input (batch_size, num_points, 2, 1)
        # decoder_input = static.unsqueeze(3)

        # Always use a mask - if no function is provided, we don't update it
        mask = torch.ones(batch_size, sequence_size, device=device)

        # Structures for holding the output sequences
        tour_idx, tour_logp = [], []
        max_steps = 10 #sequence_size if self.mask_fn is None else 1000

        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        for _ in range(max_steps):

            if mask.byte().any():
                decoder_input, dynamic_hidden, last_hh, dynamic, tour_logp, tour_idx, mask = self.step(decoder_input, static_hidden, dynamic_hidden, last_hh, static, dynamic, tour_logp, tour_idx, mask)

#        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return decoder_input, tour_logp

    def step(self, decoder_input, static_hidden, dynamic_hidden, last_hh, static, dynamic, tour_logp, tour_idx, mask):
        # 扩展 indices(decoder_input:[256,21]) 以便在 dim=2 上进行 gather 操作
        batch_size, input_size, sequence_size = static.size()

        indices_expanded = decoder_input.unsqueeze(1).expand(-1, 2, -1)  # [256, 2, 21]

        # print(decoder_input)

        # 使用 gather 在 dim=2 上进行操作，结果形状为 [256, 2, 21]
        gathered = torch.gather(static, 2, indices_expanded)

        # # 调整形状为 [256, 21, 2]
        # decoder_input_value = gathered.permute(0, 2, 1)

        # ... but compute a hidden rep for each element added to sequence
        # decoder_input.size() B,2,1 这个记录了当前车的位置 经过encoder B,2,hidden_size
        # 现在的decoder_input B,n,2,1 记录每个点置换到哪里去的排序 经过encoder B,n,2,hidden_size
        # decoder_hidden = self.decoder(decoder_input_value.unsqueeze(3))
        # decoder_hidden = self.decoder(decoder_input_value.unsqueeze(3))
        decoder_input_value = gathered.permute(0, 2, 1).reshape((-1,2)).unsqueeze(2)
        decoder_hidden = self.decoder(decoder_input_value).view(batch_size,sequence_size, input_size,-1)

        probs, last_hh = self.pointer(static_hidden,
                                      dynamic_hidden,
                                      decoder_hidden, last_hh)

        # last_hh.size() 1,B,hidden
        # mask.size() B, 20+1 if masked then 0 else 1
        first_probs = F.softmax(probs + mask.log(), dim=1)

        # When training, sample the next step according to its probability.
        # During testing, we can take the greedy approach and choose highest
        if self.training:
            m = torch.distributions.Categorical(first_probs)

            # Sometimes an issue with Categorical & sampling on GPU; See:
            # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
            first_ptr = m.sample()
            while not torch.gather(mask, 1, first_ptr.data.unsqueeze(1)).byte().all():
                first_ptr = m.sample() # 检查下sample的有无被mask掉
            first_logp = m.log_prob(first_ptr)
        else:
            prob, first_ptr = torch.max(first_probs, 1)  # Greedy
            first_logp = prob.log()

        # After visiting a node update the dynamic representation
        # dynamic B,2,20+1 2:load demand
        # dynamic_hidden B,hidden_size,20+1

        # if self.update_fn is not None:
        #     dynamic = self.update_fn(dynamic, ptr.data)
        #     dynamic_hidden = self.dynamic_encoder(dynamic)
        #
        #     # Since we compute the VRP in minibatches, some tours may have
        #     # number of stops. We force the vehicles to remain at the depot
        #     # in these cases, and logp := 0
        #     is_done = dynamic[:, 1].sum(1).eq(0).float()
        #     logp = logp * (1. - is_done)

        # # And update the mask so we don't re-visit if we don't need to
        # if self.mask_fn is not None:
        #     mask = self.mask_fn(mask, dynamic, ptr.data).detach()
        # ptr 256
        # chosen_point = torch.gather(static, 2,
        #                              ptr.view(-1, 1, 1)
        #                              .expand(-1, input_size, 1)).detach()
        # 256,21,1

        mask[torch.arange(first_ptr.size(0)), first_ptr] = 0

        second_probs = F.softmax(probs + mask.log(), dim=1)

        if self.training:
            m = torch.distributions.Categorical(second_probs)

            # Sometimes an issue with Categorical & sampling on GPU; See:
            # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
            second_ptr = m.sample()
            while not torch.gather(mask, 1, second_ptr.data.unsqueeze(1)).byte().all():
                second_ptr = m.sample() # 检查下sample的有无被mask掉
            second_logp = m.log_prob(second_ptr)
        else:
            prob, second_ptr = torch.max(second_probs, 1)  # Greedy
            second_logp = prob.log()

        chosen_points = torch.stack((first_ptr, second_ptr), dim=1)
        # print(chosen_points)
        decoder_input = utils.apply_batch_permutation_pytorch(chosen_points,decoder_input)
        tour_logp.append((first_logp.unsqueeze(1)))
        tour_idx.append((first_ptr.data.unsqueeze(1),second_ptr.data.unsqueeze(1)))
        return decoder_input,dynamic_hidden,last_hh,dynamic,tour_logp,tour_idx,mask


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
