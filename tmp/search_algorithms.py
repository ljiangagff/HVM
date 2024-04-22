import torch
import torch.utils.data as tud

MAX_SEQ_LENGTH = 160
START_TOKEN = 1
END_TOKEN = 2
'''
Can use torch.ones to initialize
'''

def greedy_search_batch(
    model, 
    X, 
    predictions = MAX_SEQ_LENGTH+1,
):
    """
    Implements Greedy Search to compute the output with the sequences given in X. The method can compute 
    several outputs in parallel with the first dimension of X.

    Parameters
    ----------    
    X: LongTensor of shape (examples, length)
        The sequences to start the decoding process.

    predictions: int
        The number of tokens to append to X.

    progress_bar: bool
        Shows a tqdm progress bar, useful for tracking progress with large tensors.

    Returns
    -------
    Y: LongTensor of shape (examples, length + predictions)
        The output sequences.

    probabilities: FloatTensor of length examples
        The estimated log-probabilities for the output sequences. They are computed by iteratively adding the 
        probability of the next token at every step.
    """
    with torch.no_grad():
        device = X.device
        Y = torch.ones(X.shape[0], 1).to(device).long()
        # probs = torch.zeros(X.shape[0]).to(device)
        cache = None
        for i in range(predictions):
            logits, cache = model.greedy_forward(X, Y, cache)
            next_probs = logits[:, -1].log_softmax(-1)
            max_next_probs, next_tokens = next_probs.max(-1)
            next_tokens = next_tokens.unsqueeze(-1)
            Y = torch.cat((Y, next_tokens), axis = 1)
    return Y


def beam_search(
    model,
    X,
    MAX_LEN=64,
    beam_width=5,
):
    """
    Implements Beam Search to compute the output with the sequences given in X. The method can compute 
    several outputs in parallel with the first dimension of X.

    Parameters
    ----------    
    X: LongTensor of shape (examples, length)
        The sequences to start the decoding process.

    predictions: int
        The number of tokens to append to X.

    beam_width: int
        The number of candidates to keep in the search.

    Returns
    -------
    Y: LongTensor of shape (examples, length + predictions)
        The output sequences.

    probabilities: FloatTensor of length examples
        The estimated log-probabilities for the output sequences. They are computed by iteratively adding the 
        probability of the next token at every step.
    """
    with torch.no_grad():
        batch_size = X.shape[0]
        device = X.device
        Y = torch.ones(X.shape[0], 1).to(device).long()
        # Y = torch.ones(X.shape[0], 1).to(next(model.parameters()).device).long()
        # The next command can be a memory bottleneck, can be controlled with the batch
        # size of the predict method.
        next_probs = model.beam_forward(X, Y)[:, -1, :]
        vocab_size = next_probs.shape[-1]
        probs, next_tokens = next_probs.squeeze().log_softmax(-1)\
            .topk(k=beam_width, axis=-1)
        if batch_size == 1:
            probs = probs.unsqueeze(0)

        Y = Y.repeat((beam_width, 1))
        next_tokens = next_tokens.reshape(-1, 1)
        Y = torch.cat((Y, next_tokens), axis=-1)
        # This has to be minus one because we already produced a round
        # of predictions before the for loop.
        for _ in range(MAX_LEN-1):
            next_probs = []
            X_DATA = X.repeat((beam_width, 1, 1, 1)).transpose(0, 1)
            X_DATA = X_DATA.flatten(end_dim=1)
            dataset = tud.TensorDataset(X_DATA, Y)
            loader = tud.DataLoader(dataset, batch_size=5000)
            for x, y in iter(loader):
                next_prob = model.beam_forward(x, y)
                next_prob = next_prob[:, -1, :].log_softmax(-1)
                next_probs.append(next_prob)

            next_probs = torch.cat(next_probs, axis=0)
            next_probs = next_probs.reshape(
                (-1, beam_width, next_probs.shape[-1]))
            probs = probs.unsqueeze(-1) + next_probs
            probs = probs.flatten(start_dim=1)
            probs, idx = probs.topk(k=beam_width, axis=-1)
            next_tokens = torch.remainder(
                idx, vocab_size).flatten().unsqueeze(-1)
            candi = (idx / vocab_size).long()
            candi += torch.arange(Y.shape[0] // beam_width,
                                  device=X.device).unsqueeze(-1) * beam_width
            Y = Y[candi].flatten(end_dim=-2)
            Y = torch.cat((Y, next_tokens), axis=1)
        predictions = Y.reshape(-1, beam_width, Y.shape[-1])
        return predictions
        # return Y.reshape(-1, beam_width, Y.shape[-1]), probs

def greedy_search(model, X):
    """ Used for inference """
    device = X.device
    memory = X.permute(1, 0, 2)

    Y = (torch.LongTensor([START_TOKEN]).to(
        device).unsqueeze(1))  # 1, 1
    output = [START_TOKEN]
    cache = None
    # generation loop
    while len(output) < MAX_SEQ_LENGTH+1:  # max length a generation
        text_tokens, _ = model.text_encoder(Y)
        tgt = model.pos(text_tokens)
        decoded, cache = model.decoder(
            tgt,
            memory,
            cache,
        )

        logits = model.classification_layer(
            decoded[-1, :, :])  # 1, vocab_size
        new_token = logits.argmax(1).item()
        if new_token == END_TOKEN:  # end of generation
            output.append(END_TOKEN)
            break
        output.append(new_token)
        Y = torch.cat(
            [Y,
                torch.LongTensor([new_token]).unsqueeze(1).to(device),],
            dim=0,
        )  # current_output_len, 1
    return output
