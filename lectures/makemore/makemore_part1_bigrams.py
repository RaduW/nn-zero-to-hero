import marimo

__generated_with = "0.11.8"
app = marimo.App(width="medium")


@app.cell
def imports():
    import matplotlib.pyplot as plt
    import marimo as mo
    from itertools import islice

    import torch
    import torch.nn.functional as F
    return F, islice, mo, plt, torch


@app.cell
def load_words():
    words = open("names.txt","r").read().splitlines()
    letters = ['.']+[chr(ch) for ch in range(ord('a'),ord('z')+1)]

    itos = {idx:s for idx,s in enumerate(letters)}
    stoi = {s:idx for idx,s in enumerate(letters)}
    return itos, letters, stoi, words


@app.cell
def load_frequencies(stoi, torch, words):
    def load_frequencies(words: list[str])->torch.Tensor:
        n = torch.zeros((28,28), dtype=torch.int32)
        for bigram in create_bigrams(words):
            idx0 = stoi[bigram[0]]
            idx1 = stoi[bigram[1]]
            n[idx0,idx1] += 1
        return n

    def create_bigrams(words: list[str]):
        for word in words:
            w0 = '.' + word
            w1 = word + '.'
            digrams = zip(w0,w1 )
            yield from digrams

    def frequencies_to_probabilities(freqs:torch.Tensor)->torch.Tensor:
        float_t = freqs.to(torch.float32)
        return float_t / float_t.sum(1, keepdim=True)

    N = load_frequencies(words)
    P = frequencies_to_probabilities(N)
    return N, P, create_bigrams, frequencies_to_probabilities, load_frequencies


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Display Frequencies

        ## Display as image
        """
    )
    return


@app.cell
def simple_display_digrams(N, plt):
    _fig, _ax = plt.subplots()
    _ax.imshow(N)
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Nice display""")
    return


@app.cell(hide_code=True)
def display_digrams(N, itos, plt, pytorch):
    def display_digrams( n:pytorch.Tensor):
        _fig, _ax = plt.subplots(figsize=(16, 16))
        _ax.imshow(n, cmap='Blues')
        for i in range(27):
            for j in range(27):
                chstr = itos[i] + itos[j]
                plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
                plt.text(j, i, n[i, j].item(), ha="center", va="top", color='gray')
        _ax.axis('off')
        return _ax

    display_digrams(N)
    return (display_digrams,)


@app.cell(hide_code=True)
def generate_name(P, itos, torch):
    _g = torch.Generator().manual_seed(2147483647)
    def generate_name():
        result= ""
        idx = 0
        while True:
            result= result + itos[idx]
            row = P[idx]
            idx = torch.multinomial(row, num_samples=1, replacement=True, generator=_g).item()
            if idx == 0 :
                break
        return result[1:]


    for _i in range(10):
        print(generate_name())
    return (generate_name,)


@app.cell(hide_code=True)
def generate_random_name(itos, torch):
    def generate_random_name():
        result= ""
        idx = 0
        while True:
            result= result + itos[idx]
            idx = torch.randint(0,len(itos),(1,)).item()
            if idx == 0 :
                break
        return result[1:]

    for _i in range(10):
        print(f"'{generate_random_name()}'")
    return (generate_random_name,)


@app.cell
def likelihood(P, stoi, torch, words):
    # display probabilities of the bigrams
    def log_likelihood( vocabulary):
        one_word = True if len(vocabulary) == 1 else 0
        ll = 0
        count=0
        for word in vocabulary:
            w0 = '.' + word
            w1 = word + '.'
            digrams = zip(w0,w1 )
            for d in digrams:
                ix1= stoi[d[0]]
                ix2 =stoi[d[1]]
                prob = P[ix1,ix2]
                log_prob = torch.log(prob)
                ll += log_prob
                count +=1
                if one_word:
                    print(f"{d[0]}{d[1]} {prob:.4f} {log_prob:.4f}")
        print(f"Negative log likelihood {-ll/count}")
    log_likelihood(words)
    log_likelihood(["andrej"])
    return (log_likelihood,)


@app.cell
def _(mo):
    mo.md(r"""# Neural Network Digram LLM""")
    return


@app.cell
def smooth_model(N, frequencies_to_probabilities):
    # smooth the probability distribution
    Ps = frequencies_to_probabilities(N+1)
    return (Ps,)


@app.cell
def _(mo):
    mo.md(r"""## Create the training set""")
    return


@app.cell
def create_training_set(create_bigrams, stoi, torch, words):
    #def create_training_set(words:list[string])->tuple(torch.tensor, torch.tensor):
    def create_training_set(words):
        xs = []
        ys = []
        for bigram in create_bigrams(words):
            idx1 = stoi[bigram[0]]
            idx2 = stoi[bigram[1]]
            xs.append(idx1)
            ys.append(idx2)
        return torch.tensor(xs), torch.tensor(ys)

    xs, yx = create_training_set(words)
    xs[:30],yx[:30]
    return create_training_set, xs, yx


@app.cell
def _(xs):
    repr(xs[:5])
    return


@app.cell(hide_code=True)
def one_hot_encoding(F, mo, xs):
    # create a one hot encoding for the first 5 examples
    _xenc = F.one_hot(xs[:5], num_classes=27)
    # it will create 5 rows of 27 

    mo.md(
        rf"""
        ```python
        >>> xs[:5]
        {xs[:5]}
        >>> xenc = F.one_hot(xs[:5], num_classes=27)
        {_xenc}
        ```
        """
    )
    return


@app.cell
def _(F, plt, xs):
    #let's see the 5 hot ecnoded examples as an image
    _xenc = F.one_hot(xs[:5], num_classes=27)
    plt.imshow(_xenc)
    return


@app.cell
def _(torch):
    # create the neurons weights 
    # we have 27 inputs ( 1 hot encoding of 27 values)
    # we have 27 outputs ( each output will become a probability distribution for the next token)
    _g = torch.Generator().manual_seed(2147483647)
    W = torch.rand((27,27), generator=_g)  
    return (W,)


@app.cell
def _(F, W, xs):
    # using one hot encoding you can effectively select a rows from a matrix
    # create one hot encoded vectors for the first two token ids
    _xenc = F.one_hot(xs[:2], num_classes=27).float()

    # select the rows for the first two examples
    _xenc @ W

    return


@app.cell
def _(mo):
    mo.md(
        rf"""
        ### Interpretation

        $W$ is a $27\times27$ matrix.

        `X = F.one_hot(xs[i])` gives us the a one hot vector for training sample`i` and:

        $$
        X @ W ~~~~~~~~(1)
        $$

        selects the $xs[i]$ th row of $W$.

        We interpret the row as the $log(count)$ of the next token.
        """
    )
    return


@app.cell
def _(W):
    W
    return


@app.cell
def _(F, W, torch):
    F.one_hot(torch.tensor([1,2]),num_classes=27).float()@W
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def torch_operations(mo, torch):
    z = torch.zeros((3,2), dtype=torch.int32)
    r =  torch.randint(0,10, (3,2), dtype=torch.int32)
    _xs = torch.arange(3)
    _ys = torch.tensor([0,1,0])

    mo.md(
        rf"""
        # PyTorch operations

        ## Tensor creation

        ```python
        >>> n = torch.zeros((3,2), dtype=torch.int32)
        {z}

        >>> r =  torch.randint(0,10, (3,2), dtype=torch.int32)
        {r}
        ```

        ## Get tensor elements

        ```python
        # first row
        >>> r[0]  # same as r[0,:]
        {r[0]}
        # first column
        >>> r[:,0]
        {r[:,0]}
        # element at row 0 , column 1
        >>> r[0,1]
        {r[0,1]}
        # pluck from various position
        >>> xs = torch.arange(2)
        {_xs}
        >>> ys = torch.tensor([0,1,0])
        >>> r[xs,ys]
        {r[_xs,_ys ]}
        ```

        ## Aggregation

        ```python
        # sum over a row, (add columns)
        >>> r.sum(1, keepdim=True)
        {r.sum(1,keepdim=True)}
        # sum over a colum, (add rows)
        >>> r.sum(0, keepdim=True)
        {r.sum(0, keepdim=True)}
        ```
        """
    )
    return r, z


@app.cell(hide_code=True)
def _(mo, torch):
    _g = torch.Generator().manual_seed(2147483647)
    _x = torch.rand(3)
    _p=_x/_x.sum()
    _ix = torch.multinomial(_p, num_samples=15, replacement=True, generator=_g)
    _ix
    mo.md(
        f"""
        ## Sample from distribution

        ```python
        # random numbers
        >>> _g = torch.Generator().manual_seed(2147483647) # create a rand generator for repetability
        >>> _x = torch.rand(3)
        {_x}
        # normalize to sum to 1 (create a distribution)
        >>> _p=_x/_x.sum()
        {_p}
        # check it does add up to 1 (approximately)
        >>> _p.sum()
        {_p.sum()}
        >>> _ix = torch.multinomial(_p, num_samples=15, replacement=True, generator=_g)
        >>> _ix
        {_ix}
        ```
        """
    )
    return


@app.cell
def _(itos):
    len(itos)
    return


@app.cell
def _(torch):
    torch.tensor([0,1,0
                ])
    return


@app.cell
def _(r, torch):
    _xs = torch.arange(3)
    _ys = torch.tensor([0,1,0])
    r[_xs,_ys]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
