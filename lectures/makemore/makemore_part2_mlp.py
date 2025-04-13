import marimo

__generated_with = "0.11.8"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt # for making figures
    return F, mo, plt, torch


@app.cell
def words():
    # read in all the words
    words = open('names.txt', 'r').read().splitlines()
    # build the vocabulary of characters and mappings to/from integers
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    return chars, itos, stoi, words


@app.cell
def dataset(stoi, torch, words):
    # build the dataset

    block_size = 3 # context length: how many characters do we take to predict the next one?
    X, Y = [], []
    for w in words:
  
      #print(w)
      context = [0] * block_size
      for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        #print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + [ix] # crop and append
  
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    # display the first 3 training samples
    #[[itos[y.item()] for y in x] for x in X[:3] ]
    print(X.shape, Y.shape)
    return X, Y, block_size, ch, context, ix, w


@app.cell
def embeddings(torch):
    #create a 2 dminesional space for our embeddings, 27 tokens (the letters + '.' ) and vector size 2 for each embedding
    C_=torch.randn((27,2))
    # get mutiple rows at a time
    C_[[2,2,3,4]]
    return (C_,)


@app.cell
def examples_embeddings(C_, X):
    emb_ = C_[X]
    emb_.shape
    return (emb_,)


@app.cell
def hidden_layer(torch):
    # the hidden layer
    # weights: 3 chars with embedding size 2 , to 100 neurons
    W1_ = torch.randn(2*3, 100)
    # biases: 100 neurons
    B1_ = torch.rand(100)
    return B1_, W1_


@app.cell
def _(mo):
    mo.md(rf"""

    We want to calculate $emb \times W_1+B_1$, but $emb$ has the wrong shape $[228146,3,2]$ and we want to flatten the last 2 dimenssions.

    """)
    return


@app.cell
def _(emb_, torch):
    # first try at creating input for hidden layer
    _x = torch.cat((emb_[:, 0,:],emb_[:, 1,:],emb_[:, 2,:]), 1)
    _x.shape
    return


@app.cell
def _(emb_, torch):
    # do it in a more generic way:
    _x = torch.cat(torch.unbind(emb_,1),1)
    _x.shape
    return


@app.cell
def _(B1_, W1_, emb_, torch):
    # do it in a more efficient way (just re-interpret the tensor shape)
    emb_.view(emb_.shape[0], 6).shape
    h_= torch.tanh(emb_.view(emb_.shape[0],6) @ W1_ + B1_)
    return (h_,)


@app.cell
def _(h_):
    h_.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The final layer

    The final layer weights $W_2$ will have the shape $100 \times 27$ since it has an input from $100$ and it has the output the size of the vocabulary, $27$. 

    The bias $B_2$ will have the shape $27$.
    """)
    return


@app.cell
def _(h_, torch):
    W2_ = torch.randn([100,27])
    B2_ = torch.randn(27)
    logits_ = h_ @ W2_ + B2_
    counts_ = logits_.exp()
    probs_ = counts_/counts_.sum(1, keepdim=True)
    probs_.shape
    return B2_, W2_, counts_, logits_, probs_


@app.cell
def _(Y, probs_, torch):
    # get the network assigned probability for our training sets
    _sample_probs = probs_[torch.arange(probs_.shape[0]),Y]
    loss_ = - _sample_probs.log().mean()
    loss_
    return (loss_,)


@app.cell
def _(mo):
    mo.md(r"""## Clean Version""")
    return


@app.cell
def _(block_size, stoi, torch, words):
    # build the dataset
    #block_size = 3 # context length: how many characters do we take to predict the next one?

    def build_dataset(words):  
      X, Y = [], []
      for w in words:

        #print(w)
        context = [0] * block_size
        for ch in w + '.':
          ix = stoi[ch]
          X.append(context)
          Y.append(ix)
          #print(''.join(itos[i] for i in context), '--->', itos[ix])
          context = context[1:] + [ix] # crop and append

      X = torch.tensor(X)
      Y = torch.tensor(Y)
      print(X.shape, Y.shape)
      return X, Y

    import random
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))

    Xtr, Ytr = build_dataset(words[:n1])
    Xdev, Ydev = build_dataset(words[n1:n2])
    Xte, Yte = build_dataset(words[n2:])


    return Xdev, Xte, Xtr, Ydev, Yte, Ytr, build_dataset, n1, n2, random


@app.cell
def _(torch):
    g = torch.Generator().manual_seed(2147483647) # for reproducibility
    C = torch.randn((27, 2), generator=g)
    W1 = torch.randn((6, 100), generator=g)
    b1 = torch.randn(100, generator=g)
    W2 = torch.randn((100, 27), generator=g)
    b2 = torch.randn(27, generator=g)
    parameters = [C, W1, b1, W2, b2]

    for p in parameters:
        p.requires_grad = True
    return C, W1, W2, b1, b2, g, p, parameters


@app.cell
def _(C, F, W1, W2, X, Y, b1, b2, parameters, torch):
    def fit():
        emb = C[X]
        # forward pass
        h = torch.tanh( emb.view(emb.shape[0],6) @ W1 + b1) 
        logits = h @ W2 + b2
        # counts = logits.exp()
        # probs = counts / counts.sum(1, keepdim=True)
        # # get the network assigned probability for our training sets
        # loss = - probs[torch.arange(probs.shape[0]), Y].log().mean()
        # calculate loss directly
        loss= F.cross_entropy(logits,Y)
        print(loss.item())
        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            p.data += - 0.1 * p.grad

    for _ in range(10):
        fit()

    return (fit,)


@app.cell
def _():


    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
