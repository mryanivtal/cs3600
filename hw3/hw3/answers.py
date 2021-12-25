r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 256
    hypers['seq_len'] = 64
    hypers['h_dim'] = 1024
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.25
    hypers['learn_rate'] = 0.001
    hypers['lr_sched_factor'] = 0.5
    hypers['lr_sched_patience'] = 2
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    temperature = 0.3
    start_seq = 'First Citizen: What hath done, he cannot help being undone'
    # ========================
    return start_seq, temperature



part1_q1 = r"""

We split the corpus into sequences instead of training on the whole text in order to avoid overfitting. If we use the whole text, during training model will learn to memorize it instead of learning to generalize.


"""

part1_q2 = r"""

It is possible that the generated text shows memory longer than the sequence length because hidden state doesn't depend on sequence length.

"""

part1_q3 = r"""

We are not shuffling the order of batches when training because we aim to pass relevant hidden state across bacthes. This way the model can learn from the order of original text.

part1_q4 = r"""
**Your answer:**
"""
# ==============

