# Q-Trader

> ⚠️ **Use at your own risk.** This is an educational reinforcement-learning
> experiment, not financial advice or a production trading system.

A PyTorch implementation of a Deep Q-Network (DQN) trading agent, ported from
[edwardhdlu/q-trader](https://github.com/edwardhdlu/q-trader) (originally Keras).

The agent observes a sliding window of recent price changes and learns a policy
over three actions — **sit**, **buy**, and **sell** — using Q-learning with a
target network and experience replay.

## Results

Some examples of results on test sets:

![HSI2018](images/%5EHSI_2018.png)

Starting capital: $100,000. HSI, 2017–2018. Profit of $10,702.13.

## How It Works

- **State** — an `n`-day window of normalized price changes (`functions.getState`),
  where each value is the sigmoid of the day-over-day close difference.
- **Actions** — `0` sit, `1` buy, `2` sell.
- **Reward** — realized profit when a held position is sold (clamped at 0).
- **Model** — a small fully-connected network (`agent/model.py`) trained with
  RMSprop and a Huber (smooth L1) loss.
- **Training loop** — DQN with a separate target network synced every 10 episodes
  and a replay buffer of 10,000 transitions (`agent/agent.py`).

## Project Structure

```
agent/
  agent.py     # DQN agent: action selection + optimization step
  model.py     # the Q-network
  memory.py    # replay buffer (Transition namedtuple + ReplayMemory)
functions.py   # data loading, state representation, helpers
train.py       # training entry point
evaluate.py    # evaluation / plotting
visualize.ipynb# notebook to inspect training results
data/          # input price CSVs (Yahoo! Finance format)
images/        # result charts
```

## Setup

```
pip install -r requirements.txt
```

## Running the Code

Download training and test CSV files from
[Yahoo! Finance](https://ca.finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC)
into `data/`. The loader reads the `Close` column (column index 4), so keep the
default Yahoo Finance CSV format.

Train the model — arguments are `[stock] [window] [episodes]`:

```
mkdir -p models
python train.py ^HSI 10 1000
```

Checkpoints are written to `models/` every 10 episodes. Allow at least ~200
episodes before the results become meaningful.

When training finishes, inspect the results in the notebook:

```
jupyter notebook   # then open visualize.ipynb
```

`evaluate.py` runs the trained agent against `^HSI_2018` and plots buys/sells
against the price series.

## References

- [Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/) —
  Q-learning overview and the original agent skeleton.
- [edwardhdlu/q-trader](https://github.com/edwardhdlu/q-trader) — the Keras
  implementation this project is based on.
