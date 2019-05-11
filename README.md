# Q-Trader

** Use in your own risk **

Pytorch implmentation from q-trader(https://github.com/edwardhdlu/q-trader)

## Results

Some examples of results on test sets:

![HSI2018](images/%5EHSI_2018.png)
Starting Capital: $100,000
HSI, 2017-2018. Profit of $10702.13.

## Running the Code

To train the model, download a training and test csv files from [Yahoo! Finance](https://ca.finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC) into `data/`
```
mkdir models
python train ^GSPC 10 1000
```

Then when training finishes (minimum 200 episodes for results):
```
jupyter notebook -> visualize.ipynb
```

## References

[Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/) - Q-learning overview and Agent skeleton code
