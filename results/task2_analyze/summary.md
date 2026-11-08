metric: `val_bpb` (lower is better)

```
              n_seeds  val_bpb_mean  val_bpb_std  params_M  train_min  delta_vs_gpt_textbook  paired_p
name                                                                                                  
gpt_modern          2        2.1159       0.0010    3.1626     3.1627                -0.0694    0.0038
lstm                2        2.1372       0.0046    1.0698     1.1336                -0.0481    0.0572
gru                 2        2.1597       0.0018    0.8067     4.6042                -0.0256    0.0047
lstm_512            2        2.1701       0.0083    4.2368     2.5701                -0.0152    0.1947
gpt_textbook        2        2.1852       0.0016    3.4383     2.3367                 0.0000       NaN
rnn                 2        2.4367       0.0063    0.2803     2.7473                 0.2515    0.0085
```
