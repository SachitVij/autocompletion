# Predicting words using MakeMore


We have a dataset of name. Here, what we do is we convert that dataset into a new datasets defined as X and Y. Lets say one of the name in the dataset is emilly
So X and Y are defined as :
X takes the present character and previous 2 character and Y represent the next character.

| X          | Y(next character)   |
| ---------- | ------------------- |
| [0, 0, 0]  | 'e' (5)             |
| [0, 0, 5]  | 'm' (13)            |
| [0, 5, 13] | 'i' (9)             |
| [5, 13, 9] | 'l' (12)            |
.... so on


Next step is defining embedding matrix C which represent all the 27 character('.' and a-z) into 2D vectors.

Next steps is: C(X) which takes all the rows and identifying the characters in each rows with its corresponding 2d vector. The result is a (7,3,2) matrix like 
After passing through the neural network, the model outputs a **probability distribution over 27 characters** for each training sample. 

| sample            | Prob a | b     | e    | m    | i   | l    | y    | .    |
| ----------------- | ------ | ----- | ---- | ---- | --- | ---- | ---- | ---- |
| 1(predicting 'e') | 0.009  | 0.003 | 0.2  | 0.05 | 0.1 | 0.02 | 0.01 | 0.01 |
| 2(predicting 'm') | 0.02   | 0.006 | 0.05 | 0.3  | 0.1 | 0.02 | 0.03 | 0.02 |
| 3(predicting 'i') | 0.04   | 0.008 | 0.1  | 0.4  | 0.1 | 0.3  | 0.02 | 0.03 |
| 7(predicting'.')  | 0.005  | 0.005 | 0.05 | 0.05 | 0.1 | 0.1  | 0.02 | 0.7  |</br>
</br>
Each row represents the **probabilities assigned to each character** when making a prediction.</br>
prob[torch.arange(prob.shape[0]), Y] will extract/fetches the probability assigned to the correct target character, then we calculate loss and backpropagate to optimize the weights.

#### inspiration from https://github.com/karpathy/makemore 
