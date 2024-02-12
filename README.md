# MLP-V1

Rather than using PyTorch or library to create neural networks, I am building a rudimentary Neural Network here only using Addition, Subtraction, Multiplication, and Exponentiation. I decided to do this to understand larger models better at the lowest level, and to improve my explanatory skills to those who do not have a technical background.

---


The first step is to create a Class which holds the data for each neuron as well as a tuple that will hold the Children neurons which created it, as well as the type of operand to get there. 

After creating a graph using the operands, I used a back propogation function to calculate the Gradient, or the the local derivative between the nodes. This gradient tells us the amount of influence each weight has on the outcome, and if tweaked how much it will change. 

