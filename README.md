# AI-Problem--Horizon-Detection-Using-Bayes-Net-Inference
Detecting horizons in images using inference on HMM using Viterbi algorithm

**CASE 1:**
For this, we want to solve Bayes net as given

    si* = argmax P(Si = Si | W1, . . .,Wm)
 
We want to find the row Si for which the above probability is maximim. From the Bayes net, we can assume that each column is independent of other column. Hence,

    P(S1 = row ridge | W1 ... Wm) = P(S1 = row ridge | W1)
    
    = P(W1 | s1)  * P(S1)
      -------------------
             P(W1)
             
 Every row is equally likely and hence P(S) is same for all rows. Alos, P(W) is same for all columns, so we can ignore it for agrmax calculation. Hence, for simple case, we are considering max gradient row at every column.
 
 ![alt text](https://github.iu.edu/cs-b551-fa2019/nibafna-nrpai-sjejurka-a2/blob/master/images/Blue.jpeg)
 
 **Case 2:**
 
Using the Viterbi algorithm to solve for the maximum a posterior estimate with below formula.

      arg max P(S1 = s1,...,Sm = sm|w1,...,wm)
      s1,...,sm

In this approach, we tried different emissions and transmission probabilties:

Emission probabilties:

We want to set emission probablity high when actual row of ridge is closer to strong edge according to gradient image. We tried the following: 

      1. col[index]/max(col)
      2. col[index]**0.90
      3. col[index]/sum(col)

From the above emission probabilties, our algorithm worked best with emission probability : **col[index]/sum(col)**

Transmission probabilties:

We want to set transmission probablities high to encourage smoothness, that is when corresponding rows are closer we set higher probablities. We tried the following:

    1. sqrt(r2-r1)**2 + (c2 - c1) **2 /rows
    2. (rows - abs(r2-r1)) ** 2
    3. if abs(r2 - r1) == 0 then 1 else 1/abs(r2-r1)
    
For the above transission probabilties, our algorithm worked best with transmission probabilty : 
**if abs(r2 - r1) == 0 then 1 else 1/abs(r2-r1)**

However, in this approach since we are starting with max gradient initial probability , it didnt worked accurately for all images.
To overcome this, we implemeneted third case.

![alt text](https://github.iu.edu/cs-b551-fa2019/nibafna-nrpai-sjejurka-a2/blob/master/images/Red.jpeg)

**CASE 3:**

In this case, we take feedback from human and ask them to input the row and column co-ordinate. We start with these co-ordiante, and set its probabiltiy as highest. We then move forward and backward columnwise.
![alt text](https://github.iu.edu/cs-b551-fa2019/nibafna-nrpai-sjejurka-a2/blob/master/images/green.jpeg)
