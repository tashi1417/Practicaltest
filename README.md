# Practicaltest


# mean of the dataframe
df['AI_Score'].mean()
df1 = df[['AI_Score', 'Math_Score']] # Mean of two columns
df1.mean()

from scipy import stats
print(stats.gmean([4,11,15,16,5,7]))
# Geometric Mean of the column in dataframe
stats.gmean(df.iloc[:,1:3],axis=0)
# Geometric mean of the specific column
stats.gmean(df.loc[:,"Stats_Score"])

print(stats.hmean([4,11,15,16,5,7]))
stats.hmean(df.iloc[:,1:3],axis=0)
stats.hmean(df.loc[:,"Stats_Score"])

import statistics as stats
stats.median([4,-11,-5,16,5,7,9])
df.iloc[:,1:4].median()
df.iloc[:,1:4].median()

import statistics as stats
stats.mode(['lion', 'cat', 'cat','dog','tiger'])
df.mode()

import pandas as pd
df = pd.DataFrame([(1, 2), (0, 3), (2, 0), (1, 1)], columns=['dogs', 'cats'])
df.cov()
df = pd.DataFrame(np.random.randn(1000, 5), columns=['A', 'B', 'C', 'D', 'E'])
df.cov()

df.corr(method='pearson') #Pairwise correlation of all columns in the dataframe
import seaborn as sns
sns.heatmap(df.corr(method='pearson'), annot = True)

print(1 in A) # to check the elements in the set
B.issubset(A)
universal = set(np.arange(10))
A.union(B)
A.intersection(B)
A.difference(B)
A_Compliment = universal.difference(A)

import numpy as np
import random

#Function for roll the Dice

def roll_the_dice(n_simulations = 1000):
  count = 0

  #Each iteration of the for loop is trial
  for i in range(n_simulations):

    #Roll each Die
    die1 = random.randint(1,7)
    die2 = random.randint(1,7)

    #Sum the values to get the score
    score = die1 + die2

    #decide if we should add it to the count
    if score % 2 == 0 or score > 7:
      count += 1
  return count/n_simulations

string = 'The probability of rolling an even number or greater than 7 is:'
print(string, np.round(roll_the_dice()*100, 2), '%')

import numpy as np
import random
 # Let's set up the dictionary that we will use for this question
 # This dictionary will allow us to randomly choose a color
d = {}
for i in range(61):
  if i < 10:
    d[i] = 'white'
  elif i > 9 and i < 30:
    d[i] = 'red'
  else:
    d[i] = 'green'

#Initialize important variables
n_simulations = 10000
part_A_total = 0
part_B_total = 0

for i in range(n_simulations):

  #make a list of the colors that we choose
  list = []
  for i in range(5):
    list.append(d[random.randint(0,59)])

  #convert it to a numpy
  list = np.array(list)

  #find the number of each that we picked
  white = sum(list == 'white')
  red = sum(list == 'red')
  green = sum(list == 'green')

  #Keep track if the combination met the above critria
  if white == 3 and red == 2:
    part_A_total += 1

  if red == 5 or white == 5 or green == 5:
    part_B_total +=1

print('The probability of 3 white and 2 red is: ', part_A_total/n_simulations*100, '%')
print('The probability of all the same color is: ', part_B_total/n_simulations*100, '%')

# permutations of given length
from itertools import permutations

# Get all permutations of length 2
perm = permutations([1,2,3], 2)

# Print the obtained permutations
for i in tuple(perm):
	print (i)

# combinations of given length
from itertools import combinations

# Get all combinations of [1, 2, 3]
# and length 2
comb = combinations([1, 2, 3], 2)

# Print the obtained combinations
for i in tuple(comb):
	print(i)

from itertools import combinations_with_replacement

# Get all combinations of [1, 2, 3] and length 2
comb = combinations_with_replacement([1, 2, 3], 2)

# Print the obtained combinations
for i in tuple(comb):
	print (i)

df['grade_A'] = np.where(df['G3']*5 >= 80, 1, 0)
df['high_absenses'] = np.where(df['absences'] >= 10, 1, 0)
df['count'] = 1
df = df[['grade_A','high_absenses','count']]
pd.pivot_table(df, values='count', index=['grade_A'], columns=['high_absenses'],
               aggfunc=np.size, fill_value=0)

import pandas as pd
import matplotlib.pyplot as plt
#Create dataframe
data = pd.DataFrame({'x':[0, 1, 2, 3], 'P(X = x)': [1/8, 3/8, 3/8, 1/8]})
#Visualize using bar plot
fig = plt.figure(figsize = (5,3))
plt.bar(data.x, data['P(X = x)'], color = 'maroon')
plt.title('Probability Distribution')
plt.xlabel('x')
plt.ylabel('P(X = x)')
plt.show()

import matplotlib.pyplot as plt
from scipy.stats import bernoulli
# Instance of Bernoulli distribution with parameter p = 0.7
bd = bernoulli(0.7)
# Outcome of experiment can take value as 0, 1
X = [0, 1]
# Create a bar plot; Note the usage of "pmf" function to determine the
#probability of different values of random variable
plt.figure(figsize=(5,3))
plt.bar(X, bd.pmf(X), color='green')
plt.title('Bernoulli Distribution (p=0.7)', fontsize='12')
plt.xlabel('Values of Random Variable X (0, 1)', fontsize='12')
plt.ylabel('Probability', fontsize='12')
plt.show()

from scipy.stats import binom
#Calculate first expected value or mean
E_X = binom.moment(1, 15, 0.18)
print('Expectation of X is:', round(E_X,2))
#calculate binomial probability
result_0 = binom.pmf(k=0, n=15, p=0.18)
result_1 = binom.pmf(k=1, n=15, p=0.18)
#Print the result
print("Binomial Probability when X = 0: ", round(result_0, 2),
      '\nBinomial Probability when X = 1:', round(result_1, 2))

required_prob = 1 - result_0 - result_1
print('The required probability: ', round(required_prob, 2))

from scipy.stats import binom
import matplotlib.pyplot as plt
import pandas as pd
#Setting the values of n and p
n, p = 15, 0.18
#Defining the list of x values
x = list(range(n+1))
#Create DataFrame which consist of x, pmf, and cdf
rv = binom(n, p)
df = pd.DataFrame({'x': x, 'pmfs': rv.pmf(x), 'cdfs': rv.cdf(x)})
df.head()
mean = binom.mean(n = 15, p = 0.18)
var = binom.var(n = 15, p = 0.18)
std = binom.std(n = 15, p = 0.18)
print('\nMean: ', round(mean, 2), '\nVariance: ', round(var, 2),
      '\nStandard deviation: ', round(std, 2))

from scipy.stats import geom
import matplotlib.pyplot as plt
p_6 = geom.pmf(k = 6, p = 0.04) #Mass function of geometric distribution
print('Prob. of first defective:', round(p_6, 3))

from scipy.stats import geom
import matplotlib.pyplot as plt
import pandas as pd
# X = Discrete random variable representing number of throws
# p = Probability of the perfect throw
#Create a DataFrame which consist of x, pmfs, and cdfs
x = list(range(1,11))
p = 0.6
df = pd.DataFrame({'x': x, 'pmfs':geom.pmf(x, p), 'cdfs': geom.cdf(x, p)})
df.head()
mean = geom.mean(p = 0.6)
var = geom.var(p = 0.6)
std = geom.std(p = 0.6)
print('\nMean: ', round(mean,2), '\nVariance: ', round(var,2),
      '\nStandard deviation: ', round(std,2))

from scipy.stats import poisson
#Generate random values from Poisson distribution with mean=3 and sample size=10
poisson.rvs(mu = 3, size = 10)

from scipy.stats import poisson
#calculate probability
prob = poisson.pmf(k=5, mu=3)
print('Required prob.: ', round(prob, 4))

from scipy.stats import poisson
#calculate probability
prob = poisson.cdf(k=4, mu=7)
print('Required prob.: ', round(prob, 4))

mean = poisson.mean(mu = 7)
var = poisson.var(mu = 7)
std = poisson.std(mu = 7)
print('\nMean: ', round(mean,2), '\nVariance: ', round(var,2),
      '\nStandard deviation: ', round(std,2))

#Using Sympy library
import sympy as smp
x = smp.Symbol('x')
p_x = smp.integrate(x, (x, 0, 0.5))
print('The probability is: ', round(p_x,3))

#Solution of part 1, to check f(x) is pdf, integral of f(x) should be equal to 1.
import sympy as smp
x = smp.Symbol('x')
fx = 3*x**(-4)
smp.integrate(fx, (x, 1, smp.oo))
#It is pdf since integral over the limit is 1.

#Solution of part 2, E(X)
import sympy as smp
x = smp.Symbol('x')
fx1 = x*3*x**(-4)
E_X = smp.integrate(fx1, (x, 1, smp.oo))
print('Expectation of X is:', E_X)
#Var(X) = E(X^2) - (E(X))^2
fx2 = (x**2)*(3)*(x**(-4))
E_X2 = smp.integrate(fx2, (x, 1, smp.oo))
print('Expectation of X^2 is:', E_X2)
Var_X = E_X2 - (E_X)**2
print('Variance of X is: ', Var_X)
#Standard deviation
std_X = (Var_X)**(0.5)
print('The standard deviation of X is: ', std_X)

#P(X < 3)
from scipy.stats import uniform
prob = uniform(loc = 0, scale = 10).cdf(3) - uniform(loc = 0, scale = 10).cdf(0)
print('P(X < 3):', prob)
#P(X > 7)
prob = uniform.cdf(x = 10, loc = 0,  scale = 10) - uniform.cdf(x = 7, loc = 0,
                                                               scale = 10)
print('P(X > 7):', round(prob, 3))
#P(1 < X < 6)
prob = uniform.cdf(x = 6, loc = 0,  scale = 10) - uniform.cdf(x = 1, loc = 0,
                                                              scale = 10)
print('P(1 < X < 6):', prob)

#Find P(17 < X < 19) if X is uniformly distributed
from scipy.stats import uniform
#'loc': the minimum point of the distribution
#'scale': is the range of the interval
prob = uniform(loc = 15, scale = 10).cdf(19) - uniform(loc = 15, scale = 10).cdf(17)
print('The probability that bird weighs between 17 and 19 grams is:', prob)

from scipy.stats import expon
#Calculate probability that x is less than 10 when mean rate is 5
expon.cdf(x = 10, scale = 5)
#Then the required probability is given by
prob = 1 - expon.cdf(x = 10, scale = 5)
print('The required probability is:', round(prob,3))

from scipy.stats import expon #Import required libraries
import pandas as pd
#Define x_values
x = np.linspace(1,20,1000)
exponential = expon(scale = 4)
#Create dataframe which consist of x, pdfs, and cdfs of uniform random variable.
df = pd.DataFrame({'x': x, 'pdfs': exponential.pdf(x), 'cdfs': exponential.cdf(x)})
df.head()

from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Define x values and find pdfs and cdfs of normal distribution
x = np.linspace(-20,20,1000)
normal = norm(loc = 3, scale = 2) #Where loc = mean, and scale = standard deviation
df = pd.DataFrame({'x': x, 'pdfs': normal.pdf(x), 'cdfs': normal.cdf(x)})
df.head()
mean = norm.mean(loc = 3, scale = 2)
var = norm.var(loc = 3, scale = 2)
std = norm.std(loc = 3, scale = 2)

#Using python
from scipy.stats import norm
mean  = 175
sd = 6
#Calculate z-score
z1 = (170 - mean) / sd
z2 = (180 - mean) / sd
#Calculate probability
prob = norm.cdf(z2) - norm.cdf(z1)
print('The required probability:', prob)

#Probability of height to be under 4.5 ft.
prob_1 = norm(loc = 5.3 , scale = 1).cdf(4.5)
print('Prob. of height under 4.5ft:', prob_1)
#probability that the height of the person will be between 6.5 and 4.5 ft.
cdf_upper_limit = norm(loc = 5.3 , scale = 1).cdf(6.5)
cdf_lower_limit = norm(loc = 5.3 , scale = 1).cdf(4.5)
prob_2 = cdf_upper_limit - cdf_lower_limit
print('Prob. of in between 6.5ft and 4.5ft:', prob_2)
