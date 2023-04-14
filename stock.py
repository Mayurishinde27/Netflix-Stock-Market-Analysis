# %%
import pandas as pd # for csv file handling
import numpy as np # for numerical calculations
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs for better visualization
from sklearn.linear_model import LinearRegression # for linear regression
from sklearn.model_selection import train_test_split # for splitting data into training and testing
from sklearn.metrics import r2_score # for checking accuracy of model

# %%
df = pd.read_csv('NFLX.csv') #reading the csv file 
df.head() #printing first 5 rows of the file

# %%
df.describe() #describing the data

# %%
df.set_index('Date', inplace=True) #setting the index as date
df.head() 

# %%
#plotting the graph
df[['Open','High','Low','Close','Adj Close']].plot(figsize=(16,11)) 
plt.title('Netflix Stock Price', fontsize=23) 
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()


# %%
# Plotting the graph using seaborn
sns.pairplot(df) # for plotting the graph in pair
plt.show() # for showing the graph


# %%
data_l = df.sort_values( by = ['Low'], ascending=True) #sorting the data according to low
data_l['Low'].plot( figsize = (10,6), color='Red') #plotting the graph
plt.title('Yearwise trend of Netflix stock lows') #setting the title

# %%
# establishes a relationship between two variables adj close and volume
sns.jointplot( x ='Volume', y ='Adj Close', data = df, kind='reg', color = 'green') 

# %%
# Graph for Daily Returns
df['Daily_returns'] = df['Adj Close'].pct_change() #calculating the daily returns
plt.title('Daily returns') #setting the title
df['Daily_returns'].plot(figsize = (10,5), color='orange') #plotting the graph


# %%
# Graph for Cumulative Returns
df['Cumulative_returns'] = (df['Daily_returns']+1).cumprod() #calculating the cumulative returns
plt.title('Cumulative returns') #setting the title
df['Cumulative_returns'].plot(figsize = (10,5), color='red') #plotting the graph


# %%
# Graph for mean of Closing price
df['Close'].rolling(window = 30).mean().plot() #calculating the mean of closing price
plt.title('Mean of Closing price',fontsize=15) #setting the title
df['Close'].plot(figsize=(15,5),color='orange') #plotting the graph and setting the figure size, color

# %%
x = df[['High','Low', 'Open']].values # setting the values for x
y = df[['Close']].values # setting the values for y

# %%
# splitting the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2,random_state = 0) 

# %%
# linear regression
lr = LinearRegression() #creating the object of linear regression
lr.fit(x_train,y_train) #fitting the data into linear regression

# %%
lr.coef_ #returns the coefficient of the linear regression

# %%
lr.score(x_train,y_train) # calculates the coefficient of determination

# %%
# Predicting the values
predictions = lr.predict(x_test) # contains the predicted target values for the test dataset ie x_test


# %%
# use for checking the accuracy of the model
r2_score(y_test, predictions) # returns the coefficient of determination

# %%
# creating a dataframe for actual and predicted values
data_frame = pd.DataFrame({'Actual Data':y_test.flatten(),'Predicted Data':predictions.flatten()}) # flattening the array
data_frame.head() # printing the first 5 rows of the dataframe


# %%
# plotting the graph for actual and predicted values
graph = data_frame.head(10) # taking the first 10 rows of the dataframe
graph.plot.bar() # plotting the graph
plt.title('Actual vs Predicted') # setting the title
plt.ylabel('Closing price') # setting the y label

# %%
# plotting the graph for actual and predicted values for linear regression
plt.scatter(y_test,predictions, color = 'orange') # plotting the graph
plt.title('Actual vs Predicted') # setting the title



# %%



