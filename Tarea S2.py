


import datetime
import numpy as np
import numpy as np
import pandas as pd

    # 10 Working With Datetime #

Res = input('Quiere saber la fecha y hora exacta de este instante?, responda Y/N')
x = datetime.datetime.now()

if(Res == 'Y'):
    print(x)
    print(x.year)
    print(x.strftime("%A"))
    print(x.strftime("%B"))
    print(x.strftime("%d"))
    print(x.strftime("%H:%M:%S %p"))
else:
    print('Buen dia')

    # 11 Numpy #
print('Parte numpy')
# Create array
a = np.arange(24).reshape( 6, 4) # Create array with range 0-14 in 3 by 5 dimension
b = np.zeros((2,10)) # Create array with zeroes
c = np.ones( (3,2,2), dtype=np.int16 ) # Create array with ones and defining data types
d = np.ones((1,1))

print('Soy a',a)
print()
print('Soy b', b)
print()
print('Soy c',c)
print()
print('Soy d',d)
print()
print('La dim. de a',a.shape) # Array dimension
print()
print('La long de b', len(b))# Length of array
print()
print('Dim. de la colecc. en c', c.ndim) # Number of array dimensions
print()
print ('Numero de elementos en la coleccion c', c.size) # Number of array elements
print()
b.dtype # Data type of array elements
print()
print ('Que hay en c', c.dtype.name) # Name of data type
print()
c.astype(float) # Convert an array type to a different type
print()
a_1 = np.arange(6).reshape(2, 3) # Create array with range 0-14 in 3 by 5 dimension
b_1 = np.zeros((2,3)) # Create array with zeroes
c_1 = np.ones( (2,2,3), dtype=np.int16 ) # Createarray with ones and defining data types
d_1 = np.ones((2,3))
print(a_1)
print()
print( b_1)
print()
print('|',c_1,'|')
print()
print(d_1)
print()
print('suma de a_1 y b_1', np.add(a_1,b_1)) # Addition
print('resta de a_1 y c_1',np.subtract(a_1,c_1)) # Substraction
print('div de d_1 y a_1', np.divide(a_1,d_1)) # Division
print('mul de a_1 y b_1',np.multiply(a_1,b_1)) # Multiplication
print('Son =es d_1 y c_1',np.array_equal(d_1,c_1)) # Comparison - arraywise
a_2 = np.arange(15).reshape(3, 5)
b_2 = np.ones((3,5))
c_2 = np.ones( (2,3,4), dtype=np.int16 ) # Createarray with ones and defining data types
d_2 = np.ones((3,5))
w = np.subtract(a_2,b_2)
print(a_2)
print(b_2)
print(w)
print('Sumar todos los numeros que aparecen en a_2',a_2.sum())
print('Restaremos a_2 con b_2 y tomaremos el min. que debe ser ',w.min() )
print('Promedio de los elementos de a del principio',a.mean() )
print('La ultima fila de a',a.max(axis=0))
print('Que elemento de la matriz a esta en la primera fila tercera columna',a[0,2])
print(a[0:3])
print('Soy w')
print(w)
print('La primera fila de w')
print(w[:1])
print('Los elementos menores a 12 de w', w[w<12])
print('La transpuesta de a de toda la vida', np.transpose(a)) # Transpose array 'a'
print('Volver la matriz a una sola columna', a.ravel()) # Flatten the array
print('Volver a una matriz de 3x8 en lugar de 6x4',a.reshape(3,8)) # Reshape but don't change the data
print(np.append(a_2,b_2)) # Append items to the array
print(np.concatenate((a_2,d_2), axis=0)) # Concatenate arrays
print(np.vsplit(a_2,3)) # Split array vertically at 3rd index
print(a)
print(np.hsplit(a,2)) # Split array horizontally at 5th index
print('Empezamos con panda')
df = pd.DataFrame({'hab en millones': [128, 329, 43, 50],
                   'hab en cap en millones': [9, 1, np.nan, 7],
                   'eleccion de numeroos random': [143, 433 , np.nan, 432]},
                   index=['Mex', 'USA', 'Antartida', 'Col'])
print(df) # Display dataframe df

df1 = pd.date_range('20130101', periods=6)
df1 = pd.DataFrame(np.random.randn(6, 4), index=df1, columns=list('ABCD'))
print(df1)


print(df1.head(2)) # View top data
print(df1.tail(2)) # View bottom data
print(df1.index) # Display index column
print(df1.dtypes) # Inspect datatypes
print(df.describe()) # Display quick statistics summary of data


print(df1.T) # Transpose data
print(df.sort_index(axis=1, ascending=False)) # Sort by an axis
print(df1.sort_values(by='B')) # Sort by values
print(df1['A']) # Select column A
print (df1[0:3]) # Select index 0 to 2
print(df1['20130102':'20130104']) # Select from index matching the values
print(df1.loc[:, ['D', 'C']]) # Select on a multi-axis by label
print(df1.iloc[3]) # Select via the position of the passed integers
print(df1[df1 > 0])# Select values from a DataFrame where a boolean condition is met
df2 = df1.copy() # Copy the df1 dataset to df2
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three'] # Add column E with value
print(df2[df2['E'].isin(['two', 'four'])]) # Use isin method for filtering

print(df.dropna(how='any')) # Drop any rows that have missing data
df.dropna(how='any', axis=1) # Drop any columns that have missing data
df.fillna(value=5) # Fill missing data with value 5
pd.isna(df) # To get boolean mask where data is missing

#######

df = pd.DataFrame({'num_legs': [2, 4, np.nan, 0],
                   'num_wings': [2, 0, 0, 0],
                   'num_specimen_seen': [10, np.nan, 1, 8]},
                   index=['falcon', 'dog', 'spider', 'fish'])
df.to_csv('foo.csv') # Write to CSV file
df.to_excel('foo.xlsx', sheet_name='Sheet1') # Write to Microsoft Excel file
pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA']) 
###############
import matplotlib.pyplot as plt
ts = pd.Series(np.random.randn(1000),index=pd.date_range('1/1/2000', periods=1000)) 
ts.head()
ts = ts.cumsum()
ts.plot() # Plot graph
plt.show()
# On a DataFrame, the plot() method is convenient to plot all of the columns with labels
df4 = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,columns=['A', 'B', 'C', 'D'])
df4 = df4.cumsum()
df4.head()
df4.plot()
plt.show()
t = 23
