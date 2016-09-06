
# coding: utf-8

# In[76]:

import numpy as np


# In[77]:

import matplotlib.pyplot as plt


# In[78]:

from matplotlib import pyplot


# Funcion Lineal $$f(x)=x$$  

# In[39]:

def f(x):
    return x


# In[43]:

x= range(-1,2)


# In[44]:

pyplot.plot(x,[f(i) for i in x])


# In[45]:

pyplot.xlim(-2,2)


# In[46]:

plt.title("Funcion Lineal")


# In[47]:

plt.xlabel("X")


# In[877]:

plt.ylabel("f(x)")


# In[49]:

pyplot.show()


# Funcion Lineal Mixta $$f(x) = 
# \begin{cases} 
# 0 & \mbox{si } x<0 \\ x &\mbox{si } x<=0 <=1\\ 1 & \mbox{si } x > 1 
# \end{cases} 
# $$

# In[12]:

Hv = lambda x: np.piecewise(x, [x <0, x==0,x==1, x > 1.0], [0,lambda x:x,lambda x:x,1.0])


# In[13]:

x = np.linspace(-2, 2, 5)


# In[14]:

plt.axis([x[0], x[-1], -1, 1.3])


# In[15]:

plt.plot(x, Hv(x),'-b' )


# In[16]:

plt.xlabel('x'), plt.ylabel('f(x)')


# In[17]:

plt.legend(['Satlin'])


# In[18]:

plt.title('Funcion Lineal Mixta')


# In[19]:

plt.show()


# Funcion Escalona $$f(x) = 
# \begin{cases} 
# 1 & \mbox{si } x>= 0 \\ 0 & \mbox{si } x < 0 
# \end{cases} 
# $$

# In[218]:

Hv = lambda x: np.piecewise(x, [x < 0.0, x >= 0.0], [0.0, 1.0])


# In[219]:

x = np.linspace(-2, 2, 100)


# In[220]:

plt.axis([x[0], x[-1], -1, 1.5])


# In[221]:

plt.plot(x, Hv(x),'-b' )


# In[222]:

plt.xlabel('x'), plt.ylabel('f(x)')


# In[223]:

plt.legend(['Hardlim'])


# In[224]:

plt.title('Funcion Escalon')


# In[225]:

plt.show()


# Función Tangente Hiperbólica $$f(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$

# In[230]:

def f(x):
      return (np.exp(x)-np.exp(-x))/ (np.exp(x)+np.exp(-x))  
      


# In[231]:

x = np.linspace(-10,10, num=1000)


# In[232]:

plt.title("Funcion Tangente Hiperbolica")


# In[233]:

pyplot.xlim(-10,10)


# In[234]:

pyplot.ylim(-1.2,1.5)


# In[235]:

plt.plot(x,f(x))
plt.xlabel("X")
plt.ylabel("f(x)")
plt.legend(['Tansig'])


# In[236]:

plt.show()


# Función Sigmoidal $$f(x)=\frac{1}{1+e^{-x}}$$

# In[35]:

def f(x):
      return 1/ (1+np.exp(-x))  
      


# In[36]:

x = np.linspace(-10,10, num=1000)


# In[37]:

plt.title("Funcion Sigmoidal")


# In[38]:

pyplot.xlim(-10,10)


# In[39]:

pyplot.ylim(-1.2,1.5)


# In[40]:

plt.plot(x,f(x))
plt.xlabel("X")
plt.ylabel("f(x)")
plt.legend(['Logsig'])


# In[41]:

plt.show()


# Funcion Gaussiana  

# In[305]:

sigma= 0.01
x=np.linspace(-5,5, num=1000)


# In[306]:

def f(x):
            return 1/(sigma *np.sqrt(2 * np.pi)) *(np.exp(-x**2) / (2 * sigma**2))   


# In[307]:

plt.plot(x,f(x))
plt.xlabel("X")
plt.ylabel("f(x)")


# In[308]:

plt.grid()
plt.show()


# In[ ]:



