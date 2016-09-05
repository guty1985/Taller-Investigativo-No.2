
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

import matplotlib.pyplot as plt


# In[3]:

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
# -1 & \mbox{si } x<-1 \\ x &\mbox{si } x<=0 <=1\\ -1 & \mbox{si } x < -1 
# \end{cases} 
# $$

# In[625]:

Hv = lambda x: np.piecewise(x, [x <0, x==0,x==1, x > 1.0], [0,lambda x:x,lambda x:x,1.0])


# In[626]:

x = np.linspace(-2, 2, 5)


# In[627]:

plt.axis([x[0], x[-1], -1, 1.3])


# In[628]:

plt.plot(x, Hv(x),'-b' )


# In[629]:

plt.xlabel('x'), plt.ylabel('f(x)')


# In[630]:

plt.legend(['Satlin'])


# In[631]:

plt.title('Funcion Lineal Mixta')


# In[632]:

plt.show()


# Funcion Escalona $$f(x) = 
# \begin{cases} 
# 1 & \mbox{si } x>= 0 \\ 0 & \mbox{si } x < 0 
# \end{cases} 
# $$

# In[331]:

Hv = lambda x: np.piecewise(x, [x < 0.0, x >= 0.0], [0.0, 1.0])


# In[332]:

x = np.linspace(-2, 2, 100)


# In[333]:

plt.axis([x[0], x[-1], -1, 1.5])


# In[334]:

plt.plot(x, Hv(x),'-b' )


# In[335]:

plt.xlabel('x'), plt.ylabel('f(x)')


# In[336]:

plt.legend(['Hardlim'])


# In[337]:

plt.title('Funcion Escalon')


# In[338]:

plt.show()


# Función Tangente Hiperbólica $$f(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$

# In[985]:

def f(x):
      return (np.exp(x)-np.exp(-x))/ (np.exp(x)+np.exp(-x))  
      


# In[986]:

x = np.linspace(-10,10, num=1000)


# In[987]:

plt.title("Funcion Tangente Hiperbolica")


# In[988]:

pyplot.xlim(-10,10)


# In[989]:

pyplot.ylim(-1.2,1.5)


# In[990]:

plt.plot(x,f(x))
plt.xlabel("X")
plt.ylabel("f(x)")
plt.legend(['Tansig'])


# In[991]:

plt.show()


# Función Sigmoidal $$f(x)=\frac{1}{1+e^{-x}}$$

# In[994]:

def f(x):
      return 1/ (1+np.exp(-x))  
      


# In[995]:

x = np.linspace(-10,10, num=1000)


# In[996]:

plt.title("Funcion Sigmoidal")


# In[997]:

pyplot.xlim(-10,10)


# In[998]:

pyplot.ylim(-1.2,1.5)


# In[999]:

plt.plot(x,f(x))
plt.xlabel("X")
plt.ylabel("f(x)")
plt.legend(['Logsig'])


# In[1069]:

plt.show()


# In[ ]:




# In[ ]:



