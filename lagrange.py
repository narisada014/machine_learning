from sympy import *

# sympy.init_printing()

x = Symbol('x')
y = Symbol('y')

print(diff((2*x+3*y), x))
print(diff((2*x+3*y), y))


