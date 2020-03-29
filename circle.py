# 1. prompt user for the radius
# 2. apply the area forluma
# 3. print the result
import math
radius_string = input("Enter the radius of your circle:")
rafius_float = float(radius_string)
circumference = 2 * math.pi * rafius_float
area = math.pi * rafius_float * rafius_float
print()
print ("The circumference of your circle is: ", circumference, \
    ", and the area is:", area)