#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 14:01:48 2020

@author: vgs23
"""

import ftype
print(ftype.__doc__)
type(ftype.foo),type(ftype.data)
ftype.foo()
 
ftype.data.a = 3
ftype.data.x = [1,2,3]
ftype.foo()
 
ftype.data.x[1] = 45  
ftype.foo(24)
 
ftype.data.x