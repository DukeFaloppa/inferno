 <!-- (        )  (          (        )     )   
 )\ )  ( /(  )\ )       )\ )  ( /(  ( /(   
(()/(  )\())(()/(  (   (()/(  )\()) )\())  
 /(_))((_)\  /(_)) )\   /(_))((_)\ ((_)\   
(_))   _((_)(_))_|((_) (_))   _((_)  ((_)  
|_ _| | \| || |_  | __|| _ \ | \| | / _ \  
 | |  | .` || __| | _| |   / | .` || (_) | 
|___| |_|\_||_|   |___||_|_\ |_|\_| \___/  
                                       -->

# :fire: inferno :fire: 
PINN based solver for heat equation

## introduction

this is a simple 0D time dependent heat equation sovler.

$ \Delta T(t) = \Delta T_0 e^{-r(t-t0)} + Q_0 e^{s(t-t_0)} $

It consider the temperature evolution as a function with two source terms:

+ heat source, that increase the temperature
+ heat sink, that remove part of the heat source

# v0.0

for the moment we are dealing wiht convection only

$ \Delta T(t) = \Delta T_0 e^{-r(t-t0)} $

this is already available, see the example *0D_cooling_only.py*

If we consider an heat source term of the kind:

$ \Delta T(t) = \Delta T_0 e^{-r(t-t0)} + Q_0 e^{s(t-t_0)} $

then look at *0D_heat_source.py*

**!!! important**
parameters optimization must be done when you fit your model

## the repository

### requirements

+ pytorch
+ numpy
+ matplotlib (for plotting)
+ seaborn