#!/usr/bin/env python
# coding: utf-8

# In[98]:


#Script for Part II of the project.
#This script concerns scaling early dark energy (EDE), a component of the energy density that exists 
#before CMB decoupling, and whether it could alleviate or even solve the H_0 and S_8 tensions.
#This part of the code will evaluate the possible values of the EDE density parameter (Omega_e) and 
#the conditions on the dark energy period for which the H_0 tension could be solved. 
#The potential values of Omega_e will be calcuated through comparing values for the sound horizon at
#decoupling from the EDE and LCDM models. 

#Essential imports:
import math
import matplotlib.pyplot as plt 
import numpy as np

#Density parameters:
Omega_m_0 = 0.32 #Current value of the matter density parameter
Omega_de_0 = 0.68 #Current value of the dark energy density parameter
Omega_gamma_0 = (2.07*(10**-51.0))/(8.10*(0.674**2)*(10**-47.0)) #Current photon density parameter
Omega_b = 0.02222/(0.674**2) #Current value of the baryon density parameter

#Initial conditions and constants:
z_initial = 38500 #Initial redshift
a_0 = 1.0 #Present value of the scale factor
m_pl = 2.38*10**18 #Conversion from GeV to m_pl
m = 1.0 #Mass of the scalar particle constituting the scalar field 
R = 0 #Ratio of baryon density parameter to photon density parameter
c_s = 0 #Speed of sound in the baryon-photon plasma

#Present energy density values (units of GeV^4):
rho_0 = (8.10*(0.674**2)*(10**-47.0)) #Present critical density.
rho_m_0 = (0.32*rho_0) #Present matter density
rho_r_0 = (1.68*(2.07*(10**-51.0))) #Present radiation density
rho_de_0 = Omega_de_0*rho_0 #Present dark energy energy density (oof that's a mouthful)
rho_gamma_0 =Omega_gamma_0*rho_0 #Present photon energy density
rho_b_0 = Omega_b*rho_0 #Present baryon energy density




def first_run(z_b, z_e, Omega_e, model):
    #-------------------------------------------------------------------------------------------------------
    #A function for finding the value of the sound horizon 
    
    #Inputs:
    #z_b: The redshift value corresponding to the beginning of the dark energy period.
    #z_e: The redshift value corresponding to the end of the dark energy period.
    #Omega_e: Fraction of the total energy density that would be in EDE at the initial redshift (z_i).
    #It's equivalent to rho_phi/rho.
    #model: The model being used (either TSQ or LCDM)
    
    #Outputs:
    #H: Value of the sound horizon (depending on the model).
     #-------------------------------------------------------------------------------------------------------
    
    #Setting variables' initial values:
    
    z_i = 38500 #Initial redshift
    z_dec = 1100 #Redshift at decoupling
    a_dec = a_0/(1.0 + z_dec) # scale factor at decoupling
    a = a_0/(1.0 + z_i) #Scale factor's initial value
    r_s = 0.0 #Sound horizon
    d_r_s = 0.0 #Sound horizon step size
    Lambda = ((4/Omega_e)**(0.5))
    V = (rho_r_0*(a_0/a)**4.0)/(3.0*(((Lambda**2.0)/4.0)-1.0)) #Initial value of the scalar potential
    phi = (m_pl/Lambda)*np.log((m_pl**4.0)/V) #Initial value of the scalar field
    z = np.sqrt(4.0*V) #Initial value of the first order time derivative of phi
    H = 0.0 #Hubble parameter corresponding to EDE
    H_hat = 0.0 #Hubble parameter corresponding to idealised TSQ
    H_0 = 0.0 #Value of the Hubble constant (model dependent)
    t = 0.0 #Time in seconds
    rho_phi = 0.0 #Energy density of the scalar field
    s = 0.0 #Iteration parameter for a 
    ds = 0.0001*0.5 #Step size of s
   
    while a < a_dec:
        
        #Changing the scalar potential
        V = (m_pl**4)*np.exp((-Lambda*phi)/m_pl)

        #Calculating the components of rho:
        rho_b = rho_b_0*(a_0/a)**3
        rho_r = rho_r_0*(a_0/a)**4
        rho_m = rho_m_0*(a_0/a)**3
        rho_gamma = rho_gamma_0*(a_0/a)**4 #Assuming photons and radiation scale together
        rho_phi = (0.5*z*z) + V #Scaling energy density
        r = (3/4)*(rho_b/rho_gamma) #Ratio used in c_s
        c_s = 1/((3*(1 + r))**0.5) #Speed of sound in the baryon-photon fluid
        
        #Determining the sound horizon's value depending on the cosmological model used:
        if model == 'TSQ':
             
            #Calculating the redshift and using it to determine whether the current redshift
            #is within the dark energy period:
            f = 1 #Function used to switch off EDE outside of the dark energy period
            red_shift = (a_0/a) - 1
            #If redshift is outside the dark energy period, rho_phi = 0
            if red_shift > z_b:
                f = 0
            if red_shift < z_e:
                f = 0
                
            
            #Finding the correct expression for rho:
            #rho_hat = rho_m + rho_r + rho_phi + rho_de_0
            rho_hat = rho_m + rho_r + rho_phi
            H_hat = (rho_hat/(3.0*m_pl**2.0))**0.5
            #rho = rho_m + rho_r + (rho_phi*f) + rho_de_0
            rho = rho_m + rho_r + rho_phi
            H = (rho/(3.0*m_pl**2.0))**0.5
            
            #Changing variable increment values:
            da = a*ds #Scale factor increment
            dt = da/(a*H) #Time increment
            d_phi = z*dt #phi increment
            dz = ((m_pl**3)*Lambda*np.exp(-Lambda*phi/m_pl) - 3*H_hat*z)*dt #z (first order derivative of phi wrt time) increment
            d_r_s = a_dec*(da)*c_s/((a**2)*H) #Sound horizon increment
        
        elif model == 'LCDM':
            
            rho = rho_m + rho_r + rho_de_0
            H = (rho/(3.0*m_pl**2.0))**0.5
            
            da = a*ds 
            dt = da/(a*H)
            d_phi = z*dt 
            dz = ((m_pl**3)*Lambda*np.exp(-Lambda*phi/m_pl) - 3*H*z)*dt
            d_r_s = a_dec*(da)*c_s/((a**2)*H) 
       
        #Appending step values to their respective variables/changing necessary values:
        s = s + ds
        a = a + da
        z = z + dz
        phi = phi + d_phi
        t = t + dt
        r_s = r_s + d_r_s 
        
    print(H)   
    return r_s

def get_frac_r_s(z_b, z_e_list, Omega_e):
    #---------------------------------------------------------------------------------------------
    #A procedure for finding the fractional change in the sound horizon.
    
    #Inputs:
    #z_b: The redshift value corresponding to the beginning of the dark energy period.
    #z_e_list:A list containing redshift values corresponding to the end of the dark energy period
    #for different test cases.
    #Omega_e: Fraction of the total energy density that would be in EDE at z_i.
    #It's equivalent to rho_phi/rho.
    #----------------------------------------------------------------------------------------------
    
    for z in z_e_list:
        r_s_TSQ = first_run(z_b, z, Omega_e, 'TSQ')
        r_s_LCDM = first_run(38500, 1100, Omega_e, 'LCDM') #This value does not change depending on the value of Omega_e
        
        Delta_r_s = 9.53*10**35 #GeV^-1, modification to improve the accuracy of r_s
        frac_r_s= (r_s_TSQ - r_s_LCDM)/(r_s_LCDM + Delta_r_s)
        
        print(frac_r_s)

z_e_vals = [2000]
get_frac_r_s(5000, z_e_vals, 0.23)


# In[ ]:


n


# In[ ]:





# In[ ]:




