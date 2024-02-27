#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Script for Part 1 of the project.
#Developing a numerical solution of the scalar field equation
#for the Triggered Scaling Quintessence (TSQ) potential V(phi). 
#The code will generate luminosity distance (D_L), Growth Factor (G)
#redshift (z) and Hubble Parameter (H) values for each iteration.
#The D_L, G and H values from the TSQ model will be
#compared with those from the LCDM model over a given redshift range.
 
#Essential imports:
import math
import matplotlib.pyplot as plt 
import numpy as np

#Density parameters:
Omega_m_0 = 0.32 #Current value of the matter density parameter
Omega_de_0 = 0.68 #Current value of the energy density parameter

#Initial redshift:
z_initial = 38500

#Present energy density values (units of GeV^4):
rho_0 = (8.10*(0.674**2)*(10**-47.0)) #Present critical density.
rho_m_0 = (0.32*rho_0) #Present matter density
rho_r_0 = (1.68*(2.07*(10**-51.0))) #Present radiation density
rho_de_0 = Omega_de_0*rho_0 #Present dark energy energy density (oof that's a mouthful)

#Setting critical (related to the transition) values:
a_0 = 1.0 #Present value of the scale factor
m_pl = 2.38*10**18 #Conversion from GeV to m_pl
m = 1.0 #Mass of the scalar particle constituting the scalar field 
z_c = 15.0 #Critical redshift (redshift value at which the scalar potential transitions)
a_c = a_0/(1 + z_c) #Scale factor's value at the transition
Omega_phi = (2*Omega_de_0)/((2*Omega_de_0)+(Omega_m_0*((a_0/a_c)**3))) #Density parameter of phi (scalar field)
Lambda_c = ((3/Omega_phi)**(0.5)) #Perturbation length scale at the transition.
eta = 10.0 #Parameter that adjusts the potential's slope at phi > phi_c

#Creating lists to hold variables. Variables will be appended to lists at the end of each iteration.
#Values will then be written to files, with the files then exported to a different script for plotting.    

a_list = []
rho_list = []
phi_list = []
H_list = []

#Prelim run:
def Prelim_run(z_i, Lambda):
    #-------------------------------------------------------------------------------------------------------
    #A function for finding the difference between the current energy densities of phi and dark energy.
    #This value will then be used to correct for the small amount of energy due to the oscillations of the
    #phi field about the scalar potential's minimum.

    #Inputs:
    #z_i: Redshift value at the transition
    #Lambda: Perturbation length scale at the transition
    
    #Outputs:
    #delta_rho_phi_0: Difference between the present values of scalar field energy density and
    #cosmological constant/dark energy energy density.
    #-------------------------------------------------------------------------------------------------------
    
    #Setting variables' initial values:
    
    a = a_0/(1.0 + z_i) #Scale factor's initial value
    V = (rho_r_0*(a_0/a)**4.0)/(3.0*(((Lambda**2.0)/4.0)-1.0)) #Initial value of the scalar potential
    phi = (m_pl/Lambda)*np.log((m_pl**4.0)/V) #Initial value of the scalar field
    z = np.sqrt(4.0*V) #Initial value of the first order time derivative of phi
    H = 0.0 #Hubble parameter 
    t = 0.0 #Time in seconds
    rho_phi = 0.0 #Energy density of the scalar field
    D = 0.0 #Luminosity distance as a function of redshift
    s = 0.0 #Iteration parameter for a (equal to da/a)
    ds = 0.000001*0.5 #Step size of s
    V_c = rho_de_0 #Value of the scalar field potential at the transition.
    phi_c = (m_pl/Lambda)*np.log(m_pl**4/V_c) #Value of phi at the transition
    
    while a < 1.0:
        
        #Evaluating the value of phi to see whether the transition has occured.
        #If phi is larger than phi_c, the trigger is activated and the potential
        #will be switched. 
        
        if phi < phi_c:
            V = (m_pl**4)*np.exp((-Lambda*phi)/m_pl)
        if phi > phi_c:
            V = (m_pl**4)*np.exp((-Lambda*phi_c)/m_pl)*np.exp((eta*Lambda*(phi - phi_c))/m_pl)

        #Calculating the components of rho (so H can be found init):
        #Mass and photon energy densities will have to be scaled back from their present values.

        #Scaling energy density values depending on the value of a:
        
        rho_r = rho_r_0 *(a_0/a)**4
        rho_m = rho_m_0 *(a_0/a)**3
        rho_phi = (0.5*z*z) + V
        rho = rho_m + rho_r +rho_phi    
        H = (rho/(3.0*m_pl**2.0))**0.5
        
        #Changing derivative step values:
       
        da = a*ds #Scale factor increment
        dt = da/(a*H) #Time increment
        d_phi = z*dt #phi increment
        
        #Changing the second derivative of phi depending on whether the transition has occured:
        
        if phi < phi_c:
            dz = ((m_pl**3)*Lambda*np.exp(-Lambda*phi/m_pl) - 3*H*z)*dt
        if phi > phi_c:
            dz = (-(m_pl**3)*eta*Lambda*np.exp(-Lambda*phi_c/m_pl)*(np.exp(eta*(Lambda*(phi - phi_c)/m_pl))) -3*H*z)*dt

        #Appending step values to their respective variables/changing necessary values:
        
        s = s + ds
        a = a + da
        z = z + dz
        phi = phi + d_phi
        t = t + dt
    
    #Finally, finding the difference between the current energy densities of phi and dark energy:
    
    rho_phi_0 = rho_phi
    delta_rho_phi_0 = rho_phi_0 - rho_de_0
    return delta_rho_phi_0

def first_run(z_i, Lambda, model):
    #-------------------------------------------------------------------------------------------------------
    #A function for computing the first run of the luminosity distance against redhsift sequence for a given
    #model of dark energy.
    
    #Inputs:
    #z_i: Redshift value at the transition
    #Lambda: Perturbation length scale at the transition
    #model: The model being used (either TSQ or LCDM)
    
    #Outputs:
    #D: Physical distance as a function of redshift from initial to final value of the scale factor.
    #H: Value of the Hubble Constant (depending on the model).
    
    #-------------------------------------------------------------------------------------------------------
    
    #Setting variables' initial values:
    
    a = a_0/(1.0 + z_i) #Scale factor's initial value
    V = (rho_r_0*(a_0/a)**4.0)/(3.0*(((Lambda**2.0)/4.0)-1.0)) #Initial value of the scalar potential
    phi = (m_pl/Lambda)*np.log((m_pl**4.0)/V) #Initial value of the scalar field
    z = np.sqrt(4.0*V) #Initial value of the first order time derivative of phi
    H = 0.0 #Hubble parameter
    t = 0.0 #Time in seconds
    rho_phi = 0.0 #Energy density of the scalar field
    D = 0.0 #Luminosity distance as a function of redshift
    s = 0.0 #Iteration parameter for a 
    ds = 0.000001*0.5 #Step size of s
    delta_rho = Prelim_run(z_initial, Lambda_c) #Difference in the present energy densities of phi and dark matter.
    V_c = rho_de_0*(1 - (delta_rho/rho_de_0)) #Value of the scalar field potential at the transition.
    phi_c = (m_pl/Lambda)*np.log(m_pl**4/V_c) #Value of phi at the transition
    
    while a < 1.0:
        
        #Evaluating the value of phi to see whether the transition has occured.
        #If phi is larger than phi_c, the trigger is activated and the potential
        #will be switched.
        
        if phi < phi_c:
            V = (m_pl**4)*np.exp((-Lambda*phi)/m_pl)
        if phi > phi_c:
            V = (m_pl**4)*np.exp((-Lambda*phi_c)/m_pl)*np.exp((eta*Lambda*(phi - phi_c))/m_pl)

        #Calculating the components of rho (so H can be found init):
        #Mass and photon energy densities will have to be scaled back from their present values.
        #Note that when calculating H, the third density component differs between the two models.
        #For the TSQ model, the third component is due to the energy density of the scalar field.
        #For the LCDM model, the third component is due to the energy density of the cosmological
        #constant/dark energy.

        #Scaling energy density values depending on the value of a:
        
        rho_r = rho_r_0 *(a_0/a)**4
        rho_m = rho_m_0 *(a_0/a)**3
        
        if model == 'TSQ':
            rho_phi = (0.5*z*z) + V
            rho = rho_m + rho_r +rho_phi    
            H = (rho/(3.0*m_pl**2.0))**0.5
        
        elif model == 'LCDM':
            rho_lambda = rho_de_0
            rho = rho_m + rho_r +rho_de_0    
            H = (rho/(3.0*m_pl**2.0))**0.5 
            
        #Changing derivative step values:
       
        da = a*ds #Scale factor increment
        dt = da/(a*H) #Time increment
        d_phi = z*dt #phi increment
        
        #Changing the second derivative of phi depending on whether the transition has occured:
        
        if phi < phi_c:
            dz = ((m_pl**3)*Lambda*np.exp(-Lambda*phi/m_pl) - 3*H*z)*dt
        if phi > phi_c:
            dz = (-(m_pl**3)*eta*Lambda*np.exp(-Lambda*phi_c/m_pl)*(np.exp(eta*(Lambda*(phi - phi_c)/m_pl))) -3*H*z)*dt

        #Calculating luminosity distance D_L as a function of redshift z:
        #Code will be run twice.
        #First run will will compute the complete D(a_i) integral (physical distance as a function of redshift)
        #from a_i to a_0.
        
        dD = (da)/((a**2)*H) #physical distance increment size
        D = D + dD 

        #Appending step values to their respective variables/changing necessary values:
        s = s + ds
        a = a + da
        z = z + dz
        phi = phi + d_phi
        t = t + dt
        
    return D, H

    
#Second run of the code: 

def second_run(z_i, Lambda, model):
    #-------------------------------------------------------------------------------------------------------
    #A procedure for iterating the second run of the luminosity distance-redshift sequence for a given
    #model of dark energy. The growth factor-redshift sequence (for the same chosen dark energy model) 
    #will be iterated in tandem.
    
    #Inputs:
    #z_i: Redshift value at the transition
    #Lambda: Perturbation length scale at the transition
    #model: The model being used (either TSQ or LCDM)
    
    #Outputs:
    #What files are outputs can be chnaged through changing the code.
    #TSQ_DL.txt or LCDM_DL.txt (depending on model): text file containing luminosity distances as a function 
    #of redshift. 
    #TSQ_G.txt or LCDM_G.txt (depending on model): text file containing growth factor values as a function
    #of redshift.
    #TSQ_H.txt or LCDM_H.txt (depending on model): text file containing hubble parameter values as a function
    #of redshift.
    #TSQ_z.txt or LCDM_z.txt (depending on model): text file containing redshift values within a chosen 
    #range.
    #-------------------------------------------------------------------------------------------------------
    
    #Setting variable initial values:
   
    a = a_0/(1.0 + z_i) #Scale factor's initial value
    V = (rho_r_0*(a_0/a)**4.0)/(3.0*(((Lambda**2.0)/4.0)-1.0)) #Initial value of the scalar potential
    phi = (m_pl/Lambda)*np.log((m_pl**4.0)/V) #Initial value of the scalar field
    z = np.sqrt(4.0*V) #Initial value of the first order time derivative of phi
    H = 0.0 #Hubble parameter
    t = 0.0 #Time in seconds
    rho_phi = 0.0 #Energy density of the scalar field
    s = 0.0 #Iteration parameter for a 
    ds = 0.000001*0.5 #Step size of s
    
    #Finding the value of the Hubble constant depending on the model:
    if model == 'TSQ':
        D, H_0 = first_run(z_i, Lambda_c, 'TSQ')
        
    if model == 'LCDM':
        D, H_0 = first_run(z_i, Lambda_c, 'LCDM')
    
    delta_rho = Prelim_run(z_initial, Lambda_c) #Difference in the present energy densities of phi and dark matter.
    V_c = rho_de_0*(1 - (delta_rho/rho_de_0)) #Value of the scalar field potential at the transition.
    phi_c = (m_pl/Lambda)*np.log(m_pl**4/V_c) #Value of phi at the transition
    
    I = 0.0 #Integral for the second run of the luminosity distance code. 
    
    #Creating lists to hold parameter values from each iteration;
    z_list = [] 
    DL_list = []
    G_list = []
    H_list = []
    
    #Variables related to the growth of density perturbations:
    G = 1.0 #Growth factor (function of time)
    r = 0.0 #First order derivative of G with respect to time
    
    while a < 1.0:
        #Evaluating the value of phi to see whether the transition has occured:
        if phi < phi_c:
            V = (m_pl**4)*np.exp((-Lambda*phi)/m_pl)
        if phi > phi_c:
            V = (m_pl**4)*np.exp((-Lambda*phi_c)/m_pl)*np.exp((eta*Lambda*(phi - phi_c))/m_pl)

        #Calculating the components of rho (so H can be found init):
        #Mass and photon energy densities will have to be scaled back from their present values

        rho_r = rho_r_0 *(a_0/a)**4
        rho_m = rho_m_0 *(a_0/a)**3
        
        if model == 'TSQ':
            rho_phi = (0.5*z*z) + V
            rho = rho_m + rho_r +rho_phi    
            H = (rho/(3.0*m_pl**2.0))**0.5
        
        elif model == 'LCDM':
            rho_lambda = rho_de_0
            rho = rho_m + rho_r +rho_de_0    
            H = (rho/(3.0*m_pl**2.0))**0.5 

        #Changing the step value for each variable:
        da = a*ds 
        dt = da/(a*H) 
        d_phi = z*dt    
        
        if phi < phi_c:
            dz = ((m_pl**3)*Lambda*np.exp(-Lambda*phi/m_pl) - 3*H*z)*dt
        if phi > phi_c:
            dz = (-(m_pl**3)*eta*Lambda*np.exp(-Lambda*phi_c/m_pl)*(np.exp(eta*(Lambda*(phi - phi_c)/m_pl))) -3*H*z)*dt

        #Calculating luminosity distance D(a) as a function of redshift z:
        #Second run involves subtracting I from the D(a_i) value obtained during the first run
        
        #dI = (da)/((a**2)*H)
        #D_a = D - I 
        
        #Calculating redshift as D_a only needs to be evaluated over a specific redshift range:
        #red_shift = (a_0/a) - 1
        
        #Calculating the growth factor for each iteration:
        #dG = r*dt
        #dr = ((3/2)*Omega_m_0*(H_0**2)*((1 + red_shift)**3)*G - 2*H*r)*dt
        
        
        
        #Appending D_L, G, z and H values to their respective lists: 
        #if red_shift >= 0.0 or red_shift <= 20.0:
            #z_list.append(red_shift)
            #DL = (1 + red_shift)*D_a #Calculating D_L values 
            #DL_list.append(DL)
            #G_list.append(G)
            #H_list.append(H)
            
        #Appending step values to their respective variables/changing necessary values:
        s = s + ds
        a = a + da
        #I = I + dI
        z = z + dz
        phi = phi + d_phi
        #r = r + dr
        #G = G + dG
        t = t + dt
    
    print(H)
    #Writing D_L, G, H and z values to separate files:
    
    #with open('LCDM_DL_3_zero_to_20.txt', 'w+') as f:
        #for i in DL_list:
            #i = str(i)
            #f.write('%s\n' %i)
            
    #f.close()
    
    #with open('LCDM_z_3_zero_to_20.txt', 'w+') as g:
        #for i in z_list:
            #i = str(i)
            #g.write('%s\n' %i)
            
    #g.close()
    
    #with open('LCDM_G_15_zero_to_20.txt', 'w+') as m:
        #for i in G_list:
            #i = str(i)
            #m.write('%s\n' %i)
            
    #m.close()
    
    #with open('LCDM_H_15_zero_to_20.txt', 'w+') as k:
        #for i in H_list:
            #i = str(i)
            #k.write('%s\n' %i)
            
    #k.close()
        
            
    
TSQ = second_run(z_initial, Lambda_c, 'TSQ')    
LCDM = second_run(z_initial, Lambda_c, 'LCDM')
print(TSQ)
print(LCDM)
result = (TSQ - LCDM)/LCDM
print(result)
    
    

    


# In[ ]:





# In[ ]:




