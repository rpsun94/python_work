l_first_call      = True
l_last_call       = False
ldebug_blk_algos  = False
nb_itt            = 5        #number of itteration in the bulk algorithm
rpi               = 3.141592653589793
rt0               = 273.15   #freezing point of fresh water [K]
rtt0              = 273.16   #riple point of temperature    [K]
grav              = 9.8      #acceleration of gravity    [m.s^-2]
Patm              = 101000   #
rho0_a            = 1.2      #Approx. of density of air         [kg/m^3]
Cp0_a             = 1015.0   #Specic heat of moist air          [J/K/kg]
Cp_dry            = 1005.0   #Specic heat of dry air, constant pressure[J/K/kg]
Cp_vap            = 1860.0   #Specic heat of water vapor, constant pressure  [J/K/kg]
R_dry             = 287.05   #Specific gas constant for dry air              [J/K/kg]
R_vap             = 461.495  #Specific gas constant for water vapor          [J/K/kg]
Cp0_w             = 4190     #Specific heat capacity of seawater (ECMWF 4190) [J/K/kg]
rho0_w            = 1025     #Density of sea-water  (ECMWF->1025)             [kg/m^3]
nu0_w             = 1.e-6    #kinetic viscosity of water                      [m^2/s]
k0_w              = 0.6      #thermal conductivity of water (at 20C)          [W/m/K]
reps0             = R_dry/R_vap #ratio of gas constant for dry air and water vapor => ~ 0.622
rctv0             = R_vap/R_dry -1 #for virtual temperature (== (1-eps)/eps) => ~ 0.608
nu0_air           = 1.5e-5   #kinematic viscosity of air    [m^2/s]
L0vap             = 2.46e6   #Latent heat of vaporization for sea-water in J/kg
vkarmn            = 0.4      #Von Karman's constant
Pi                = 3.141592654
twoPi             = 2.*Pi
eps_w             = 0.987    #emissivity of water
sigma0            = 5.67e-8  #Stefan Boltzman constant
oce_alb0          = 0.066    #Default sea surface albedo over ocean when nothing better is available
                             #NEMO: 0.066 / ECMWF: 0.055
Tswf              = 273      #BAD!!! because sea-ice not used yet!!!
#Tswf              = 271.4    #freezing point of sea-water (K)
to_red            = Pi/180.0
R_earth           = 6.3781e6   #Earth radius (m)
rtilt_earth       = 23.5
Sol0              = 1366     #Solar constant W/m^2
tdmn              = [ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ]
tdml              = [ 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ]

