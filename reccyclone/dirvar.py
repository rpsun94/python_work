def dirvarvort( V ) :
#Define the threshold value alpha (in degree) of the cyclone tracking direction variation
#according to the cyclone movement speed V (in m/s)
    if V <= 10:
        alpha = 180
    elif 15 >= V > 10 :
        alpha = 140
    elif 20 >= V > 15 :
        alpha = 100
    elif 25 >= V > 20 :
        alpha = 80
    elif 30 >= V > 25 :
        alpha = 60
    elif 35 >= V > 30 :
        alpha = 50
    elif 45 >= V > 35 :
        alpha = 40
    elif 50 >= V > 45 :
        alpha=30
    elif V > 50 :
        #print('ValueError : please check the speed ' )
        alpha=0
    return alpha
