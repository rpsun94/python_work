"""Do RSDttest"""


def RSDttest(ts,T,reversion=False,alpha=0.1):
    import numpy
    import json
    def get_t(l1,l2,reversion) :
        n = len(l1)
        m = len(l2)
        x1 = numpy.array(list(range(n)))
        x2 = numpy.array(list(range(m)))
        p1 = numpy.polyfit(x1,l1,1)
        p2 = numpy.polyfit(x2,l2,1)
        nn = numpy.sum([(i-(n+1)/2)**2 for i in range(1,n+1)])
        mm = numpy.sum([(i-(m+1)/2)**2 for i in range(1,m+1)])
        c = mm*nn/(mm+nn)
        freedom = n+m-4
        if reversion :
            freedom = n+m-4
        ll1 = numpy.polyval(p1,x1)
        ll2 = numpy.polyval(p2,x2)
        Sx2 = numpy.sum([(l1[i]-ll1[i])**2 for i in range(n)])
        Sy2 = numpy.sum([(l2[i]-ll2[i])**2 for i in range(m)])
        Sxy2= 1/c*1/freedom*(Sx2+Sy2)
        t = abs((p1[0]-p2[0])/Sxy2**0.5)
        return t,freedom,p1[0]-p2[0]
    path = r'ttest'
    control_file = open(path + '\\ttest.json')
    control_dic = json.load(control_file)
    ts = numpy.array(ts)
    tao = T-2
    nts = len(ts)
    numbers = control_dic['num'][:]
    possible_points = []
    possible_points2 = []
    final_points = []
    for ipoint in range(tao,nts-tao) :
        l1 = ts[ipoint-tao:ipoint+1]
        l2 = ts[ipoint:ipoint+tao+1]
        t,freedom,dertak = get_t(l1,l2,reversion)
        if freedom in numbers:
            pt = control_dic[str(float(freedom))]
        elif freedom not in numbers:
            for i in range(len(numbers) - 2):
                if (freedom > numbers[i] and freedom < numbers[i + 1]):
                    pt = control_dic[str(numbers[i + 1])]
            if freedom >= 1000:
                pt = control_dic[str(float(1000))]
        t_st = pt[str(alpha)]
        if t>t_st :
            possible_points.append([ipoint,abs(dertak)])
    loop_list = list(range(len(possible_points)))
    for i in loop_list:
        points_order = [item[0] for item in possible_points]
        st_order = points_order[i]
        if st_order+1 not in points_order :
            possible_points2.append(int(st_order))
        elif st_order+1 in points_order :
            cold_area = [[st_order,possible_points[i][1]]]
            while (st_order+1) in points_order :
                st_order+=1
                i+=1
                loop_list.remove(i)
                cold_area.append([st_order,possible_points[i][1]])
            #print(cold_area,i)
            cold_area = numpy.array(cold_area)
            order = cold_area[:,1].argsort()
            #print(cold_area[order[-1],0])
            which = cold_area[order[-1],0]
            possible_points2.append(int(which))
    #print(possible_points2)
    divided_points = possible_points2.copy()
    divided_points.insert(0,0)
    divided_points.append(len(ts))
    #print(possible_points)
    #print(divided_points)
    for ipoint in range(len(possible_points2)) :
        st = divided_points[ipoint]
        bt = divided_points[ipoint+1]
        ed = divided_points[ipoint+2]
        #print(st,bt,ed)
        l1 = ts[st:bt+1]
        l2 = ts[bt:ed+1]
        t,freedom,dertak = get_t(l1,l2,reversion)
        t_st = pt[str(alpha)]
        if t>t_st :
            final_points.append(bt)
    return final_points
