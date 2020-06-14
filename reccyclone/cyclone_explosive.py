def cyclone_explosive(pres_list, lat_list, lon_list, dt, wind_list=[]):
    '''
    A cyclone can be called an explosive cyclone when following conditions
    are fit:
        1. pressure of center reduces at least 1 Bergeron(1hPa/h) in 12h;
        2. duration of the cyclone should be longer than 24h;
        3. and the maximum of wind speed around the center should be greater
           than 17.2m/s
    pres_list: center pressure, 1-D array-like, unit: hPa
    wind_list: maximum wind speed around the center, 1-D array-like, unit: m/s
    dt: derta t between two times, unit: hour
    standard_lat: standard lat for cauculating deepening rate.
                  50 deg for 90W-90E (A region);
                  45 deg for 90E-90W (P region).
    deepening rate =
    (p[t-6]-p[t+6])/12*(sin(standard_lat)/sin(0.5*(lat[t-6]+lat[t+6])))
    '''
    import numpy
    import numba
    # @numba.jit(nopython=True)
    def calculate_deepening_rate(pres_list, lat_list, st_list, dt):
        n = len(pres_list)
        deepening_rate = numpy.ones((n))-1
        for i in numba.prange(1, n-1):
            deepening_rate[i] = ((pres_list[i-1]-pres_list[i+1])/(2*dt)
                                 *numpy.absolute(numpy.sin(st_list[i]/180*Pi)
                                 /numpy.sin(max(5, numpy.absolute(0.5*(lat_list[i-1]
                                                 +lat_list[i+1])))/180*Pi)))
        return deepening_rate
    # judge wind speed
    if len(wind_list) == 0:
        condition1 = True
    elif max(wind_list) >= 17.2:
        condition1 = True
    elif max(wind_list) < 17.2:
        condition1 = False
    # judge duration
    if (len(pres_list)-1)*dt < 24:
        condition2 = False
    elif (len(pres_list)-1)*dt >= 24:
        condition2 = True
    # calculate deepening rate
    pres_list = numpy.array(pres_list)
    lat_list = numpy.array(lat_list)
    lon_list = numpy.array(lon_list)
    st_list = numpy.ones(lon_list.shape)-1
    for index, lon in numpy.ndenumerate(lon_list):
        if lon < 0:
            lon_list[index] += 360
        elif lon > 360:
            lon_list[index] -= 360
    for index, lon in numpy.ndenumerate(lon_list):
        if 270 > lon >= 90:
            st_list[index] = 45
        elif 360 >= lon >= 270 or 0 <= lon < 90:
            st_list[index] = 50
    Pi = 3.14159265
    deepening_rate = calculate_deepening_rate(pres_list, lat_list, st_list, dt)
    # judge deepening rate
    if max(deepening_rate) > 1:
        condition3 = True
    else:
        condition3 = False
    logic = condition1 and condition2 and condition3

    return logic, deepening_rate


def get_wind_online(lat, lon, derta, date, logic=False):
    # derta: (deg)
    #
    from ecmwfapi import ECMWFDataServer
    import numpy
    import netCDF4 as nc
    if lon > 180:
        lon -= 360
    N = min(lat + derta, 90)
    S = max(lat - derta, -90)
    W = lon - derta
    E = lon + derta
    if W < -180:
        W += 360
    if E > 180:
        E -= 360
    area = str(N)+'/'+str(W)+'/'+str(S)+'/'+str(E)
    iy, im, ida, ih = date
    # get_data:
    fmty = '%04d'
    fmtm = '%02d'
    day = str(fmty%(iy))+'-'+str(fmtm%(im))+'-'+str(fmtm%(ida))
    hour = str(fmtm%(ih))
    target = 'prepare/temp_uv/temp.nc'
    server = ECMWFDataServer()
    server.retrieve({
        "class": "ei",
        "dataset": "interim",
        "date": day,
        "expver": "1",
        "grid": "0.5/0.5",
        "levtype": "sfc",
        "param": "165.128/166.128",
        "step": "0",
        "stream": "oper",
        "time": hour+":00:00",
        # "area": "N/W/S/E", W means west start, E means east end;
                             # e.g. 60W - 0 - 60E : W=-60, E=60;
                              # or  60E- 180 - 60w: W=60, E=-60
        "area": area,
        "type": "an",
        "format": "netcdf",
        "target": target,
            })
    data = nc.Dataset(target)
    u10 = data.variables['u10'][:]
    v10 = data.variables['v10'][:]
    la = data.variables['latitude'][:]
    lo = data.variables['longitude'][:]
    wsp = (u10**2+v10**2)**0.5

    if logic:
        return numpy.max(wsp), u10, v10, la, lo
    elif not logic:
        return numpy.max(wsp)


def get_wind(lat, lon, derta, date, logic=False):
    # derta: (deg)
    #
    from demo_function import datelist
    import numpy
    import netCDF4 as nc
    N = min(lat + derta, 90)
    S = max(lat - derta, -90)
    W = max(lon - derta, 0)
    E = min(lon + derta, 359.5)
    # if W>360:
    #     W = W-360
    # elif W<0:
    #     W = W+360
    # if E>360:
    #     E = E-360
    # elif E<0:
    #     E = E+360
    iy, im, ida, ih = date
    iy_ = iy
    im_ = im+1
    if im_>12:
        iy_ = iy+1
        im_ = 1
    # get_data:
    fmty = '%04d'
    fmtm = '%02d'
    filename = str(fmty%(iy))+str(fmtm%(im))+'.nc'
    date_list = datelist((int(iy), int(im), 1, 0), (int(iy_), int(im_), 1, 0),
                         step='hours=6', remodel='date')
    date_strs = [fmty%item.year+fmtm%item.month+fmtm%item.day+fmtm%item.hour
                 for item in date_list]
    which = date_strs.index(fmty%iy+fmtm%im+fmtm%ida+fmtm%ih)
    data = nc.Dataset(r'D:/data/era_int_uv10_6h_05/'+filename)
    llo = data.variables['longitude'][:]
    lla = data.variables['latitude'][:]
    los = numpy.where(llo == W)[0][0]
    loe = numpy.where(llo == E)[0][0]+1
    las = numpy.where(lla == N)[0][0]
    lae = numpy.where(lla == S)[0][0]+1
    u10 = data.variables['u10'][which, las:lae, los:loe]
    v10 = data.variables['v10'][which, las:lae, los:loe]
    la = data.variables['latitude'][las:lae]
    lo = data.variables['longitude'][los:loe]
    wsp = (u10**2+v10**2)**0.5
    data.close()
    if logic:
        return numpy.max(wsp), u10, v10, la, lo
    elif not logic:
        return numpy.max(wsp)

def main():
    import json
    from demo_function import paint_figure
    path = r'prepare/analyse'
    f_n  = 'cyclone_track2019.json'
    fid  = open(path+'/'+f_n)
    dic  = json.load(fid)
    for ii in range(100000):
        lat = dic['lattt'][ii][0]
        if lat > 30:
            print(lat)
            lon = dic['lontt'][ii][0]
            iy = dic['yeartt'][ii][0]
            im = dic['monthtt'][ii][0]
            ida = dic['daytt'][ii][0]
            ih = dic['hourtt'][ii][0]
            wsp, u10, v10, la, lo = get_wind(lat, lon, 5, (iy, im, ida, ih), logic=True)
            basemap_option = {
                'projection': 'npstere',
                'boundinglat': 30,
                'lon_0': 180,
                # 'resolution':'l',
                'round': True,
            }
            quiver = True
            data = u10+1j*v10
            # print(data)
            data_quiver = {
                '0':{
                    'data': data,
                    'standard_wind': 0.25,
                    }
                }
            # print(data.shape, len(lo), len(la))
            print(wsp)
            name = 'prepare/temp_uv/test.png'
            paint_figure(la, lo, basemap_option, name, save=True, quiver=quiver,
                         data_quiver=data_quiver)
            break


if __name__ == '__main__':
    main()
